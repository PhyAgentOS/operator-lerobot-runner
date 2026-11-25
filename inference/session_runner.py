"""Dora runner combining the LeRobot data plane and Forge Session endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

import pyarrow as pa
from forge_msgs import PolicyCommand, ToolMessage
from forge_policy import (
    PolicyRuntimeState,
    apply_policy_command,
    build_policy_observation,
)
from forge_tool import ToolEndpointHandler, ToolEnvelope
from forge_tool.dora import (
    DoraToolEndpointBinding,
    tool_envelope_to_message,
    tool_message_to_envelope,
)
from lerobot_inference.inference.endpoint_lease import EndpointLease
from lerobot_inference.inference.session_endpoint import (
    DESCRIPTOR,
    OPERATION,
    LeRobotServeSessionEndpoint,
)

logger = logging.getLogger(__name__)
ActionBuilder = Callable[[Any], Any]


class LeRobotSessionPolicyRunner:
    def __init__(
        self,
        node: Any,
        *,
        policy: Any,
        endpoint: LeRobotServeSessionEndpoint,
        joint_order: list[str],
        image_input_id_to_alias: dict[str, str],
        build_action: ActionBuilder,
        policy_id: str = "default",
        alias_for_cameras: list[str] | None = None,
        auto_start: bool = False,
        emit_command_status: bool = True,
        call_lifecycle_hooks: bool = True,
        endpoint_instance_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        action_trigger: str = "tick",
    ) -> None:
        if action_trigger not in ("tick", "observation"):
            raise ValueError(f"unknown action_trigger {action_trigger!r}")
        self.action_trigger = action_trigger
        # observation 触发：自上次出动作以来新到达的观测输入 ID。
        self._fresh_inputs: set[str] = set()
        self.node = node
        self.policy = policy
        self.endpoint = endpoint
        self.joint_order = joint_order
        self.image_input_id_to_alias = image_input_id_to_alias
        self.build_action = build_action
        self.camera_aliases = alias_for_cameras or list(image_input_id_to_alias.values())
        self.emit_command_status = emit_command_status
        self.call_lifecycle_hooks = call_lifecycle_hooks
        self.endpoint_instance_id = endpoint_instance_id or str(uuid.uuid4())
        self.clock = clock
        self._async = asyncio.Runner()
        self.handler = ToolEndpointHandler(
            DESCRIPTOR,
            endpoint_instance_id=self.endpoint_instance_id,
            operations={OPERATION: endpoint},
        )
        self.binding = DoraToolEndpointBinding(self.handler, event_sink=self._send_batch)
        self.lease = EndpointLease(DESCRIPTOR, self.endpoint_instance_id)
        self.state = PolicyRuntimeState(
            policy_id=policy_id,
            phase="running" if auto_start else "idle",
        )
        self.cached_proprio: Any | None = None
        self.cached_images: dict[str, Any] = {}
        endpoint.bind_stop_hook(self._stop_session)

    async def _send_batch(self, batch: pa.RecordBatch) -> None:
        self.node.send_output("tool_out", batch)

    def _send_envelope(self, envelope: ToolEnvelope) -> None:
        self.node.send_output("tool_out", tool_envelope_to_message(envelope).to_arrow())

    def _send_command_status(self, result: Any) -> None:
        try:
            self.node.send_output(
                "policy_command_status",
                result.to_status(self.state.policy_id).to_arrow(),
            )
        except Exception:
            logger.exception("failed to send policy_command_status")

    def _drive_lease(self) -> None:
        request = self.lease.poll(self.clock())
        if request is not None:
            self._send_envelope(request)

    def _observation_complete(self) -> bool:
        return "proprio_state" in self._fresh_inputs and all(
            input_id in self._fresh_inputs for input_id in self.image_input_id_to_alias
        )

    def _handle_tool(self, value: object) -> None:
        message = ToolMessage.from_arrow(value)
        envelope = tool_message_to_envelope(message)
        if envelope.message_type == "endpoint.registry.response":
            self.lease.acknowledge(envelope, self.clock())
            return
        for batch in self._async.run(self.binding.dispatch_input(value)):
            self.node.send_output("tool_out", batch)

    def _stop_session(self) -> None:
        self.state.phase = "idle"
        self.cached_proprio = None
        self.cached_images.clear()
        self._fresh_inputs.clear()
        stop = getattr(self.policy, "stop", None)
        if callable(stop):
            stop()

    def run(self, events: Iterable[dict[str, Any]]) -> int:
        self._drive_lease()
        for event in events:
            result = self.process_event(event)
            if result is not None:
                return result
        return 0

    def process_event(self, event: dict[str, Any]) -> int | None:
        event_type = event.get("type")
        self._drive_lease()
        if event_type == "STOP":
            unregister = self.lease.stop()
            if unregister is not None:
                self._send_envelope(unregister)
            try:
                self._async.run(self.endpoint.shutdown())
            finally:
                self._async.close()
            return 0
        if event_type == "ERROR":
            logger.error("Dora node error: %s", event.get("error", "unknown"))
            self._async.close()
            return 1
        if event_type != "INPUT":
            return None

        input_id = event.get("id")
        value = event.get("value")
        try:
            if input_id == "tool_in" and value is not None:
                self._handle_tool(value)
                return None
            if input_id == "policy_command" and value is not None:
                command = PolicyCommand.from_arrow(value)
                result = apply_policy_command(
                    state=self.state,
                    policy=self.policy,
                    command=command,
                    call_lifecycle_hooks=self.call_lifecycle_hooks,
                )
                if result is None:
                    return None
                if result.reset_observation_cache:
                    self.cached_proprio = None
                    self.cached_images.clear()
                    self._fresh_inputs.clear()
                if self.emit_command_status:
                    self._send_command_status(result)
                return None
            if input_id == "proprio_state" and value is not None:
                self.cached_proprio = value
                self._fresh_inputs.add("proprio_state")
                return None
            if input_id in self.image_input_id_to_alias and value is not None:
                self.cached_images[input_id] = value
                self._fresh_inputs.add(input_id)
                return None
            if input_id != "tick" or not self.state.is_running:
                return None
            if self.action_trigger == "observation" and not self._observation_complete():
                return None

            if self.policy.is_observation_needed():
                observation = build_policy_observation(
                    proprio_payload=self.cached_proprio,
                    image_payloads=self.cached_images,
                    joint_order=self.joint_order,
                    image_input_id_to_alias=self.image_input_id_to_alias,
                )
                if observation is None:
                    return None
            else:
                observation = {}
            action_payload = self.policy.generate_action(observation, self.camera_aliases)
            if action_payload is not None:
                self.node.send_output("action", self.build_action(action_payload).to_arrow())
                self._fresh_inputs.clear()
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("ignored invalid policy input: %s", exc)
        except Exception:
            logger.exception("policy tick failed")
        return None


__all__ = ["LeRobotSessionPolicyRunner"]
