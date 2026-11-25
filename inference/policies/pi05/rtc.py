"""Asynchronous LeRobot Real-Time Chunking adapter for PI0.5."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import torch
from lerobot.policies.rtc import (
    ActionQueue,
    LatencyTracker,
    RTCConfig,
    reanchor_relative_rtc_prefix,
)
from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep
from lerobot_inference.inference.observation import action_tensor_to_numpy

from .sync import PI05PolicyAdapter

logger = logging.getLogger(__name__)

_IDLE_SLEEP_S = 0.01
_JOIN_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class _QueueSnapshot:
    epoch: int
    action_index: int
    original_leftover: torch.Tensor | None
    processed_leftover: torch.Tensor | None


class _RTCActionQueue(ActionQueue):
    """ActionQueue with atomic snapshot/merge and reset epochs."""

    def __init__(self, cfg: RTCConfig) -> None:
        super().__init__(cfg)
        self._epoch = 0

    def clear(self) -> None:
        # lerobot>=0.7 的基类 clear 还会清空 task/provenance 标签队列和游标。
        super().clear()
        with self.lock:
            self._epoch += 1

    def snapshot(self) -> _QueueSnapshot:
        with self.lock:
            original = (
                self.original_queue[self.last_index :].clone()
                if self.original_queue is not None
                and self.last_index < len(self.original_queue)
                else None
            )
            processed = (
                self.queue[self.last_index :].clone()
                if self.queue is not None and self.last_index < len(self.queue)
                else None
            )
            return _QueueSnapshot(
                epoch=self._epoch,
                action_index=self.last_index,
                original_leftover=original,
                processed_leftover=processed,
            )

    def merge_snapshot(
        self,
        original_actions: torch.Tensor,
        processed_actions: torch.Tensor,
        snapshot: _QueueSnapshot,
    ) -> int | None:
        """Merge using actions actually consumed since an atomic snapshot."""
        with self.lock:
            if snapshot.epoch != self._epoch:
                return None
            consumed_during_inference = max(0, self.last_index - snapshot.action_index)
            # lerobot>=0.7：每个动作带生成它的 task 标签，并与 provenance 队列逐位对齐。
            # 指令切换由适配器 reset 处理，这里不打标签；provenance 清空以保持长度一致。
            self._replace_actions_queue(
                original_actions,
                processed_actions,
                consumed_during_inference,
                None,
            )
            self._provenance_queue = []
            self._epoch += 1
            return consumed_during_inference


def _normalize_prev_actions_length(
    prev_actions: torch.Tensor,
    target_steps: int,
) -> torch.Tensor:
    if prev_actions.ndim != 2:
        raise ValueError(
            f"Expected RTC previous actions with shape [T, A], got {tuple(prev_actions.shape)}"
        )
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros(
        (target_steps, action_dim),
        dtype=prev_actions.dtype,
        device=prev_actions.device,
    )
    padded[:steps] = prev_actions
    return padded


class PI05AsyncRTCPolicyAdapter(PI05PolicyAdapter):
    """Non-blocking PI0.5 inference backed by LeRobot RTC action merging."""

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        *,
        instruction: str = "",
        expected_image_keys: set[str],
        control_hz: float = 50.0,
        queue_threshold: int = 30,
    ) -> None:
        super().__init__(
            policy,
            preprocessor,
            postprocessor,
            instruction=instruction,
            expected_image_keys=expected_image_keys,
        )
        rtc_config = getattr(policy.config, "rtc_config", None)
        if rtc_config is None or not bool(getattr(rtc_config, "enabled", False)):
            raise ValueError("PI0.5 async_rtc requires an enabled rtc_config")
        if not isinstance(rtc_config, RTCConfig):
            raise TypeError(f"Expected LeRobot RTCConfig, got {type(rtc_config).__name__}")
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be finite and positive, got {control_hz}")
        if bool(getattr(policy.config, "compile_model", False)):
            raise ValueError(
                "compile_model=true is not supported by async_rtc without explicit warmup"
            )
        if isinstance(queue_threshold, bool) or not 0 <= queue_threshold <= int(
            policy.config.chunk_size
        ):
            raise ValueError(
                "rtc.queue_threshold must satisfy "
                f"0 <= threshold <= chunk_size, got {queue_threshold}"
            )
        if not 1 <= rtc_config.execution_horizon <= int(policy.config.chunk_size):
            raise ValueError(
                "rtc.execution_horizon must satisfy "
                "1 <= execution_horizon <= chunk_size, "
                f"got {rtc_config.execution_horizon}"
            )

        self._rtc_config = rtc_config
        self._control_hz = float(control_hz)
        self._queue_threshold = int(queue_threshold)
        self._action_queue = _RTCActionQueue(rtc_config)
        self._latency_tracker = LatencyTracker()

        self._observation_lock = Lock()
        self._inference_lock = Lock()
        self._latest_observation: dict[str, np.ndarray] | None = None
        self._generation = 0
        self._error: Exception | None = None
        self._stopping = False
        self._needs_reset = False

        self._active = Event()
        self._shutdown = Event()
        self._thread: Thread | None = None

        self._relative_step = next(
            (
                step
                for step in preprocessor.steps
                if isinstance(step, RelativeActionsProcessorStep) and step.enabled
            ),
            None,
        )
        self._normalizer_step = next(
            (step for step in preprocessor.steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        *,
        tokenizer_path: str,
        device: str | None = None,
        instruction: str = "",
        expected_image_keys: set[str] | None = None,
        policy_config_overrides: dict[str, Any] | None = None,
        control_hz: float = 50.0,
        queue_threshold: int = 30,
    ) -> PI05AsyncRTCPolicyAdapter:
        loaded = PI05PolicyAdapter.from_pretrained(
            pretrained_path,
            tokenizer_path=tokenizer_path,
            device=device,
            instruction=instruction,
            expected_image_keys=expected_image_keys,
            policy_config_overrides=policy_config_overrides,
            allow_rtc=True,
        )
        return cls(
            loaded._policy,
            loaded._preprocessor,
            loaded._postprocessor,
            instruction=instruction,
            expected_image_keys=loaded._expected_image_keys,
            control_hz=control_hz,
            queue_threshold=queue_threshold,
        )

    @property
    def failed(self) -> bool:
        return self._error is not None



    def _raise_if_failed(self) -> None:
        with self._observation_lock:
            error = self._error
        if error is not None:
            raise SystemExit("PI0.5 async RTC inference failed") from error

    def start(self) -> None:
        self._raise_if_failed()
        if self._needs_reset:
            self.reset()
        if self._stopping:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("PI0.5 async RTC backend is still stopping")
            self._thread = None
            self._stopping = False
            self.reset()
        if self._thread is not None and self._thread.is_alive():
            if self._shutdown.is_set():
                raise RuntimeError("PI0.5 async RTC backend is shutting down")
            self._active.set()
            return
        self._shutdown.clear()
        self._active.set()
        self._thread = Thread(
            target=self._inference_loop,
            daemon=True,
            name="PI05AsyncRTC",
        )
        self._thread.start()
        logger.info("PI0.5 async RTC thread started")

    def pause(self) -> None:
        # Invalidate observations and executable actions immediately without
        # blocking Dora on an in-flight model call. Processors reset on resume.
        self._active.clear()
        with self._observation_lock:
            self._generation += 1
            self._latest_observation = None
        self._action_queue.clear()
        self._needs_reset = True

    def stop(self) -> None:
        self._active.clear()
        self._shutdown.set()
        self._stopping = True
        with self._observation_lock:
            self._generation += 1
            self._latest_observation = None
        self._action_queue.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    "PI0.5 async RTC thread did not stop within %.1fs",
                    _JOIN_TIMEOUT_S,
                )
            else:
                logger.info("PI0.5 async RTC thread stopped")
        if self._thread is None or not self._thread.is_alive():
            self._thread = None
            self._stopping = False
            self.reset()

    def reset(self) -> None:
        with self._observation_lock:
            self._generation += 1
            self._latest_observation = None
            self._error = None
        # Wait for any in-flight inference/merge, then perform a final clear so
        # no pre-reset generation can repopulate the executable queue.
        with self._inference_lock:
            self._action_queue.clear()
            self._latency_tracker.reset()
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()
            self._needs_reset = False

    def is_observation_needed(self) -> bool:
        # RTC continuously consumes the newest observation while actions are executed.
        return True

    def generate_action(
        self,
        observation: dict[str, Any],
        alias_for_cameras: list[str] | None = None,
    ) -> np.ndarray | None:
        del alias_for_cameras
        if not observation:
            raise ValueError("PI0.5 async_rtc requires an observation on every tick")
        self._raise_if_failed()
        copied = self._copy_observation(observation)
        self.start()
        with self._observation_lock:
            self._latest_observation = copied

        action = self._action_queue.get()
        if action is None:
            return None
        return action_tensor_to_numpy(action)

    def _snapshot_observation(self) -> tuple[dict[str, np.ndarray] | None, int]:
        with self._observation_lock:
            return self._latest_observation, self._generation

    def _inference_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                if not self._active.is_set():
                    self._shutdown.wait(_IDLE_SLEEP_S)
                    continue
                if self._action_queue.qsize() > self._queue_threshold:
                    self._shutdown.wait(_IDLE_SLEEP_S)
                    continue

                observation, generation = self._snapshot_observation()
                if observation is None:
                    self._shutdown.wait(_IDLE_SLEEP_S)
                    continue
                try:
                    self._infer_and_merge(observation, generation)
                except Exception as exc:  # noqa: BLE001 - worker error boundary
                    if self._record_failure(exc, generation):
                        return
                    logger.info("Ignoring PI0.5 RTC error from an invalidated generation")
        except Exception as exc:  # noqa: BLE001 - worker error boundary
            self._record_failure(exc, None)

    def _record_failure(
        self,
        error: Exception,
        generation: int | None,
    ) -> bool:
        with self._observation_lock:
            if self._shutdown.is_set():
                return False
            if generation is not None and generation != self._generation:
                return False
            self._error = error
        logger.error(
            "PI0.5 async RTC inference failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        self._action_queue.clear()
        self._active.clear()
        return True

    def _infer_and_merge(
        self,
        observation: dict[str, np.ndarray],
        generation: int,
    ) -> None:
        with self._inference_lock:
            started_at = time.perf_counter()
            queue_snapshot = self._action_queue.snapshot()
            previous_actions = queue_snapshot.original_leftover
            previous_processed_actions = queue_snapshot.processed_leftover
            prior_latency = self._latency_tracker.max() or 0.0
            predicted_delay = math.ceil(prior_latency * self._control_hz)

            batch = self._prepare_observation(observation)
            # RTC prefix guidance temporarily re-enables autograd inside LeRobot's
            # denoiser, which torch.inference_mode() would make impossible.
            with torch.no_grad(), self._inference_autocast_context():
                preprocessed = self._preprocessor(batch)

                if previous_actions is not None and self._relative_step is not None:
                    current_state = self._relative_step.get_cached_state()
                    if (
                        current_state is not None
                        and previous_processed_actions is not None
                        and previous_processed_actions.numel() > 0
                    ):
                        previous_actions = reanchor_relative_rtc_prefix(
                            prev_actions_absolute=previous_processed_actions,
                            current_state=current_state,
                            relative_step=self._relative_step,
                            normalizer_step=self._normalizer_step,
                            policy_device=self._device,
                        )

                if previous_actions is not None:
                    previous_actions = _normalize_prev_actions_length(
                        previous_actions,
                        target_steps=self._rtc_config.execution_horizon,
                    )

                actions = self._policy.predict_action_chunk(
                    preprocessed,
                    inference_delay=predicted_delay,
                    prev_chunk_left_over=previous_actions,
                )
                original_actions = actions.squeeze(0).clone()
                processed_actions = self._postprocessor(actions).squeeze(0)

            latency = time.perf_counter() - started_at
            self._latency_tracker.add(latency)

            with self._observation_lock:
                if generation != self._generation or self._shutdown.is_set():
                    return
            consumed = self._action_queue.merge_snapshot(
                original_actions,
                processed_actions,
                queue_snapshot,
            )
            if consumed is None:
                return
            logger.debug(
                "PI0.5 RTC chunk merged: latency=%.3fs consumed=%d queue=%d",
                latency,
                consumed,
                self._action_queue.qsize(),
            )
