# LeRobot Inference

[English](README.md) | [简体中文](README_zh.md)

Convert legacy `policy_train` ACT artifacts to LeRobot checkpoints and run online inference through CLI or Dora Nodes.

```text
policy_train ACT ── convert ──> pretrained_model/ ── infer ──> Dora JointCommand
LeRobot checkpoint ──────────────────────────────────┘
```

Inference always loads the complete `pretrained_model/` directory (weights, configuration and processors), not individual weight/statistics files.

## Node, Endpoint and Tool API

| Item | Current implementation |
| --- | --- |
| Production Nodes | 1 inference Node: `lerobot infer`; onedir entrypoint `dist/lerobot_infer/lerobot_infer`, single-file `dist/onefile/lerobot_infer`; `infer-once` and `convert` are offline CLI utilities, not additional Nodes |
| Tool support | Optional provider with inference `mode: session_endpoint`; default `dora_plain` registers no Endpoint |
| Endpoints | 1 in provider mode: `policy.runner`; 0 in plain mode |
| Operations | 1 in provider mode: `serve` (Session, stoppable, status supported, maximum concurrency 1) |
| Direct HTTP Tool API | 0; Dora `tool_in/tool_out` connect to Gateway |
| Gateway Tools | Defined by Skill/Gateway `tools.specs`; no repository-fixed external Tool ID |

Invocation mapping: `<tool_id from tools.specs> -> policy.runner / serve -> lerobot infer`.
The model loads inside the Node; the Session manages the resident policy service, while data-plane
PolicyCommand messages continue to manage episode boundaries.

Session uses stop, not Action cancel. Invoke through Gateway `POST /tools/{tool_id}:invoke`,
then read invocation status/results; stop with `POST /invocations/{invocation_id}/stop`.
Gateway owns these HTTP routes; model inference service interfaces are not Tool APIs.

Kai0 and LeRobot share Endpoint ID `policy.runner`. A Gateway has one current instance for this ID:
select them as deployment alternatives. Different Node IDs alone do not create separate routes.

Implementation and activation: [descriptor](inference/session_endpoint.py), [runner](inference/session_runner.py),
[configuration validation](inference/config.py), [inference entrypoint](inference/main.py).
Current Gateway source implements Session; historical binaries may differ, so check paired revisions and locks.

Compatibility: current dependency declarations are `forge-msgs==1.2.0`, `forge-common==1.0.1`, `forge-policy==1.0.1`, `forge-tool[dora]==1.0.0`, `dora-rs==0.4.1`. See [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). These declarations do not guarantee cross-version interoperability; also check the build revision and Skill lock when using artifacts.

Changes to Node entrypoints, Endpoints or Tool APIs must update this section and the associated configuration/contracts. Follow the [Node README contract](https://github.com/PhyAgentOS/forge/blob/master/CONTRIBUTING.md#node-readme-contract).

## Quick start

Requirements: Linux x86_64, Python 3.12 and [uv](https://docs.astral.sh/uv/).
GPU inference requires an NVIDIA driver compatible with the locked stable CUDA 13 Torch wheel.
`uv.lock` fixes Torch, TorchVision and CUDA userspace dependencies; no nightly/cu128-only index
or hand-maintained NVIDIA wheel list is used. Dependencies resolve from public PyPI without internal repositories.

The current lock fixes `torch 2.11.0+cu130`, `torchvision 0.26.0+cu130`, `triton 3.6.0` and `numpy 2.2.6`,
aligned with the trainer's core versions. The repository reports CUDA matrix multiplication, source ACT
`infer-once` and PyInstaller ACT `infer-once` on RTX 5060 (compute capability 12.0).
The wheel includes `sm_75/sm_80/sm_86/sm_90/sm_100/sm_120`; RTX 30/40 series still require real-model
acceptance before being marked tested.

```bash
git clone https://github.com/Forgelab-Robotics/operator-lerobot-runner.git
cd lerobot_runner
uv sync --extra dev

uv run lerobot --help
uv run pytest
```

The main command is `lerobot`, with these subcommands:

```bash
uv run lerobot convert --help
uv run lerobot infer --help
uv run lerobot infer-once --help
```

Compatibility commands `lerobot-convert`, `lerobot-infer` and `lerobot-infer-once` remain supported.

## Repository layout

```text
.
├── cli.py                 # Unified CLI
├── common/                # Shared checkpoint and pretrained-asset handling
├── convert/               # policy_train ACT -> LeRobot checkpoint
├── inference/             # Dora and single-step inference
│   └── policies/          # ACT, Pi0.5, Diffusion and LingBot-VA adapters
├── config/                # Per-policy conversion/inference templates
├── examples/              # Conversion and 7D/14D Dora examples
├── scripts/               # Setup, testing and packaging helpers
├── tests/                 # Unit tests
└── docs/                  # Design and implementation guides
```

## Convert ACT weights

The source directory must contain at least:

```text
<src-dir>/
├── policy_epoch_*_standard.safetensors  # Current format, or one legacy policy.ckpt
└── dataset_stats.pkl
```

Tensor-only `.safetensors` and legacy `.ckpt` are supported. Legacy loading uses
`torch.load(weights_only=True)`, without executing arbitrary Python objects in checkpoints.
Provide the training `task.json` for camera names and ACT hyperparameters when available;
otherwise default left/right/top cameras and ACT parameters are used.

```bash
uv run lerobot convert --config examples/dora_convert/convert.yaml
```

Relative YAML paths resolve against the YAML directory. The conversion output can be used directly as `policy.pretrained_path`:

```text
<dst-dir>/
├── model.safetensors
├── config.json
├── policy_preprocessor.json
├── policy_postprocessor.json
└── conversion_meta.json
```

## Run inference

Each implemented policy has its own template:

| Configuration | Backend |
| --- | --- |
| `config/inference/act.yaml` | Synchronous/asynchronous ACT action chunks; optional temporal ensemble in synchronous mode |
| `config/inference/pi05.yaml` | PI0.5 synchronous `select_action()` |
| `config/inference/pi05_async_rtc.yaml` | PI0.5 asynchronous Real-Time Chunking |
| `config/inference/diffusion.yaml` | Diffusion Policy; observation history and action chunks managed inside the policy |
| `config/inference/lingbot_va.yaml` | LingBot-VA video-action world model; policy-managed chunks and KV cache |

See `config/inference/README.md` for complete details.

Inference configuration must satisfy:

1. `joints` order/count must match model action dimensions: 7 joints for 7D, 14 for 14D.
2. `joints` constructs `observation.state` by default; configure top-level `state_joints` if state dimensions/order differ from actions.
3. `image_inputs` keys may be arbitrary Dora image topics/input IDs; aliases match checkpoint `observation.images.<alias>`. ACT and PI0.5 use the intersection of runtime/checkpoint cameras, requiring at least one match and ignoring extra topics. Missing training views may reduce performance; PI0.5 supplies masked empty images for missing views.
4. `policy.pretrained_path` must reference a directory containing `model.safetensors`.
5. `policy.torch_compile` defaults off; if enabled, finish the first forward warmup before control begins.

```bash
# Single-step inference without starting Dora
uv run lerobot infer-once --config examples/dora_infer_act_14d/policy_act.yaml

# Hardware Dora example: adapt YAML to your devices and model first
cd examples/dora_infer_act_14d
dora run dataflow.yaml
```

See `examples/dora_infer_act_7d/` and `examples/dora_infer_act_14d/` for full dataflows and policy configuration.

ACT defaults to `inference_mode: sync`, preserving LeRobot `select_action()` behavior.
With `inference_mode: async_chunked`, a background thread runs `predict_action_chunk()`;
Dora ticks only dequeue without blocking. Low queue watermarks prefetch chunks and overlapping
timesteps merge with LeRobot asynchronous inference's default weights. Pause, stop and reset
invalidate the old in-flight generation. `temporal_ensemble_coeff` is sync-only and cannot
be combined with `async_chunked`.

Pi0.5 also requires a local tokenizer path and language instruction:

```yaml
policy:
  type: pi05
  pretrained_path: /path/to/pretrained_model
  tokenizer_path: /path/to/local-paligemma-tokenizer
  instruction: pick up the object and place it in the box
```

Pi0.5 has two explicit backends: default `inference_mode: sync` uses LeRobot's native
`select_action()`; `inference_mode: async_rtc` uses background `predict_action_chunk()`,
`LatencyTracker` and `ActionQueue.merge()` for nonblocking Real-Time Chunking.
It neither accesses private action queues nor uses the old queue-length frame-skipping heuristic.
Instruction changes, pause, stop and reset discard old chunks. Loading leaves the pretrained
directory read-only and fails on corrupt weights/configuration incompatibility instead of falling
back to randomly initialized weights. Full example: `examples/dora_sim_infer_pi05_7d_bin/`.

Diffusion Policy internally manages observation history (`n_obs_steps` stacking) and action chunks.
The adapter forwards the current observation without its own queue. Nonstandard historical image
keys, such as PushT `observation.image`, can be declared explicitly:

```yaml
policy:
  type: diffusion
  pretrained_path: /path/to/pretrained_model
  expected_image_keys: [observation.image]
```

LingBot-VA is a video-action world model with camera inputs and no `observation.state`.
`used_action_channel_ids` determines action dimensions (7 for LIBERO). Inference needs frozen
Wan VAE and UMT5 weights (`vae/`, `text_encoder/`, `tokenizer/`); for offline deployment point
`wan_pretrained_path` to a local directory. UMT5 defaults to CPU and encodes each episode's instruction once:

```yaml
policy:
  type: lingbot_va
  pretrained_path: /path/to/lingbot_va/pretrained_model
  wan_pretrained_path: /path/to/lingbot_va_base_frozen
  text_encoder_device: cpu
  instruction: pick up the black bowl and place it on the plate
```

## Model sources

Checkpoint and component sources are listed below. For offline use, download locally and pin revisions;
`huggingface-cli download <repo> --revision <hash>` selects an exact version.

| Policy | Checkpoint source | Components | License |
| --- | --- | --- | --- |
| ACT | User-supplied training output, converted to LeRobot format with `convert` (see above) | None | Check supplied model |
| Pi0.5 | User-supplied pi05 training output; set `pretrained_path` to the local directory | Local PALIGEMMA tokenizer | Check supplied model |
| Diffusion | Official LeRobot Hub, such as `lerobot/diffusion_pusht`, `lerobot/diffusion_policy_simultaneous_*` | None | Apache-2.0 |
| LingBot-VA | Official `lerobot/lingbot_va_libero_long` | Frozen Wan VAE + UMT5: `robbyant/lingbot-va-base` with `vae/`, `text_encoder/`, `tokenizer/` | See each HF repository |
| VLA-JEPA | Official `lerobot/VLA-JEPA-LIBERO` | Qwen3-VL backbone `Qwen/Qwen3-VL-2B-Instruct`; V-JEPA2 encoder `facebook/vjepa2-vitl-fpc64-256` (optional during inference) | Apache-2.0 / MIT |

> ACT/Pi0.5 use supplied training artifacts; the other policies load official LeRobot Hub checkpoints,
> downloaded on demand or referenced locally. VLA-JEPA source and pinned revisions are recorded in
> `examples/dora_sim_infer_vla_jepa_8d_bin/README.md`.

## Packaging

```bash
bash scripts/build.sh
```

PyInstaller writes `dist/lerobot_infer/lerobot_infer`:

```bash
dist/lerobot_infer/lerobot_infer infer --config /path/to/policy_act.yaml
```

Build caches/artifacts are Git-ignored. Rebuild cleanly with `bash scripts/build.sh --clean`.

### Optional single-file Node build

Forge/PAOS `executable_tar_gz` installation requires exactly one executable at the archive root,
with a filename matching the Node lock's `entrypoint` (`lerobot_infer`). The default onedir
output is a directory and does not meet this contract. Use the separate single-file build
when publishing a Node; the onedir flow remains available:

```bash
bash scripts/setup.sh                 # Initial uv environment setup
bash scripts/build_node_onefile.sh    # Output: dist/onefile/lerobot_infer
```

Create a Node archive (`tar -tzf` should print only `lerobot_infer`):

```bash
tar -czf lerobot_infer-<version>-linux-x86_64.tar.gz -C dist/onefile lerobot_infer
```

Notes:

- Dependencies match onedir: `uv sync --extra build`, or `scripts/build.sh` installs `pyinstaller>=6.0.0`.
- The default interpreter is repository `.venv/bin/python`; override with `PYTHON=/path/to/python`. It never silently falls back to system Python.
- Single-file output uses `dist/onefile/` and `build/pyinstaller/lerobot_infer_node/`, separate from onedir artifacts.
- Single-file startup unpacks dependencies, increasing startup time/archive size; use it for Node delivery where needed.
- CI builds onedir assets by default; single-file builds remain optional.

## License

This project uses [Apache License 2.0](LICENSE). Runtime/build dependency licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Versioned releases

**Mandatory tag-first release rule:** merge the version/lock change, create an annotated version tag on that reviewed commit, build that exact tag, validate the artifacts, then publish. Moving branch names, untagged commit IDs and replacement of published tags are not release inputs.

| Release line | Source | Node/package version | Runtime |
| --- | --- | --- | --- |
| Forge 1.x | `stable/1.x` | `1.x`, tag `v1.X.Y` | Forge 1.x / Dora 0.4.x |
| Forge 2.x | `master` | `2.x`, tag `v2.X.Y` | Forge 2.x / Dora 1.x |

**Legacy normalization:** nodes previously at 0.x on `stable/1.x` release as **1.0.0**, retaining Forge 1.x / Dora 0.4.x dependencies and runtime behavior. Existing 1.x package versions are retained. Node/package majors now follow the Forge compatibility line; a 0.x tag is rejected.

The default branch still uses Forge 1.x; its 2.x migration must merge before a 2.x release can pass validation.

Run **Build versioned release** from `master` for either line. First merge the version/lock changes into the selected source branch, then create an unused **annotated** tag on that reviewed commit. The tag must match package metadata. Never move existing tags or reuse a published version. Dispatch uses the workflow's reviewed tooling with the exact tagged source, so legacy tags need not contain the new release helper.

```bash
# Replace with an existing, reviewed and unpublished tag matching source metadata.
TAG=v1.X.Y
gh workflow run release.yml --ref master -f release_line=1.x -f tag="$TAG"
TAG=v2.X.Y
gh workflow run release.yml --ref master -f release_line=2.x -f tag="$TAG"
```

The workflow verifies tag type, source commit, branch ancestry, source version and locked Forge/Dora compatibility. It builds source archives and wheel/sdist where supported, checks Python distribution metadata and SHA-256, and creates or attaches to a **draft GitHub Release**. Published releases/assets are not overwritten; review all build results and acceptance evidence before publishing. This workflow does not publish node packages to PyPI.

Set `build_binaries=true` to call the existing Ubuntu 20.04 binary workflow; its existing architecture, CLI and archive checks remain in force. You can also run `build-ubuntu20.yml` directly with `ref`, `version` and `release_line`; `ref` must be an annotated tag, not a moving branch. Binary artifacts are attached only after all selected jobs succeed.

The source archive preserves the tagged tree; external SDKs/submodules, model downloads and site configuration remain separate prerequisites. A dependency or SDK build failure is not successful release acceptance. Build success does not establish single-node or Gateway-chain `dora run`, hardware, real-model, sustained-run or rollback acceptance. Every acceptance chain must include `adapter-forge-gateway` from the same compatibility line. Forge 2.x is a **breaking change**: align Forge libraries, node versions and Dora across the entire deployment; keep 1.x on its maintenance branch.
