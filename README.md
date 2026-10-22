# LeRobot Inference

将旧版 `policy_train` 的 ACT 产物转换为 LeRobot checkpoint，并以 CLI 或 Dora 节点运行在线推理。

```text
policy_train ACT ── convert ──> pretrained_model/ ── infer ──> Dora JointCommand
LeRobot checkpoint ──────────────────────────────────┘
```

推理始终加载完整的 `pretrained_model/` 目录（权重、配置和 processor），而不是单独的权重或统计文件。

## Node、Endpoint 与 Tool API / Capability summary

| 项目 | 当前实现 |
| --- | --- |
| 生产 Node | 1 个推理 Node：`lerobot infer`；onedir 入口 `dist/lerobot_infer/lerobot_infer`，单文件 `dist/onefile/lerobot_infer`；`infer-once`、`convert` 是离线 CLI，不另计 Node |
| Tool 支持 | 可选 provider 模式：推理配置 `mode: session_endpoint`；默认 `dora_plain` 不注册 Endpoint |
| Endpoint 数 | provider 模式为 1：`policy.runner`；plain 模式为 0 |
| operation 数 | provider 模式为 1：`serve`（Session，可 stop、可查状态，最大并发 1） |
| 直接 HTTP Tool API | 0；`tool_in/tool_out` 通过 Dora 接入 Gateway |
| Gateway Tool 数 | 由 Skill/Gateway `tools.specs` 决定，仓库不固定外部 Tool ID |

调用映射为 `<tool_id from tools.specs> -> policy.runner / serve -> lerobot infer`。
模型在 Node 内加载；会话管理常驻策略服务，episode 边界仍由数据平面的 PolicyCommand 管理。

Session 使用 stop，不应替换成 Action cancel。对外调用通过 Gateway
`POST /tools/{tool_id}:invoke`，随后读取 invocation 状态/结果；停止使用
`POST /invocations/{invocation_id}/stop`。HTTP 路由属于 Gateway，Node 内的推理服务接口不算 Tool API。

Kai0 与 LeRobot 使用相同 `policy.runner` Endpoint ID；同一 Gateway 中该 ID 只有一个 current instance，
两者应作为部署候选选择，不能仅用不同 Node ID 就假定能同时成为独立路由。

实现与启用条件：[descriptor](inference/session_endpoint.py)、[runner](inference/session_runner.py)、
[配置校验](inference/config.py)、[推理入口](inference/main.py)。
Gateway 当前源码实现 Session，但历史二进制可能不同，需核对配套版本与依赖锁。

兼容性：当前依赖声明为 `forge-msgs==1.2.0`、`forge-common==1.0.1`、`forge-policy==1.0.1`、`forge-tool[dora]==1.0.0`、`dora-rs==0.4.1`；以 [pyproject.toml](pyproject.toml) 与 [uv.lock](uv.lock) 为准。 这些声明不构成跨版本互操作保证；使用制品时还需核对 build revision 和 Skill lock。

后续修改 Node 入口、Endpoint 或 Tool API 时，请同步更新本节及对应配置/契约，遵循 [Node README 规范](https://github.com/PhyAgentOS/forge/blob/master/CONTRIBUTING.md#node-readme-contract)。

## 快速开始

要求：Linux x86_64、Python 3.12、[uv](https://docs.astral.sh/uv/)；GPU 推理需要兼容 lock 中稳定版 CUDA 13 Torch wheel 的 NVIDIA 驱动。Torch、TorchVision 和 CUDA 用户态依赖由 `uv.lock` 唯一固定，不再使用 nightly/cu128 专用索引或手写 NVIDIA wheel 列表。所有依赖均从公共 PyPI 索引解析，无需访问内部仓库。

当前 lock 固定 `torch 2.11.0+cu130`、`torchvision 0.26.0+cu130`、`triton 3.6.0` 和 `numpy 2.2.6`，与 `lerobot_trainer` 的核心版本对齐。RTX 5060（compute capability 12.0）已通过 CUDA matrix multiplication、源码 ACT `infer-once` 和 PyInstaller ACT `infer-once`；Torch wheel 包含 `sm_75/sm_80/sm_86/sm_90/sm_100/sm_120`，但 RTX 30/40 系列仍需分别完成真实模型验收后才能标记为 tested。

```bash
git clone https://github.com/Forgelab-Robotics/operator-lerobot-runner.git
cd lerobot_runner
uv sync --extra dev

uv run lerobot --help
uv run pytest
```

主命令为 `lerobot`，子命令：

```bash
uv run lerobot convert --help
uv run lerobot infer --help
uv run lerobot infer-once --help
```

兼容命令 `lerobot-convert`、`lerobot-infer` 和 `lerobot-infer-once` 仍可使用。

## 项目结构

```text
.
├── cli.py                 # 统一 CLI 入口
├── common/                # checkpoint 与预训练资产共用逻辑
├── convert/               # policy_train ACT → LeRobot checkpoint
├── inference/             # Dora 与单次推理实现
│   └── policies/          # ACT、Pi0.5、Diffusion、LingBot-VA 策略适配器
├── config/                # 按策略拆分的转换与推理配置模板
├── examples/              # 转换、7D 与 14D Dora 示例
├── scripts/               # 安装、测试、打包辅助脚本
├── tests/                 # 单元测试
└── docs/                  # 设计与实现说明
```

## 转换 ACT 权重

源目录至少应包含：

```text
<src-dir>/
├── policy_epoch_*_standard.safetensors  # 新版，或单个旧版 policy.ckpt
└── dataset_stats.pkl
```

支持 tensor-only `.safetensors` 和旧版 `.ckpt`；legacy `.ckpt` 使用 `torch.load(weights_only=True)`，不会执行 checkpoint 中的任意 Python 对象。`task.json` 用于相机名称和 ACT 超参数，推荐提供训练时使用的文件；省略时使用默认的 left/right/top 三相机与 ACT 参数。

```bash
uv run lerobot convert --config examples/dora_convert/convert.yaml
```

YAML 中的相对路径相对于 YAML 文件所在目录。转换输出目录可直接用作 `policy.pretrained_path`：

```text
<dst-dir>/
├── model.safetensors
├── config.json
├── policy_preprocessor.json
├── policy_postprocessor.json
└── conversion_meta.json
```

## 运行推理

每个已实现的策略都有独立模板：

| 配置 | backend |
| --- | --- |
| `config/inference/act.yaml` | ACT 同步或异步 action chunk；同步可选 temporal ensemble |
| `config/inference/pi05.yaml` | PI0.5 同步 `select_action()` |
| `config/inference/pi05_async_rtc.yaml` | PI0.5 异步 Real-Time Chunking |
| `config/inference/diffusion.yaml` | Diffusion Policy（观测历史与动作 chunk 由 policy 内部管理） |
| `config/inference/lingbot_va.yaml` | LingBot-VA 视频-动作世界模型（chunk + KV cache 由 policy 内部管理） |

完整说明见 `config/inference/README.md`。

推理配置必须确保：

1. `joints` 的顺序和数量与模型 action 维度一致（7D 对应 7 个关节，14D 对应 14 个关节）。
2. 默认使用 `joints` 构造 `observation.state`；若 state 与 action 的维度或顺序不同，单独配置顶层 `state_joints`。
3. `image_inputs` 的 key 可以是任意 Dora image topic/input ID；alias 用于匹配 checkpoint 的 `observation.images.<alias>`。ACT 和 PI0.5 都使用运行时与 checkpoint 相机的交集（至少需一路匹配），额外 topic 会被忽略。减少训练时使用的视角可能降低策略效果；PI0.5 会为缺失视角生成 masked empty image。
4. `policy.pretrained_path` 指向包含 `model.safetensors` 的目录。
5. `policy.torch_compile` 默认关闭；启用时应在控制开始前完成首次 forward warmup。

```bash
# 不启动 Dora 的单次推理验证
uv run lerobot infer-once --config examples/dora_infer_act_14d/policy_act.yaml

# 真机 Dora 示例：先按设备和模型修改 YAML
cd examples/dora_infer_act_14d
dora run dataflow.yaml
```

可参考 `examples/dora_infer_act_7d/` 和 `examples/dora_infer_act_14d/` 的完整 dataflow 与策略配置。

ACT 默认 `inference_mode: sync`，保持 LeRobot `select_action()` 行为；设置
`inference_mode: async_chunked` 后由后台线程运行 `predict_action_chunk()`，Dora tick
只做非阻塞出队。队列到达低水位时会预取新 chunk，重叠 timestep 按 LeRobot
async inference 默认权重合并；pause、stop 和 reset 会使在途旧 generation 失效。
`temporal_ensemble_coeff` 仅用于 sync，不与 `async_chunked` 组合。

Pi0.5 还需要本地 tokenizer 路径和语言指令：

```yaml
policy:
  type: pi05
  pretrained_path: /path/to/pretrained_model
  tokenizer_path: /path/to/local-paligemma-tokenizer
  instruction: pick up the object and place it in the box
```

Pi0.5 提供两个显式 backend：默认 `inference_mode: sync` 使用 LeRobot 原生
`select_action()`；`inference_mode: async_rtc` 使用后台 `predict_action_chunk()`、
`LatencyTracker` 和 `ActionQueue.merge()` 实现非阻塞 Real-Time Chunking。异步实现
不访问 policy 私有 action queue，也不使用旧版按队列长度跳帧的启发式逻辑。
切换 instruction、pause、stop 或 reset 时会丢弃旧 chunk。加载过程保持 pretrained
目录只读，并在权重损坏或与 config 不兼容时直接失败，不会回退到随机初始化模型。
完整配置见 `examples/dora_sim_infer_pi05_7d_bin/`。

Diffusion Policy 的观测历史（`n_obs_steps` 帧堆叠）与动作 chunk 由 LeRobot policy
内部管理，adapter 每步转发当前观测即可，无自有队列。历史 checkpoint 的非标准图像
特征键（如 pusht 的 `observation.image`）可在 policy 配置中显式声明：

```yaml
policy:
  type: diffusion
  pretrained_path: /path/to/pretrained_model
  expected_image_keys: [observation.image]
```

LingBot-VA 是纯视频-动作世界模型：checkpoint 只有相机输入、无 observation.state，
动作维度由 `used_action_channel_ids` 决定（LIBERO 为 7）。推理需要冻结的
Wan VAE + UMT5 权重（`vae/`、`text_encoder/`、`tokenizer/` 子目录），离线部署把
`wan_pretrained_path` 指向本地目录；UMT5 默认跑在 CPU（每 episode 只编码一次指令）：

```yaml
policy:
  type: lingbot_va
  pretrained_path: /path/to/lingbot_va/pretrained_model
  wan_pretrained_path: /path/to/lingbot_va_base_frozen
  text_encoder_device: cpu
  instruction: pick up the black bowl and place it on the plate
```

## 模型来源

各策略可用的 checkpoint 与附属组件来源如下（离线部署须先本地化下载并固定
revision，`huggingface-cli download <repo> --revision <hash>` 可确定具体版本）：

| 策略 | checkpoint 来源 | 附属组件 | 许可证 |
| --- | --- | --- | --- |
| ACT | 内部训练产物，经 `convert` 转换为 LeRobot 格式（见上文"转换 ACT 权重"） | 无 | — |
| Pi0.5 | 内部 pi05 训练产物（`lerobot_trainer`），`pretrained_path` 指向本地目录 | PALIGEMMA tokenizer（本地化） | — |
| Diffusion | 官方 LeRobot Hub 系列，如 `lerobot/diffusion_pusht`、`lerobot/diffusion_policy_simultaneous_*` | 无 | Apache-2.0 |
| LingBot-VA | 官方 `lerobot/lingbot_va_libero_long` | 冻结 Wan VAE + UMT5：`robbyant/lingbot-va-base`（含 `vae/`、`text_encoder/`、`tokenizer/` 子目录） | 见各 HF 仓库 |
| VLA-JEPA | 官方 `lerobot/VLA-JEPA-LIBERO` | Qwen3-VL 骨干 `Qwen/Qwen3-VL-2B-Instruct`；V-JEPA2 编码器 `facebook/vjepa2-vitl-fpc64-256`（推理可跳过） | Apache-2.0 / MIT |

> 除标注"内部"的 ACT / Pi0.5 外，其余策略均直接加载 LeRobot 官方 Hub 发布的
> checkpoint；运行时按需下载或在配置中指向本地目录。VLA-JEPA 的完整来源与
> 固定 revision 记录见 `examples/dora_sim_infer_vla_jepa_8d_bin/README.md`。

## 打包

```bash
bash scripts/build.sh
```

PyInstaller 产物位于 `dist/lerobot_infer/lerobot_infer`：

```bash
dist/lerobot_infer/lerobot_infer infer --config /path/to/policy_act.yaml
```

构建缓存和产物均被 Git 忽略；需要重新打包时可执行 `bash scripts/build.sh --clean`。

### 单文件节点构建（可选）

以 Forge/PAOS `executable_tar_gz` 契约安装节点时，归档根目录必须只包含一个可执行文件，
且文件名与节点锁的 `entrypoint` 一致（`lerobot_infer`）。默认 onedir 产物是目录，不满足该契约；
需要按节点方式发布时使用独立脚本构建单文件变体，onedir 流程保持不变：

```bash
bash scripts/setup.sh                 # 首次准备 uv 环境
bash scripts/build_node_onefile.sh    # 产物：dist/onefile/lerobot_infer
```

打包为节点归档（`tar -tzf` 只应输出一行 `lerobot_infer`）：

```bash
tar -czf lerobot_infer-<version>-linux-x86_64.tar.gz -C dist/onefile lerobot_infer
```

说明：

- 构建依赖与 onedir 相同：`uv sync --extra build`（或 `scripts/build.sh` 自动安装的 `pyinstaller>=6.0.0`）；
- 构建解释器默认为仓库内 `.venv/bin/python`，可用 `PYTHON=/path/to/python` 覆盖，不自动回退到系统 Python；
- onefile 变体使用 `dist/onefile/` 与 `build/pyinstaller/lerobot_infer_node/`，与 onedir 产物互不覆盖；
- 单文件启动时需要解包，启动耗时和归档体积都高于 onedir，仅建议用于节点交付场景；
- CI 默认仍只构建 onedir 资产，onefile 为可选构建方式。

## License

本项目采用 [Apache License 2.0](LICENSE)。第三方运行时与构建期依赖的许可证见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
