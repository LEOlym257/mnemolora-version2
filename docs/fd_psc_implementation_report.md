# FD-PSC 实现报告

报告日期：2026-08-03（Asia/Hong_Kong）

仓库基线：`a29975964f966f2836a2c7e26f464367c795c333`

实现状态：完整实现；离线验证与真实资源 `UNRUN` 边界见第 5–8 节。

本文逐项对应 `FD_PSC_IMPLEMENTATION_PROMPT_V2.md` 第 31 节的 1–14 项。
证据状态统一使用：

- `PASS`：本工作区内有可重复命令或自动测试证据；
- `UNRUN`：未执行，且明确说明缺失资源或前置条件；
- `STRUCTURAL`：代码和配置的静态映射，可由运行时 manifest 进一步实例化，但不是一次真实 checkpoint 运行结果。

最重要的边界是：本文没有真实 ADAJEPA checkpoint、真实 external split、真实机器人/规划 rollout 或正式性能 benchmark，因此不会把 mock 测试、JEPA loss、dry-run 或未运行 canary 写成真实规划成功率。

## 1. 修改文件清单

### 1.1 原仓库接入文件

- `models/vit.py`：将 predictor causal mask 改为随设备移动且 `persistent=False` 的布尔 buffer，消除硬编码 CUDA 和 checkpoint key 变化。
- `models/visual_world_model.py`：增加冻结视觉 latent 的显式提取、projection replay 和从 frozen latent 编码的接口。
- `planning/adajepa.py`：保留原 JEPA loss/stop-gradient/optimizer/LR/steps，接入 episodic adapter、逐步 trigger、精确 SDC backward、external loss 与 residual descriptor。
- `planning/adajepa_mpc.py`：将一次 `_plan_single()` 定义为一个 episode；只在 `_post_env_feedback()` 注册 support/在线更新；正常非空返回 sleep 一次，异常 abort 且不 sleep。
- `plan.py`：运行时 preprocess identity、FD-PSC 参数传递、report-test 生命周期和实验报告输出。
- `README.md`：增加 FD-PSC 入口、默认关闭说明以及使用/设计/实现报告链接。
- 六个 AdaJEPA 规划配置：
  - `conf/adajepa_plan_cem_maze.yaml`
  - `conf/adajepa_plan_cem_diversemaze.yaml`
  - `conf/adajepa_plan_cem_pushobj.yaml`
  - `conf/adajepa_plan_gd_maze.yaml`
  - `conf/adajepa_plan_gd_diversemaze.yaml`
  - `conf/adajepa_plan_gd_pushobj.yaml`

上述六个配置均增加 Hydra `fd_psc` config group，并默认选择 `disabled`，所以不显式启用时仍走原路径。

### 1.2 新增 FD-PSC 包

```text
fd_psc/__init__.py
fd_psc/activation_subspace.py
fd_psc/canary.py
fd_psc/checkpoint.py
fd_psc/commit_gates.py
fd_psc/config.py
fd_psc/diagnostics.py
fd_psc/encoder_adapters.py
fd_psc/exception_router.py
fd_psc/experiment_reporting.py
fd_psc/external_data.py
fd_psc/gradient_geometry.py
fd_psc/gradient_hooks.py
fd_psc/injector.py
fd_psc/lora_layers.py
fd_psc/low_rank_merge.py
fd_psc/metrics.py
fd_psc/preprocess_identity.py
fd_psc/repair.py
fd_psc/replay_memory.py
fd_psc/slice_initializer.py
fd_psc/spectral_control.py
fd_psc/state_machine.py
fd_psc/trainer.py
fd_psc/transaction.py
```

### 1.3 配置、脚本、文档和测试

```text
conf/fd_psc/default.yaml
conf/fd_psc/disabled.yaml
conf/fd_psc/experiments.yaml
scripts/generate_fd_psc_manifest.py
scripts/run_fd_psc_experiment.py
docs/fd_psc.md
docs/fd_psc_audit.md
docs/fd_psc_design.md
docs/fd_psc_implementation_report.md
tests/test_fd_psc_canary.py
tests/test_fd_psc_candidate_semantics.py
tests/test_fd_psc_config_compat.py
tests/test_fd_psc_config_failfast.py
tests/test_fd_psc_experiment_runner.py
tests/test_fd_psc_injector.py
tests/test_fd_psc_integration.py
tests/test_fd_psc_lora.py
tests/test_fd_psc_math.py
tests/test_fd_psc_merge_pruning.py
tests/test_fd_psc_online_steps.py
tests/test_fd_psc_plasticity.py
tests/test_fd_psc_preprocess_identity.py
tests/test_fd_psc_residual_descriptor.py
tests/test_fd_psc_sdc_anchor.py
tests/test_fd_psc_state_data.py
```

## 2. 架构映射说明

| 论文/协议概念 | ADAJEPA 接入点 | FD-PSC 实现 |
|---|---|---|
| 一次 episode | `AdaJEPAMPCPlanner.plan()` 中的一次 `_plan_single()` | `FDPSCSystem.begin_episode()`、`end_episode_and_sleep()`、`abort_episode()` |
| 在线 support | `AdaJEPAMPCPlanner._post_env_feedback()` | `register_support_segment()`；在首次 optimizer 使用前做 identity/hash 审计 |
| 原 JEPA 更新 | `AdaJEPATrainer._prediction_loss()`、`finetune()` | `_finetune_fd_psc()` 复用相同 loss、target detach、optimizer 类型、LR 分组、steps 和 recent/hard buffer 语义 |
| 全深度 episodic LoRA | predictor 全部活跃 Linear；post-backbone projection head | `injector.py` + `lora_layers.py`；Linear 和 groupwise flattened-kernel ConvLoRA |
| Pilot → Centered | 每个真实 optimizer step 后的冲突检查 | `trainer.py`、`gradient_hooks.py`、`gradient_geometry.py`、`slice_initializer.py` |
| SDC | online JEPA backward | `spectral_control.py` 与 `FDPSCSystem.backward_with_sdc()` 的事件触发 two-pass 路径 |
| soft-NESS | 输入空间历史方向 | `activation_subspace.py` 使用 `Vh.T`、Q/λ、soft weights；`trainer.py` 做候选系数裁剪 |
| slow consolidation | sleep calibration | `low_rank_merge.py` 的 factor concatenation、QR + 小矩阵 SVD、功能误差和允许 rank 选择 |
| 候选专属功能误差 | candidate persistent forward | `_stabilize_candidate_rank()` 对每个 quick/repair clone 重新收集自己的 `H_l` 并求固定点 |
| repair | quick candidate 全部失败后 | `repair.py` + `trainer.py`，累计 checkpoint、JEPA、effective-gradient geometry、可选真实 key-output LPR |
| replay | 成功 slow commit 后 | `replay_memory.py`；连续 frozen-latent 窗口、cluster balance、GRASP、theta0 residual pattern |
| exception memory | slow/repair 不可安全提交时 | `exception_router.py`；固定 episode route、raw descriptor sum/count、local replay、replace/new/evict |
| Gates 1–6 | 唯一 final proposal 的 commit-query | `commit_gates.py`；current/history/anchor/plasticity/functional/spectral drift |
| Gate 7 | 高风险 pre-commit 与周期 post-commit | `canary.py` + `trainer.py`；固定 manifest、clone evaluator、known-good period rollback |
| 原子持久化 | gates 通过后的 swap | `transaction.py` + `checkpoint.py`；prepared journal、验证后不可变版本、atomic latest、rollback journal |
| 最终报告 | 所有 episode 完成后 | `experiment_reporting.py` + `plan.py`；report-test theta0/final 隔离评估与真实 rollout 分栏 |

状态机位于 `fd_psc/state_machine.py`。核心顺序为：

```text
IDLE -> EPISODE_PILOT -> [EPISODE_CENTERED] -> SLEEP_CALIBRATION
     -> [REPAIR] -> FINAL_PROPOSAL_READY -> FINAL_GATE
     -> COMMIT_SLOW / COMMIT_EXCEPTION / REJECT_* / ROLLBACK -> IDLE
```

`theta_0` 是整个基础 world model 的参数和持久 buffer。`FDPSCSystem` 在注入前保存逐位快照，运行中反复执行 `assert_base_frozen()`；slow/exception 从不 merge 回基础权重。

## 3. 新增配置说明

### 3.1 配置入口

- `conf/fd_psc/disabled.yaml`：严格关闭；不注入、不要求 external 文件、不创建 sidecar。
- `conf/fd_psc/default.yaml`：完整 FD-PSC 默认协议。
- `conf/fd_psc/experiments.yaml`：命名基线和消融；runner 为每次运行分配独立目录和 memory。
- `fd_psc/config.py`：typed config、未知字段拒绝、数值/依赖/文件/未实现控制 fail-fast。

完整默认配置的主要分组如下：

| 分组 | 默认含义 |
|---|---|
| `target_modules`, `episodic_lora`, `conv_lora` | 全 predictor Linear；post-backbone projection Linear/Conv2d；rank 8、alpha 16、zero-function Pilot |
| `slow_lora` | 允许 rank `[8,16,24,32]`，maximum 32，谱能量 0.99，功能误差 0.02 |
| `gradient_geometry`, `slice` | effective-weight hooks、双约束、连续冲突 trigger、SLICE exact + symmetric fallback |
| `sdc`, `spectral_surgery` | 事件触发 SDC；输出写入层优先的 calibration-only spectral surgery |
| `activation_subspace`, `merge` | Q/λ、soft-NESS、shared/safe 系数 grid 与 query 前相似度裁剪 |
| `replay`, `repair`, `exception` | bounded global replay、GRASP、累计 JEPA/LPR repair、bounded exception bank/local replay |
| `external_eval_data`, `anchor_data` | 六个固定 split、checksum、context 匹配、single-final-proposal query policy |
| `gates`, `canary` | Gates 1–6 默认开启；canary 接口存在但默认关闭，未运行时报告 `unrun` |
| `checkpoint`, `logging` | 每个持久 commit 都 journal/version；atomic latest；结构化指标和候选报告 |

`default.yaml` 中 external/anchor 路径故意是 `null`；完整运行必须由 runner 或 CLI 提供真实绝对路径。启用状态缺少路径会在 optimizer 更新前失败，不会退化为内部划分 support。

`experiments.yaml` 包含冻结/selected-layer/full-depth reset/accumulate/plain-SVD 基线，以及 soft-NESS、gates、SLICE/几何、SDC、spectral surgery、replay/repair、exception 和 slow-rank 消融。关闭 Gate 1–6 的配置必须显式设置 `gates.allow_unsafe_ablation=true`，不能标为完整方法。

## 4. 关键算法实现位置

| 功能 | 文件和主要符号 |
|---|---|
| 目标发现、reachability、runtime target manifest | `fd_psc/injector.py`: `enumerate_fd_psc_targets`, `inject_fd_psc_adapters`, `TargetManifest` |
| 冻结 latent cut/replay | `fd_psc/encoder_adapters.py`: `FrozenVisualLatent`, `DinoVisualLatentAdapter`, `ProjectionVisualLatentAdapter`, `IdentityVisualLatentAdapter` |
| Linear/Conv2d 多分支 LoRA | `fd_psc/lora_layers.py`: `DualLoRALinear`, `DualLoRAConv2d`, `ConvLoRAGroup` |
| 精确 effective-weight 梯度 | `fd_psc/gradient_hooks.py`: `EffectiveWeightGradientHooks` |
| cosine、EMA、c-PCGrad、dual constraint | `fd_psc/gradient_geometry.py` |
| Pilot/Centered SLICE 与首步 magnitude match | `fd_psc/slice_initializer.py` 与 `FDPSCSystem._activate_centered` |
| base spectrum、SDC、drift、spectral surgery | `fd_psc/spectral_control.py` |
| Q/λ 和 soft-NESS | `fd_psc/activation_subspace.py` |
| factor merge、QR/小 SVD、rank/功能误差 | `fd_psc/low_rank_merge.py` |
| 候选 grid、候选 H 固定点、screen/repair/commit | `fd_psc/trainer.py`: `_make_candidates`, `_stabilize_candidate_rank`, `_screen_candidates`, `_repair_candidate`, `_commit_candidate` |
| 连续 support 拼窗与组合 identity 重审 | `fd_psc/trainer.py`: `_eligible_replay_segments`; `fd_psc/external_data.py`: `audit_composed_support` |
| cluster replay、GRASP | `fd_psc/replay_memory.py` |
| exception router/prototype/local replay | `fd_psc/exception_router.py` |
| Gates 1–6 | `fd_psc/commit_gates.py` |
| Gate 7 canary | `fd_psc/canary.py`; `fd_psc/trainer.py`: `_run_canary_phase`, known-good period helpers |
| 单 final proposal/query state machine | `fd_psc/state_machine.py`; `fd_psc/external_data.py`: commit-query token ledger |
| 事务与 sidecar | `fd_psc/transaction.py`, `fd_psc/checkpoint.py` |
| 指标、诊断、报告 | `fd_psc/metrics.py`, `fd_psc/diagnostics.py`, `fd_psc/experiment_reporting.py` |

## 5. 测试结果

### 5.1 当前工作区完整离线测试

状态：`PASS`

```powershell
& 'C:\Users\linyimoLEO\Documents\Codex\2026-07-23\ni\work\adajepa-venv\Scripts\python.exe' `
  -m unittest discover -s tests -p 'test_fd_psc*.py' -v
```

当前主线精确复跑结果：`Ran 175 tests in 10.620s ... OK`。该结果来自上述带 `-p 'test_fd_psc*.py'` 的完整 FD-PSC 测试命令；其中 `tests.test_fd_psc_integration` 包含 `37` 项并全部通过。

静态编译检查也已通过：

```powershell
& 'C:\Users\linyimoLEO\Documents\Codex\2026-07-23\ni\work\adajepa-venv\Scripts\python.exe' `
  -m compileall -q fd_psc planning models scripts tests plan.py
```

运行环境记录：

```text
Python 3.9.25
PyTorch 2.3.0+cu121
torch.version.cuda = 12.1
torch.cuda.is_available() = True
Windows 10.0.26200
```

测试覆盖包括：

- disabled strict no-op、官方 planner 关闭路径输出/副作用兼容、CPU predictor、旧基础 checkpoint 严格加载后再做零函数注入；
- predictor full-depth、projection Linear/Conv2d、group isolation、padding=`same`/padding mode、frozen-backbone exclusion；
- zero Pilot、实际 rank scaling、Centered 连续性和 optimizer 重建时机；
- effective-gradient hook 对真实 weight grad、cosine unavailable、dual-constraint KKT/c-PCGrad；
- SLICE exact/symmetric/magnitude match、SDC two-pass/event、spectral surgery；
- `Vh.T` input subspace、soft weights、QR + 小 SVD、rank cap 和功能误差；
- candidate-specific H/rank fixed point、冷启动 Gate 2、LPR true key-output；
- manifest checksum/identity/context/leakage、single-use commit-query、report-test isolation；
- 连续短 support 跨 replan 拼窗、非连续不拼、伪边界 fail、组合 content hash 对 external 再审计；
- 两个梯度一致/冲突 episode、近零空间 safe 保留、anchor 冲突投影、slow-rank 饱和，以及 planner 正常/异常 sleep-abort 边界；
- replay/GRASP、theta0 residual pattern、exception raw sum/count/schema compatibility；
- repair/new/replace exception、Gates 1–7、known-good K-period rollback/resume；
- atomic checkpoint、latest recovery、journal/envelope/pointer 交叉验证、prepared/aborted tombstone、rollback、retention 与完整 sidecar round trip；
- `accumulate` 完整 adapter state 恢复、`plain_svd` 事务提交/故障回滚，以及 checkpoint 后 metric sequence 可复现；
- Section 28 指标的数值定义、显式 nullable status、JSONL/CSV 无 NaN 导出和 context retention 统计；
- experiment runner、结果 JSON/CSV、保留 override 和 report lifecycle。

测试期间 PyTorch 对偶数 kernel 的 `padding='same'` 打印一条可能产生内部 zero-padded copy 的 warning；对应数值 hook/Conv 几何测试通过，warning 不是失败。

### 5.2 脚本入口验证

状态：`PASS`

- `python scripts/generate_fd_psc_manifest.py --help`：exit 0。
- `python scripts/run_fd_psc_experiment.py --help`：exit 0。
- `python scripts/run_fd_psc_experiment.py --variant frozen_adajepa --seed 0 --output-root <系统临时目录> --dry-run`：exit 0，生成隔离 run/memory 路径以及命令/结果 JSON。该命令只验证 runner 编排，不加载真实模型，也不是规划 smoke。

## 6. 未运行的测试及原因

| 项目 | 状态 | 原因 |
|---|---|---|
| 真实发布 checkpoint 加载与全模型 target enumeration | `UNRUN` | 工作区无 `.pth`、`.pt` 或 `.ckpt` 文件，`model_name` 未指定 |
| 真实 external 六 split manifest 启动审计 | `UNRUN` | 工作区无 production `*manifest*.json`、calibration 或 commit-query 数据 |
| 真实 MPC maze/diversemaze/pushobj smoke | `UNRUN` | 缺 checkpoint、eval dataset 和可运行环境 |
| 真实机器人/物理环境 rollout | `UNRUN` | 未提供机器人服务、环境资产或固定 seeds |
| 真实 resettable Gate-7 canary | `UNRUN` | 未提供 canary manifest 和独立可复位 evaluator/worker |
| CUDA 数值一致性矩阵（多 GPU/精度） | `UNRUN` | 本轮目标是实现与离线 CPU/mock 验证，未安排硬件矩阵 |
| 正式性能、吞吐、显存 benchmark | `UNRUN` | 无真实 checkpoint/input/batch/hardware protocol，任意数字都不可比较 |
| 多 seed scientific baseline/ablation | `UNRUN` | 缺真实数据和 rollout budget；不能用 mock loss 替代论文结果 |

## 7. 真实环境 smoke test 结果

真实环境结果：`UNRUN`。

仓库扫描未找到模型 checkpoint 或真实 external manifest/split；因此没有执行真实模型 load、真实 planner episode、环境 success 判定或真实 report-test。已通过的 synthetic toy integration 使用 CPU 友好的 mock AdaJEPA 网络和固定 JSON fixture，只证明协议与代码路径可运行，不证明真实任务收益。

唯一已运行的命令级 smoke 是 `frozen_adajepa --dry-run`，它验证隔离目录和命令展开，明确不等价于：

- checkpoint smoke；
- FD-PSC enabled smoke；
- environment rollout；
- canary pass；
- planning success。

真实 smoke 完成前，本报告不提供 success rate。

## 8. 性能和显存开销

正式性能/显存结果：`UNRUN`。

当前实现已经记录下列可用于未来实测的指标，但本报告不填造数字：

- `online_update_latency_s`、gradient collection latency、SLICE latency、sleep latency；
- `adapter_parameter_count`；
- replay memory bytes、checkpoint bytes；
- episodic/slow/subspace rank、candidate count、exception count；
- canary rollout budget/status。

离线完整单元测试在当前环境中的本次 unittest runner 报告耗时 `10.620` 秒；这是小型 mock 测试套件时间，不是模型吞吐或算法 overhead benchmark，不能与原 ADAJEPA 性能比较。虽然当前 PyTorch 报告 CUDA 可用，本轮没有在固定真实模型/batch/precision 下采集 `torch.cuda.max_memory_allocated()`，所以显存峰值仍为 `UNRUN`。

正式 benchmark 至少要固定 checkpoint hash、encoder/predictor target manifest、输入分辨率、batch/window 数、precision、GPU、warm-up、同步策略、episode/replan 数，并分别报告 disabled、episodic-only 和完整 FD-PSC。

## 9. 与原 AdaJEPA 的兼容性说明

- `fd_psc=disabled` 不注入模块、不注册 hook、不要求 external 文件、不创建 sidecar；原 selected-layer 参数选择和 snapshot/reset 路径保持。
- FD 模式仍使用 `_prediction_loss()` 的 sliding one-step JEPA MSE、相同 observation/action mask、`wm.stop_grad`、optimizer 类型、predictor/encoder LR 分组、每次 `finetune()` optimizer 生命周期、配置 steps 和 recent/hard buffer。
- Triggered SLICE 不增加 optimizer step；只有切换后确有下一步时才按原 optimizer/LR 分组重建，让 Centered 参数从下一步受训。
- official checkpoint 先加载，之后才枚举目标和注入；基础参数与持久 buffer 保持逐位冻结，持久记忆只写独立 sidecar。
- frozen visual backbone 不注入 adapter；BatchNorm/其他基础持久 buffer 保持 eval/frozen。
- `models/vit.py` 的 mask 是 non-persistent buffer，因此修复 CPU/device 行为但不增加 checkpoint key。
- frozen-latent 协议是 additive API；`VisualLatentAdapter` 不是 `nn.Module`，缓存它不会改变模型 state dict。
- 默认承诺 between-episode sidecar 恢复；没有序列化环境和 planner warm-start 时，不承诺 mid-episode rollout 精确恢复。

## 10. 已知限制

1. 尚未对任何真实发布 checkpoint 生成 target manifest；真实层数、维度、实际裁剪 rank 和 manifest hash 未知。
2. 尚未构造/审计 production external manifest；所有 external 数据结果仅有 schema 与 mock fail-fast 测试证据。
3. 真实 canary 需要调用方提供独立、确定性可复位 evaluator；默认配置关闭 canary，环境不可用时只能报告 `unrun`。
4. frozen-latent adapter 只支持显式列出的 DINO、SmallResNet/SmallResNetGeM、ViTEncoder 和少数完全冻结 identity encoders；未知边界的 encoder fail-fast，不使用 hook 猜测。
5. manifest 生成脚本只审计已存在 split，不负责从原始轨迹生成 frozen latent 或稳定 IDs。
6. report-test 是最终离线证据之一，但仍需独立真实 rollout 才能报告规划 success；JEPA loss、commit-query pass 或 mock canary 不能替代 success。
7. 当前未提供多 GPU、混合精度、长时间 memory growth 或大 exception bank 的性能结论。
8. 周期 canary rollback 的算法内存和 journal 语义有 mock 集成覆盖；真实环境 nondeterminism 仍需单独验证和披露。
9. 所有实现改动目前位于未提交工作区；复现实验应先固定新的代码 commit，而不是只引用基线 commit。

## 11. 下一步实验建议

1. 获取每个发布 checkpoint，计算文件 SHA-256，逐一运行 target enumeration，并保存 `fd_psc_runtime_manifest.json`。
2. 用与 runtime preprocess identity 完全相同的管线生成六个 external split；先运行 manifest generator，再做一次 enabled runner dry-run。
3. 每个 encoder 变体先执行一个 episode 的真实 smoke：确认 target reachability、theta0 bitwise、单次 query、slow commit/reject 和 sidecar resume。
4. 在固定 deterministic reset 环境接入 canary evaluator，分别验证 rank expansion pre-commit 和 `K>1` known-good period rollback。
5. 建立 disabled/episodic-only/full FD-PSC 的 latency、吞吐和显存 benchmark，固定硬件、精度、window/batch 和 warm-up。
6. 按 `experiments.yaml` 对 baseline/ablation 做多 seed、相同 support/external/canary/rollout budget 的实验；只用 calibration 调参，commit-query 不参与跨 run 选择。
7. 结果表分别报告真实 rollout、report-test、calibration 和 commit-query，不把后两者写成 test performance。

## 12. 可复制的运行命令

### 12.1 离线测试

```powershell
$Python = 'C:\Users\linyimoLEO\Documents\Codex\2026-07-23\ni\work\adajepa-venv\Scripts\python.exe'
& $Python -m unittest discover -s tests -p 'test_fd_psc*.py' -v
& $Python -m unittest tests.test_fd_psc_integration -v
& $Python scripts/generate_fd_psc_manifest.py --help
& $Python scripts/run_fd_psc_experiment.py --help
```

### 12.2 计算真实 checkpoint hash

```powershell
$BaseCheckpoint = 'C:\abs\checkpoints\MODEL\checkpoints\model_latest.pth'
$BaseHash = (Get-FileHash -Algorithm SHA256 $BaseCheckpoint).Hash.ToLowerInvariant()
$BaseHash
```

### 12.3 生成 external manifest

```powershell
& $Python scripts/generate_fd_psc_manifest.py `
  --output C:\abs\data\fd_psc\manifest.json `
  --base-checkpoint-hash $BaseHash `
  --preprocess-hash <64-character-runtime-preprocess-sha256> `
  --latent-adapter-schema fd-psc-frozen-visual-latent-v1 `
  --manifest-id maze-fd-psc-v1 `
  --episode-contexts C:\abs\data\fd_psc\episode_contexts.json `
  --calibration C:\abs\data\fd_psc\calibration.json `
  --commit-query C:\abs\data\fd_psc\commit_query.json `
  --plasticity-support C:\abs\data\fd_psc\plasticity_support.json `
  --plasticity-query C:\abs\data\fd_psc\plasticity_query.json `
  --report-test C:\abs\data\fd_psc\report_test.json `
  --anchor C:\abs\data\fd_psc\anchor.json
```

### 12.4 隔离 dry-run 与真实运行

```powershell
# 已验证 exit 0 的 disabled runner dry-run；它不加载真实模型。
& $Python scripts/run_fd_psc_experiment.py `
  --variant frozen_adajepa `
  --seed 0 `
  --output-root $env:TEMP\fd_psc_dry_run `
  --dry-run

# Enabled dry-run：需要真实 manifest；本工作区尚未运行。
& $Python scripts/run_fd_psc_experiment.py `
  --variant dual_constraint `
  --plan-config adajepa_plan_cem_maze `
  --seed 100 `
  --manifest C:\abs\data\fd_psc\manifest.json `
  --output-root C:\abs\runs\fd_psc `
  --dry-run `
  ckpt_base_path=C:\abs\checkpoints `
  model_name=MODEL `
  eval_data_path=C:\abs\data\point_maze_medium `
  +wandb_logging=false

# 确认 dry-run 展开的命令、hash 和路径后，删除 --dry-run 才执行真实规划。
```

### 12.5 直接运行完整默认方法

```powershell
& $Python plan.py --config-name adajepa_plan_cem_maze `
  fd_psc=default `
  seed=100 `
  ckpt_base_path=C:\abs\checkpoints `
  model_name=MODEL `
  eval_data_path=C:\abs\data\point_maze_medium `
  fd_psc.external_eval_data.manifest_path=C:\abs\data\fd_psc\manifest.json `
  fd_psc.external_eval_data.calibration_path=C:\abs\data\fd_psc\calibration.json `
  fd_psc.external_eval_data.commit_query_path=C:\abs\data\fd_psc\commit_query.json `
  fd_psc.external_eval_data.plasticity_support_path=C:\abs\data\fd_psc\plasticity_support.json `
  fd_psc.external_eval_data.plasticity_query_path=C:\abs\data\fd_psc\plasticity_query.json `
  fd_psc.external_eval_data.report_test_path=C:\abs\data\fd_psc\report_test.json `
  fd_psc.anchor_data.manifest_path=C:\abs\data\fd_psc\manifest.json `
  fd_psc.anchor_data.data_path=C:\abs\data\fd_psc\anchor.json `
  fd_psc.checkpoint.state_directory=C:\abs\runs\fd_psc-main\memory `
  fd_psc.checkpoint.latest_pointer_path=C:\abs\runs\fd_psc-main\memory\latest.json `
  hydra.run.dir=C:\abs\runs\fd_psc-main `
  +wandb_logging=false
```

以上 enabled 命令是可复制模板，不是本报告已执行的真实结果。

## 13. External manifest schema、生成命令、hash 与泄漏审计

### 13.1 六个 split 与用途

| split | 唯一允许用途 |
|---|---|
| `calibration` | gradient/SLICE、候选生成、screen、repair、唯一候选选择 |
| `commit_query` | 只 gate calibration 冻结的唯一 final proposal，一次性 token |
| `plasticity_support` | Gate 4 clone 的一次真实配置 update event |
| `plasticity_query` | 评估 Gate 4 clone 的前后 gain |
| `report_test` | 全 episode 结束后的 theta0/final 报告，不进入算法状态或决策 |
| `anchor` | immutable anchor gradient、regression 与安全约束 |

### 13.2 split record schema

```json
{
  "schema_version": 1,
  "records": [
    {
      "record_id": "cal-maze-a-0001",
      "context_identifier": "maze-a",
      "trajectory_id": "traj-cal-0001",
      "transition_ids": ["traj-cal-0001:t000"],
      "frame_ids": ["traj-cal-0001:f000", "traj-cal-0001:f001"],
      "content_hash": "<64-character-sha256>",
      "payload": {
        "frozen_visual_latent": {
          "tensor": "<JSON tensor or file-backed mapping>",
          "layout": "tokens|feature_map|vector",
          "encoder_type": "<fully-qualified-type>",
          "cut_path": "<explicit-frozen-cut>",
          "schema_version": "fd-psc-frozen-visual-latent-v1",
          "metadata": {}
        },
        "proprio": "<T+1 sequence>",
        "actions": "<T sequence>"
      },
      "metadata": {}
    }
  ]
}
```

每条记录必须在 `payload` 与 `payload_path` 中恰选一种。JSON file-backed payload 的 `content_hash` 是解析内容的 canonical JSON hash；`.pt/.pth` file-backed payload 使用文件字节 SHA-256，并要求加载后是 mapping。

### 13.3 顶层 manifest schema

```json
{
  "schema_version": 1,
  "manifest_id": "maze-fd-psc-v1",
  "base_checkpoint_hash": "<sha256>",
  "preprocess_hash": "<sha256>",
  "latent_adapter_schema": "fd-psc-frozen-visual-latent-v1",
  "splits": {
    "calibration": {"path": "calibration.json", "sha256": "<sha256>"},
    "commit_query": {"path": "commit_query.json", "sha256": "<sha256>"},
    "plasticity_support": {"path": "plasticity_support.json", "sha256": "<sha256>"},
    "plasticity_query": {"path": "plasticity_query.json", "sha256": "<sha256>"},
    "report_test": {"path": "report_test.json", "sha256": "<sha256>"},
    "anchor": {"path": "anchor.json", "sha256": "<sha256>"}
  },
  "contexts": {"maze-a": {"calibration": ["cal-maze-a-0001"]}},
  "episode_contexts": {"sample:0": "maze-a", "seed:101": "maze-a"},
  "leakage_audit": {
    "status": "pass",
    "record_count": 0,
    "context_count": 0,
    "identity_counts": {
      "trajectory": 0,
      "transition": 0,
      "frame": 0,
      "content": 0
    }
  },
  "manifest_content_hash": "<sha256>"
}
```

示例中的计数 `0` 只是 schema 占位符，不是本次数据结果。

### 13.4 Hash 定义与审计顺序

- inline/JSON payload hash：`SHA256(UTF-8 json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))`。
- split hash：split 文件原始字节 SHA-256。
- manifest content hash：在加入 `manifest_content_hash` 字段前，对整个 manifest 做上述 canonical JSON hash；运行时删除声明字段后重算并 constant-time 比较。
- base checkpoint hash：实际加载的 `model_<epoch>.pth` 文件字节 SHA-256。
- preprocess hash：action/state/proprio stats、observation/encoder transform、frameskip、`num_hist/num_pred` 与 latent layout 的版本化 identity hash。
- leakage audit：六 split 在 trajectory、transition、frame、content 四层全量两两不重叠；在线 support 在首次更新前对同一 external index 增量检查；跨 replan 组合 support 还以新的组合 content hash 做只读四层再审计。

生成命令见第 12.3 节。生成器成功会原子写 manifest 并输出 record/context/identity counts；运行时仍重新读取 split、校验 checksum/schema/context/base/preprocess/latent schema。

### 13.5 本次实际 hash/审计结果

| 项目 | 结果 |
|---|---|
| production base checkpoint hash | `UNRUN`：无 checkpoint 文件 |
| production preprocess hash | `UNRUN`：没有真实 PlanWorkspace/checkpoint run |
| production manifest content hash | `UNRUN`：无真实 manifest |
| production split file hashes | `UNRUN`：无真实六 split |
| production leakage audit | `UNRUN`：无真实数据可审计 |
| generator/registry mock schema、hash 和 fail-fast 测试 | `PASS` |
| inline payload mismatch、payload_path 伪 hash、跨 split 复用检测 | `PASS` |
| support/external overlap和 composed-support 新 content collision | `PASS` |
| single-use proposal-bound commit-query 与 report-test isolation | `PASS` |

本报告不公布 synthetic fixture hash，因为它们不是实验数据版本，也不能代替 production manifest hash。

## 14. Checkpoint/encoder target manifest 与 zero projection-target

### 14.1 Runtime target manifest schema

目标 manifest 由 `fd_psc.injector` 在真实基础 checkpoint 加载后、adapter 注入前生成，schema 为 `fd-psc-target-manifest-v1`。每个 entry 至少记录：

```text
module_path, logical_layer_id, layer_type,
in_features, out_features, module_group, role,
active_in_forward, active_detection,
default_inject, injected, actual_rank,
attention_output, mlp_output, final_projection,
kernel_size, stride, padding, dilation, groups, logical_group, bias
```

顶层 metadata 记录：

```text
encoder_type
projection_head_exists
projection_module_paths
zero_projection_targets
projection_status
excluded_subtrees = [encoder.base_model, decoder]
```

运行目录的 `fd_psc_runtime_manifest.json` 同时写入 `base_checkpoint_hash`、`preprocess_hash`、external/canary manifest hash、`target_manifest_hash` 和完整 `target_manifest`。sidecar 也绑定 base 与 target-manifest hash；路径、类型、维度、Conv 几何或 hash 不同即拒绝恢复。

### 14.2 Predictor target 规则

所有 encoder/checkpoint 变体共享 predictor 规则：schema dry-run 实际经过的 `nn.Linear` 才能注入，包括每层：

```text
predictor.transformer.layers.<i>.0.to_qkv
predictor.transformer.layers.<i>.0.to_out.0   # 非 Identity 时
predictor.transformer.layers.<i>.1.net.1
predictor.transformer.layers.<i>.1.net.4
```

final norm、positional embedding、bias、激活和 decoder 不注入。默认 action/proprio encoder target 关闭。

### 14.3 仓库 encoder 变体的结构预期

下表是 `STRUCTURAL` 映射，不是已运行真实 checkpoint manifest；实际 active path、维度、group 数、actual rank 和 hash 必须以每个 checkpoint 的 runtime JSON 为准。

| 配置/encoder | frozen cut | projection target 结构预期 | zero projection 语义 | 真实 checkpoint manifest |
|---|---|---|---|---|
| `conf/encoder/dino.yaml`，DINO patch tokens，无 projector | `base_model.forward_features['x_norm_patchtokens']` | 无 projection target；仅 active predictor Linear | 合法 `not_applicable_no_projection_head` | `UNRUN` |
| `conf/encoder/dino_cls.yaml`，DINO CLS token | `base_model.forward_features['x_norm_clstoken']` | 无 projection target；仅 active predictor Linear | 合法 `not_applicable_no_projection_head` | `UNRUN` |
| `conf/encoder/dino_channel.yaml` | DINO patch tokens，projector 前 | `encoder.projector.conv_layers.*` 的 active Conv2d；每个 Conv group 独立 logical target | head 存在时不得为零 | `UNRUN` |
| `conf/encoder/dino_global.yaml` | DINO patch tokens，projector 前 | active `encoder.projector.mix`、`down_blocks` 内 Conv2d、`head`；每 group 独立 | head 存在时不得为零 | `UNRUN` |
| `SmallResNet` / `SmallResNetGeM` | `encoder.projection:input` | active `encoder.projection` Linear | head 存在时不得为零 | `UNRUN` |
| `ViTEncoder` | `encoder.to_out:input` | active `encoder.to_out` Linear | head 存在时不得为零 | `UNRUN` |
| `resnet18` / `ResNetSpatial` / `R3M` / `DummyModel` identity adapters | `encoder:output` | 无 post-backbone projection target | 完全冻结 encoder 时合法 not-applicable | `UNRUN` |

`DinoV2Encoder.agg_mlp` 不在默认 `VWorldModel.encode_obs()` JEPA forward 中，reachability dry-run 会将其排除；不能因为它是 Linear 就静态注入。

### 14.4 Zero projection-target 判定

- encoder 明确没有 projection head：`zero_projection_targets=true`、`projection_status=not_applicable_no_projection_head`，默认是合法 not-applicable。
- encoder 声明存在 projection head，且 `require_projection_targets_if_head_exists=true`：若 dry-run 未找到 active Linear/Conv2d，启动必须 fail-fast。
- 任意 `encoder.base_model` 下的候选是硬错误，不得用“无 target”掩盖 backbone 注入。
- grouped Conv2d 为每个 `::group=<g>` 生成独立 logical entry 和实际裁剪 rank。

相关 mock 测试状态为 `PASS`：no-head explicit not-applicable、present-head empty target fail-fast、DINO/channel/groupwise manifest、inactive optional target、duplicate injection 和 frozen-backbone exclusion。

### 14.5 每个真实 checkpoint 的交付状态

工作区没有任何真实 checkpoint，因此无法诚实列出每个 checkpoint 的实际 target entries/hash。此项真实数据状态为 `UNRUN`，不是 `PASS`。获得 checkpoint 后，每个 checkpoint/encoder 变体至少保存：

```text
checkpoint absolute path + file SHA-256
resolved encoder type/config
fd_psc_runtime_manifest.json
target_manifest_hash
projection_status / zero_projection_targets
injected logical target count by predictor/projection/group
actual_rank by logical layer
frozen-backbone exclusion assertion
theta0 bitwise assertion result
```

只有这些 runtime artifacts 生成并归档后，才能把第 14 项从结构映射升级为真实 checkpoint 验证结果。
