# FD-PSC 使用指南

本文说明如何在 ADAJEPA 中准备并运行 FD-PSC（Full-Depth Plasticity with Safe Consolidation）、恢复持久状态，以及如何执行可复现的基线与消融实验。实现所依据的模型接入点和生命周期审计见 [`fd_psc_audit.md`](fd_psc_audit.md)。

## 1. 运行边界与核心不变量

FD-PSC 在原有 ADAJEPA MPC 在线更新时机上增加两类 LoRA：

- `episodic` 分支只在当前 episode 内学习，episode 结束或异常时清空；
- `slow` 和 `exception` 分支只在 sleep proposal 通过全部启用的安全 gate 后持久化。

以下边界不可混用：

- 官方 ADAJEPA checkpoint 是只读基础模型 `theta_0`，FD-PSC 不覆盖它；
- 当前 episode 的执行数据只能作为 support，不能事后切成 calibration 或 commit-query；
- calibration 只参与候选生成和选择；commit-query 只 gate 已选出的唯一 final proposal；
- commit-query 不是最终测试集；最终结论只能来自 `report_test` 或从未参与提交决策的真实 rollout；
- `fd_psc=disabled` 走原 ADAJEPA 参数选择、snapshot/reset 和规划路径，不要求 FD-PSC 外部数据或 sidecar。

一次 `_plan_single()` 对应一个 FD-PSC episode。仅当它正常返回且 planner buffer 非空时执行一次 sleep；异常返回必须 abort，且不得生成 sleep proposal。

## 2. 安装与快速验证

仓库的标准环境是 Python 3.9、PyTorch 2.3，完整依赖固定在根目录的 `environment.yaml`：

```bash
conda env create -f environment.yaml
conda activate ts
```

该 Conda 文件含 Linux 平台包；在 Windows 上应使用具有等价 Python/PyTorch/Hydra 版本的独立环境，不要在已有生产环境中混装平台包。

从仓库根目录运行离线测试：

```bash
python -m unittest discover -s tests -v
```

测试不依赖网络、`torch.hub` 下载、机器人服务或用户缓存。真实 checkpoint、真实评估数据或可复位环境不可用时，真实 smoke/canary 必须标记为“未运行”并给出原因，不能把它记为通过。

## 3. 基础 checkpoint

规划配置仍使用 ADAJEPA 原有目录结构：

```text
<checkpoint-model-dir>/
  hydra.yaml
  checkpoints/
    model_<model_epoch>.pth
```

默认 `model_epoch=latest`。外部数据 manifest 中的 `base_checkpoint_hash` 必须是本次实际加载的 `model_<model_epoch>.pth` 的 SHA-256，而不是目录、`hydra.yaml` 或下载压缩包的 hash。例如在 PowerShell 中：

```powershell
$BaseCheckpoint = "checkpoints/mediummaze_dynamics_shift/checkpoints/model_latest.pth"
$BaseHash = (Get-FileHash -Algorithm SHA256 $BaseCheckpoint).Hash.ToLowerInvariant()
$BaseHash
```

启动时先加载基础 checkpoint，再发现目标模块、生成运行时目标 manifest、注入 adapter，最后才允许恢复 FD-PSC sidecar。基础 hash、模块路径、层类型、维度或 Conv2d 几何不匹配时必须拒绝恢复。

## 4. 准备外部固定数据

### 4.1 六个严格隔离的 split

每次完整 FD-PSC 运行需要下列六个 JSON split：

| split | 只允许的用途 |
|---|---|
| `calibration` | SLICE、候选生成、screening、repair 和候选选择 |
| `commit_query` | 对唯一 final proposal 执行一次最终 commit gate |
| `plasticity_support` | Gate 4 的临时真实 update event |
| `plasticity_query` | 只评估上述临时 update 的收益 |
| `report_test` | 最终离线报告，不参与任何学习或决策 |
| `anchor` | 不可变安全锚点、梯度和回归约束 |

六者必须在 trajectory、transition、frame 和规范化内容 hash 四个层面两两不重叠。每个新在线 support window 在第一次更新前也必须通过同样的外部集合泄漏检查。

### 4.2 split schema

split 文件可以是记录数组，也可以使用带版本的 envelope：

```json
{
  "schema_version": 1,
  "records": [
    {
      "record_id": "cal-maze-a-0001",
      "context_identifier": "maze-a",
      "trajectory_id": "traj-cal-0001",
      "transition_ids": ["traj-cal-0001:t000-t004"],
      "frame_ids": ["traj-cal-0001:f000", "traj-cal-0001:f001"],
      "content_hash": "<64-character-sha256>",
      "payload": {"<adapter-defined-field>": "<value>"},
      "metadata": {}
    }
  ]
}
```

也可用 `trajectory_ids` 表示多条 trajectory。字段要求如下：

- `record_id` 在本 split 内唯一；
- `context_identifier` 必须由 episode 开始前可见的 evaluation manifest 或显式环境/任务 metadata 提供，不能从 query 或最终结果反推；
- `trajectory_id(s)`、`transition_ids`、`frame_ids` 必须稳定且可复现，transition/frame 列表不得为空；
- `content_hash` 是 64 位 SHA-256；若记录内含 `payload`，它必须等于对 payload 使用 UTF-8、键排序、紧凑分隔符、禁止 NaN 的 canonical JSON hash；
- payload 的实际张量/字段协议由 `latent_adapter_schema` 标识。记录也可带 `payload_path`，但内容身份和文件版本仍必须可审计。

同一 `context_identifier` 至少要能映射到该运行所需的 calibration、commit-query、plasticity 和 report 数据。默认 `missing_context_policy=error`，缺少映射应在 episode 第一次在线更新前失败。

### 4.3 生成并审计 manifest

`scripts/generate_fd_psc_manifest.py` 不生成轨迹或 latent；它读取已经准备好的 split，校验 schema、内容 hash 和跨 split 泄漏，然后原子写入 manifest。

```powershell
python scripts/generate_fd_psc_manifest.py `
  --output data/fd_psc/manifest.json `
  --base-checkpoint-hash $BaseHash `
  --preprocess-hash <64-character-preprocess-sha256> `
  --latent-adapter-schema frozen_backbone_latent/v1 `
  --manifest-id maze-fd-psc-v1 `
  --episode-contexts data/fd_psc/episode_contexts.json `
  --calibration data/fd_psc/calibration.json `
  --commit-query data/fd_psc/commit_query.json `
  --plasticity-support data/fd_psc/plasticity_support.json `
  --plasticity-query data/fd_psc/plasticity_query.json `
  --report-test data/fd_psc/report_test.json `
  --anchor data/fd_psc/anchor.json
```

`preprocess_hash` 不能是格式正确的随机占位值。启用 FD-PSC 时，`PlanWorkspace` 会对 action/state/proprio 的 mean/std、观测 transform、world-model encoder transform、frameskip、`num_hist/num_pred` 和版本化视觉布局生成 `fd_psc_preprocess_identity.json`；manifest 中的 hash 必须与其中 `preprocess_hash` 完全一致，否则在任何 online update 前拒绝启动。生成 frozen latent 的管线必须使用同一 identity 定义。

可选的 `episode_contexts.json` 是 JSON 对象，例如 `{"sample:0":"maze-a","seed:101":"maze-a"}`。它把 episode 开始前已知的 sample/seed 显式映射到 context；映射 value 必须出现在 split context 索引中。未提供此映射时，环境/任务 metadata 必须直接给出 `context_identifier`。即使所有 calibration 记录恰好只有一个 context，也禁止据此反推当前 episode context。

生成成功的 schema-v1 manifest 包含：

- 基础 checkpoint hash、预处理 hash 和 latent adapter schema；
- 六个 split 相对路径及文件 SHA-256；
- `context_identifier -> split -> record_ids` 完整索引；
- `leakage_audit.status=pass`、记录数和四类身份计数；
- `manifest_content_hash`。

生成器成功只证明静态 JSON 通过审计。运行时仍会重新验证文件可读性、checksum、schema、context 索引、基础 checkpoint hash，以及在线 support 与全部外部集合的隔离。

### 4.4 两类 manifest 不要混淆

- 上述 external-data manifest 由用户预先生成，通过 `--manifest` 传入；
- target-module manifest 在加载真实基础 checkpoint 后由运行时生成，记录实际活跃的 predictor Linear 和 post-backbone projection Linear/Conv2d、logical layer/group ID、实际裁剪 rank 与几何信息。

目标发现必须排除 `encoder.base_model` 等 frozen backbone 子树。encoder 没有 projection head 时 projection target 为零是合法的 `not_applicable`；存在 projection head 却没有识别到活跃 Linear/Conv2d 时必须 fail-fast。sidecar 绑定的是基础 checkpoint 和运行时目标 manifest，不能 best-effort 套到另一模型。

## 5. 运行 FD-PSC

### 5.1 先做 dry run

实验 runner 会从 `conf/fd_psc/experiments.yaml` 读取命名 variant，为每次调用建立带唯一 run ID 的独立运行目录，并自动从 external manifest 展开六个 split 路径：

```powershell
python scripts/run_fd_psc_experiment.py `
  --variant dual_constraint `
  --plan-config adajepa_plan_cem_maze `
  --seed 100 `
  --manifest data/fd_psc/manifest.json `
  --output-root fd_psc_runs/smoke `
  --dry-run `
  ckpt_base_path=./checkpoints `
  model_name=mediummaze_dynamics_shift `
  eval_data_path=./data/point_maze_medium `
  +wandb_logging=false
```

dry run 会写 `run_metadata.json`、`result.json` 和 `result.csv`，并打印最终 `plan.py` 命令，但不启动规划。确认 checkpoint、评估数据、manifest 和 `hydra.run.dir` 后，删除命令中的 `--dry-run` 执行。可用 `--run-id <唯一标签>` 指定可读的运行标签；目标目录已存在时 runner 会拒绝复用。

启用 FD-PSC 的 variant 必须提供 `--manifest`。额外的 Hydra override 放在 runner 选项之后。完整主方法使用 `conf/fd_psc/default.yaml`；`experiments.yaml` 中的名字是明确标注的比较运行，不能把关闭 gate 或改变算法路径的 variant 报告为完整 FD-PSC。

external/anchor split、checkpoint sidecar、seed、run mode 和 `hydra.run.dir` 是 runner 保留项。CLI 不能用后置 Hydra override 改写或删除这些键，且保留值最后追加到子进程命令。完整 `fd_psc` variant 若缺少 file-backed `report_test` 会在启动规划前失败；关闭 FD-PSC 的基线明确记录 `report_test.status=not_applicable`，其最终效果使用独立真实 rollout。

### 5.2 直接运行完整默认方法

需要运行不带消融标签的完整默认配置时，可以直接调用 `plan.py`。此时必须显式提供所有外部路径和独立 sidecar 路径：

```powershell
python plan.py --config-name adajepa_plan_cem_maze `
  fd_psc=default `
  seed=100 `
  ckpt_base_path=./checkpoints `
  model_name=mediummaze_dynamics_shift `
  eval_data_path=./data/point_maze_medium `
  fd_psc.external_eval_data.manifest_path=C:/abs/data/fd_psc/manifest.json `
  fd_psc.external_eval_data.calibration_path=C:/abs/data/fd_psc/calibration.json `
  fd_psc.external_eval_data.commit_query_path=C:/abs/data/fd_psc/commit_query.json `
  fd_psc.external_eval_data.plasticity_support_path=C:/abs/data/fd_psc/plasticity_support.json `
  fd_psc.external_eval_data.plasticity_query_path=C:/abs/data/fd_psc/plasticity_query.json `
  fd_psc.external_eval_data.report_test_path=C:/abs/data/fd_psc/report_test.json `
  fd_psc.anchor_data.manifest_path=C:/abs/data/fd_psc/manifest.json `
  fd_psc.anchor_data.data_path=C:/abs/data/fd_psc/anchor.json `
  fd_psc.checkpoint.state_directory=fd_psc_state `
  fd_psc.checkpoint.latest_pointer_path=fd_psc_state_latest.json `
  hydra.run.dir=C:/abs/runs/fd_psc-main-seed100 `
  +wandb_logging=false
```

相对的 `state_directory`、`latest_pointer_path` 和 `resume_path` 按 Hydra runtime output directory 解析，而不是按进程启动目录猜测。外部数据建议使用绝对路径；runner 已自动将 manifest 内相对 split 路径解析为绝对路径。

### 5.3 严格旧路径

显式关闭 FD-PSC：

```powershell
python plan.py --config-name adajepa_plan_cem_maze `
  fd_psc=disabled `
  ckpt_base_path=./checkpoints `
  model_name=mediummaze_dynamics_shift `
  eval_data_path=./data/point_maze_medium `
  +wandb_logging=false
```

关闭时不得要求 manifest、external split 或 sidecar 存在。

## 6. Sidecar checkpoint 与恢复

runner 的典型输出布局如下：

```text
<output-root>/
  results.csv
  <variant>-seed<seed>-<run-id>/
    run_metadata.json
    result.json
    result.csv
    fd_psc_experiment_report.json
    fd_psc_experiment_report.csv
    memory/
      latest.json
      state-<commit-id>-<hash-prefix>.pt
      journal-<commit-id>.json
```

`fd_psc_experiment_report.json/csv` 由规划进程生成，包含真实 rollout 指标以及隔离的 report-test theta_0/final JEPA loss 与 gain；`result.json/csv` 和汇总 `results.csv` 在此基础上再加入 variant、seed、manifest hash、返回码、耗时和隔离 sidecar 路径。`external_calibration_gain`、`commit_query_gain` 与 `report_test_gain` 始终分栏，commit-query 明确不是 test set。规划前只验证 report-test context、文件和 checksum 是否完整，模型 loss 不会提前计算；所有 episode 和真实 rollout 完成后，外层 workspace 才在 eval mode 下分别评估全 adapter 禁用的 theta_0 和最终记忆状态。两者都不会传入候选选择、gate、routing 或回滚。

每次持久提交遵循以下协议：先写 `prepared` journal，把完整状态写入临时文件，重新加载并验证 schema/content/base/target-manifest hash，移动为不可变版本，原子更新 latest pointer，最后将 journal 标记为 `committed`。任何一步失败都必须恢复提交前 live state 和旧 latest；未完成的 `prepared` 记录不能恢复。

事务协议要求每个持久 commit 都生成 journal/version，因此合规运行只接受 `checkpoint.save_every_episodes=1`；不能用稀疏 checkpoint 跳过中间提交。版本保留数在启用周期 canary 时至少为 `canary.every_episodes + 1`。

两个显式基线有不同但同样明确的持久化边界：

- `run_mode=accumulate` 不产生 slow commit，也不进入 sleep/gate/bank；跨 episode 延续的是 live episodic adapter。默认每个 episode 的终态 `snapshot-episode-XXXXXXXX` 除通用 sidecar 状态外，还保存 `accumulate_adapter_state`：按完整 logical-layer registry 保存 slow/exception/Pilot/Centered 动态 factors、初始 Centered factors以及 enabled/frozen 等 adapter 生命周期字段。恢复时该字段必须存在、registry 完全一致且数值有限；系统逐项恢复它，清除当前 routed exception，并先把可训练 episodic 参数设为非训练态，直到下一次原定 online update 再按正常参数组启用。因此从该 snapshot 继续时，累积 adapter 与 snapshot 边界逐位一致，而不是只恢复通常为空的 `adapter_slow`。若把 `save_every_episodes` 改为大于 1，只能恢复到最近一次实际 snapshot，不能宣称其后 episode 已 durable；snapshot 写失败也不会假装回滚已经发生的 accumulate update，旧 latest 保留且 durability tombstone 阻止从过期边界继续。
- `run_mode=plain_svd` 在每个 episode 将“原 slow + 本 episode task”做普通 fixed-rank factor SVD。adapter slow 交换、commit/counter/lifecycle 更新与 sidecar journal/version/latest 写入位于同一个 `StateTransaction` 内，只有 sidecar 成功后才 `transaction.commit()`。在启用 checkpoint 的默认路径上，任一写入或重载校验失败都会抛错，并恢复进入 plain-SVD commit 前的完整 adapter state、计数器、状态机和 RNG；旧 latest 不变，不能报告本次 slow commit 成功。若显式关闭 checkpoint，该基线只能保留进程内 slow 更新，不具备跨进程 durability。

### 6.1 between-episode 恢复

恢复时必须使用完全相同的基础 checkpoint、目标模块结构、外部 manifest、预处理协议和关键配置。基线/消融 runner 强制每次从全新 memory 开始，因此会拒绝 `fd_psc.checkpoint.resume_path` 等保留 override。需要显式续跑时，直接调用 `plan.py` 并使用新的 Hydra 输出目录，同时将 `resume_path` 指向已验证 sidecar：

```powershell
python plan.py --config-name adajepa_plan_cem_maze `
  fd_psc=default `
  seed=100 `
  ckpt_base_path=./checkpoints `
  model_name=mediummaze_dynamics_shift `
  eval_data_path=./data/point_maze_medium `
  fd_psc.external_eval_data.manifest_path=C:/abs/data/fd_psc/manifest.json `
  fd_psc.external_eval_data.calibration_path=C:/abs/data/fd_psc/calibration.json `
  fd_psc.external_eval_data.commit_query_path=C:/abs/data/fd_psc/commit_query.json `
  fd_psc.external_eval_data.plasticity_support_path=C:/abs/data/fd_psc/plasticity_support.json `
  fd_psc.external_eval_data.plasticity_query_path=C:/abs/data/fd_psc/plasticity_query.json `
  fd_psc.external_eval_data.report_test_path=C:/abs/data/fd_psc/report_test.json `
  fd_psc.anchor_data.manifest_path=C:/abs/data/fd_psc/manifest.json `
  fd_psc.anchor_data.data_path=C:/abs/data/fd_psc/anchor.json `
  fd_psc.checkpoint.state_directory=C:/abs/fd_psc_runs/original/memory `
  fd_psc.checkpoint.latest_pointer_path=C:/abs/fd_psc_runs/original/memory/latest.json `
  fd_psc.checkpoint.resume_path=C:/abs/fd_psc_runs/original/memory/latest.json `
  hydra.run.dir=C:/abs/fd_psc_runs/resumed `
  +wandb_logging=false
```

`resume_path` 可以指向 latest JSON，也可以指向某个已验证的不可变 `state-*.pt`。加载顺序始终是基础模型 -> 目标 manifest/adapter -> FD-PSC state。`run_mode` 属于严格匹配项；`accumulate` 按上文恢复完整 adapter state，其他模式只恢复 persistent memory 并清空 episodic Pilot/Centered。默认保证 between-episode 恢复；除非环境状态和 planner-local warm start 也可序列化，不应承诺真实 rollout 的精确 mid-episode 恢复。

### 6.2 latest 损坏

不要手工修改 journal、latest 或版本文件。latest 丢失/损坏时，恢复逻辑只能扫描 `status=committed` 的 journal，校验其 base hash、target-manifest hash、版本文件 hash 和 state content hash，再选择 commit sequence 最大的完整版本并修复 pointer。若没有完整匹配版本，应报错而不是选择“看起来最新”的 `.pt`。

runner 自动为每次调用生成唯一 run ID，并拒绝覆盖既有目录，因此不同 variant/seed/重复调用不会共享可写 memory。显式 `--run-id` 也必须是新目录；续跑使用上面的直接 `plan.py` 流程，不属于全新基线/消融比较。

## 7. Episode、proposal 与故障语义

正常 episode 的状态流是：

```text
begin_episode
  -> 原 MPC feedback 时机执行 episodic online update
  -> sleep calibration（可比较多个候选，但不可读 commit-query）
  -> 冻结至多一个 final proposal
  -> commit-query 一次
  -> 原子 commit 或 reject/rollback
  -> 清理 episodic/local state
```

必须按下列语义解释状态：

- soft-NESS coefficient grid 只可由 commit-query 之前的三个确定性信号裁剪：current calibration 与相关 history 的全局加权 effective-gradient cosine、当前 frozen context prototype 与相关 history window 的最近 cosine、以及 theta0-only `theta0_jepa_pattern_v1` residual pattern 与相关 history 的最近 cosine。只有非 `null` 阈值对应的信号参与；任一已配置信号缺失或 match/conflict 信号互相矛盾时保留完整 grid。conflict 仅保留配置中属于 `{0,0.1,0.25}` 的 `alpha_shared`，match 仅保留属于 `{0.25,0.5,0.75,1}` 的值；`alpha_safe` 不受影响。这一步只减少 calibration 候选，绝不形成 proposal 或读取 commit-query；`soft_ness_enabled=false` 仍是完整 task vector、系数 1 的字面消融；
- 每个 quick/spectral 候选与 repair 的累计 screening clone 都保留自己的未截断 factor-space merge，并在该候选持久状态与候选路由下重新采集 calibration `H_l`。压缩 rank 与候选激活迭代到因素/秩固定点才允许进入 Gate 5/6；cycle 或未收敛直接拒绝；最大 rank 仍不可行的候选保留其最大允许 rank 截断作为受限 repair 种子，但不能走 quick path 或直接提交。不能复用 Pfast 的 `H_l`。功能误差按平方 Frobenius 能量比 `||(M-Mhat)H^T||_F^2/(||MH^T||_F^2+epsilon)` 计算；
- repair 的 LPR 在固定 old replay 上缓存 Pbefore 的真实配置 key-layer forward 输出；每步用 candidate 持久状态重新前向并直接约束 `h_l(candidate)-h_l(before)`。因此上游 adapter、base 层和非线性导致的输入变化都在约束内；关闭 LPR 或 old replay 为空时不建立该缓存；
- Gate 4 的 before 与每个 candidate 都从同一个 per-episode RNG snapshot、同一个 `plasticity-probe-paired` Pilot 初始化和相同 optimizer 参数组开始；probe 外部的 RNG、adapter 和 module mode 在每次比较后恢复。`sdc.event_triggered=false` 的 always-on 消融从 probe 的第一个 optimizer step 起复用 online effective-gradient hook 与精确 two-pass SDC helper，而不是对 A/B 梯度做近似；
- event-triggered SDC 只在 `check_every_replans` 的 scheduled check 上评估固定 immutable anchor。`before` 定义为 episode-start 的固定路由持久状态加零函数 Pilot，`current` 定义为本次 finetune 后的 live episodic 状态；同一组 anchor records 与 `before` loss 在 episode 内缓存，之后每次只重算 `current`，并将 `current-before` 与每层独立的 `rho_anchor` 一起传给 tracker。评估使用 frozen eval 且恢复 RNG、module mode、adapter 参数身份/数值和 requires-grad；缺失或非有限 anchor 直接 fail-closed；
- 一个 replan 内配置了多个实际 optimizer step 时，每一步完成后立即评估一次冲突 trigger；第 1 步若激活 Centered/SLICE，则只在确有下一步时按原 optimizer 类型和 predictor/encoder LR 分组重建 optimizer，使新增 Centered 参数从第 2 步开始受训。finetune event 末尾不重复评估 trigger，SDC event 更新与 replan index 仍只推进一次；
- 没有完成任何原定 online update、support 不足以构造 JEPA loss、或 task vector 在容差内为零：`REJECT_NO_PROPOSAL`。这是正常终态；不读取 commit-query、不创建 exception、不更新 replay/Q；
- calibration 阶段可以 repair 或选择其他候选，但一旦 final proposal 冻结，commit-query 只能调用一次；失败后不得根据结果换候选、再 repair 或创建 exception；
- 任何启用 gate 返回 fail、异常、非有限值或必需数据缺失，proposal 都拒绝；不能把异常静默记为 pass；
- 初次 slow commit 前 historical replay 为空时，history gate 可以是带原因的 `not_applicable`；第一次 slow commit 后由 sidecar 持久化的成功计数锁定这一事实，即使有界 replay 当前为空也不能重新进入冷启动；采样失败或损坏必须 fail；
- 单个 replan 的 support 太短时，只能与同 episode/context/preprocess/schema、同轨迹、replan 序号连续、共享稳定边界 frame ID 且边界观测逐位相同的相邻 segment 拼成完整窗口；组合 identity 必须再次对全部 external splits 做 trajectory/transition/frame/content 四层审计，任何不连续、歧义或伪造边界都 fail-closed；
- slow commit 成功后，support 才能进入 global replay 并更新 Q/activation subspace；普通拒绝、slow 失败或 exception commit 不更新这些全局量；
- exception commit 只原子更新对应 adapter、prototype、usage 和有界 local replay，不修改 global slow/replay/Q；context 与 residual prototype 同时持久化 float32 raw descriptor sum 和独立 count，replace 从 raw sum 更新后再归一化，不能用已归一 prototype 反推 running mean。router schema-1 checkpoint 显式按 `prototype × count` 构造兼容统计，随后保存为 schema 2；
- checkpoint 写入或 canary 失败属于事务失败，所有 persistent state 必须整体回滚；
- 周期 Gate-7 的 `before` 是上一次实际通过 canary 的 known-good 快照，而不是当前提交的 immediate previous；因此 `every_episodes=K` 可以一次验证并在失败时撤销该周期内的多次提交。若第 K 个 episode 没有可提交 proposal，则在边界后的第一次成功候选上补跑，不把周期静默延长到下一个整除点；
- known-good 算法内存、其 commit/episode 序号和待验证 commit ID 链随 sidecar 持久化。周期失败会恢复 slow/replay/Q/router/gradient reference/repair sampler/RNG 等算法内存，但保留已经消费的 external query、gate invocation、当前 episode 计数和单调 commit-ID 高水位；随后写新的 `rollback-XXXXXXXX` 不可变 sidecar 并令 latest 指向它。被撤销的 `commit-*` journals 标记为 `rolled_back`，旧版本保留审计证据但不能再显式加载或参与 recovery；
- `_plan_single()` 异常时 abort 且不 sleep；FD 模式下最终 reset 只清 episodic/local state，不能覆盖 persistent state；
- canary 关闭或环境不支持确定性 reset 时记录 `unrun` 及原因，不能等同于 canary pass 或规划成功。

## 8. 基线与消融

所有命名比较来自 `conf/fd_psc/experiments.yaml`，统一通过 `scripts/run_fd_psc_experiment.py --variant <name> ...` 运行。

### 8.1 基线

| variant | 含义 |
|---|---|
| `frozen_adajepa` | 冻结 ADAJEPA，`planner.adapt.steps=0` |
| `selected_layer_adajepa` | 原仓库 selected-layer ADAJEPA |
| `full_depth_episode_reset` | 相同目标/rank 的 full-depth episodic，episode 后丢弃 |
| `full_depth_accumulate` | 固定 rank live adapter 跨 episode 累积，不做 sleep/bank；完整 adapter state 随 episode snapshot 保存/恢复 |
| `episodic_slow_plain_svd` | episodic+slow 普通 factor-space SVD，无 soft-NESS，标记 unsafe；slow 更新与启用的 sidecar 写入同事务，写失败回滚 |

### 8.2 核心消融名称

- soft-NESS/gates：`without_soft_ness`、`without_current_history_anchor_gates`、`without_plasticity_gate`；
- SLICE/几何：`without_triggered_slice`、`slice_exact`、`slice_symmetric`、`c_pcgrad`、`dual_constraint`、`per_step_pcgrad_negative_control`；
- SDC：`sdc_off`、`sdc_always`、`sdc_event_triggered`；
- spectral surgery：`spectral_off`、`spectral_output_only`、`spectral_all_layers`；
- replay/repair：`replay_uniform`、`replay_grasp`、`repair_jepa`、`repair_jepa_lpr`；
- 冲突处理：`reject_conflict`、`exception_adapter`；
- slow rank：`slow_rank_8`、`slow_rank_16`、`slow_rank_adaptive`。

批量实验时每个 variant、seed 都应从同一个只读 base checkpoint 和全新隔离 memory 开始，保持相同 episode 顺序、support、六个 external split、canary budget 和真实 rollout budget。超参数与跨 run model selection 只能看 calibration；不能查看 commit-query 来选择 seed、variant、checkpoint 或超参数。

关闭主方法 Gate 1–6 任一项的配置必须同时设置 `gates.allow_unsafe_ablation=true`，并在运行名和结果元数据中标出被关闭的 gate。此类结果不能标为完整 FD-PSC。

上述 accumulate round-trip 与 plain-SVD checkpoint fault rollback 是本地 CPU mock 自动化测试覆盖的协议证据，不是实际任务效果证据。真实 checkpoint、真实 MPC rollout、canary、性能和显存是否已运行，以 [`fd_psc_implementation_report.md`](fd_psc_implementation_report.md) 的 `PASS/FAIL/UNRUN` 记录为准；不得从 sidecar 测试推断规划成功或吞吐/显存结论。

## 9. 指标与报告口径

至少分开记录并报告以下指标，不能合并或改名掩盖数据用途：

- 当前适应：`current_jepa_loss`、`fast_adaptation_gain`、`external_calibration_gain`、`commit_query_gain`、`report_test_gain`、真实 rollout `planning_success`、`time_to_threshold_replans`；
- 历史保持：historical replay loss、每个 context 的 before/candidate loss、worst-context regression、forgetting、backward transfer；
- 基础能力与可塑性：anchor loss/gradient cosine、canary planning regression、one-update-event plasticity gain、下一 episode 前三步 loss decline、plasticity gate ratio；
- 梯度几何：`rho_history`、`rho_anchor`、per-layer/global EMA、SLICE trigger、gradient correction norm、dual-constraint active set/count；
- 谱与子空间：base spectral drift `D_l`、episodic/slow/Q rank、`lambda`/soft-`p` 分布、最终 proposal 的 retained factor spectral-energy fraction、functional error、`alpha_shared`、`alpha_safe`；激活协方差特征值总和另记为 `activation_energy_total`，不得冒充 rank-selection spectral-energy fraction；
- 成本与协议：online/gradient/SLICE/sleep latency、replay/checkpoint bytes、adapter parameter count、exception count、route id/similarity/rejection count、calibration candidate count、final proposal type、commit-query invocation count、rollback count，以及每个 gate 的 `pass/fail/not_applicable` 理由。

确实不可用的指标使用 `value=null` 并在 tags 中写明 `status`（如 `unavailable`、`not_applicable`、`pending`、`not_reached`）和原因；不得用 `0` 或 NaN 伪造观测。`lambda`、`p` 和 Q 更新发生在提交事务候选态时标为 provisional；事务失败会回滚算法状态，但诊断事件仍保留为 provisional，报告端必须结合该 episode 的最终 outcome/journal，不能把它当成 committed state。最终 proposal 的 rank/spectral 指标也明确标记为 final gate 前 provisional。

无 proposal 时 calibration candidate/final proposal/commit-query invocation 应为 0；有唯一 proposal时 commit-query invocation 应为 1。同一 gate 内对 before/fast/candidate 的三次 forward 仍算一次 proposal gate，不是三次。

最终表格只能用 `report_test` 或独立真实 rollout。真实规划 success 只能由实际环境 rollout 统计，JEPA loss 改善、commit-query 通过或 canary 未运行都不能替代 success。建议报告均值、离散度/置信区间、每个 seed、episode 数、context 分布，并明确：

- base checkpoint 文件及 SHA-256；
- external manifest ID/hash、preprocess hash、latent adapter schema 和泄漏审计结果；
- runtime target manifest/hash，以及无 projection head 时的 `not_applicable`；
- 完整 Hydra 配置、variant、seed、episode 顺序、代码版本和命令；
- Python/PyTorch/CUDA/cuDNN、GPU/CPU 型号、精度、确定性设置；
- canary/真实 smoke 是否执行，未运行原因；
- 所有失败、reject、rollback 和数据缺失计数。

`run_metadata.json` 保存 runner 展开的实际命令和 manifest 文件 hash；`results.csv` 只适合调度审计。不要直接把它当作论文结果 CSV。

## 10. 常见启动与运行错误

| 现象 | 处理 |
|---|---|
| manifest base hash mismatch | 对实际加载的 `model_<epoch>.pth` 重算 hash，并用同一 checkpoint 重新生成 manifest |
| split checksum/schema mismatch | 恢复原始不可变 split；数据内容有变化时生成新 manifest/version，不要手改 hash |
| external split leakage | 为冲突 trajectory/transition/frame/content 重新划分；不能只改 record ID 掩盖重叠 |
| context 缺失 | 在 episode 开始前提供稳定 `context_identifier`，并为该 context 准备完整外部集合 |
| frozen backbone 被命中 | 检查 target manifest；任何 `base_model`/等价 frozen 子树目标都应视为错误 |
| projection target 为零 | 无 projection head 时合法；有 head 时检查 dry-run reachability 和 Linear/Conv2d 识别并 fail-fast |
| sidecar 无法恢复 | 确认 base/target manifest/config 相同；使用已验证 latest 或 immutable version，勿跳过 hash/shape 校验 |
| latest JSON 损坏 | 保留整个 memory 目录，让 journal/hash 恢复逻辑选择最近完整 commit；不要按文件时间手选 |
| rank 超过层维度 | 使用 target manifest 中的实际裁剪 rank/scaling；压缩仍超上限则 repair/exception/reject，不能静默提交 |
| gate 指标 NaN/Inf | 该 proposal 必须 fail 并保持 persistent state 不变 |
| canary 环境不可复位 | 记录 `unrun` 和限制；不要宣称 canary 或真实 planning success 已验证 |

在提交实验结果前，至少重新运行完整离线测试、一次 `--dry-run`、一个禁用兼容运行，以及具备本地资源时的一次真实 checkpoint/规划 smoke。所有命令和未运行项都应随实现报告保存。
