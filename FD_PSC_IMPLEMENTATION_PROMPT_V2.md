你现在要在官方 AdaJEPA 代码库上实现一个完整的跨 episode 持续记忆系统：

# FD-PSC

**Full-Depth LoRA Gradient–Spectral Projected Sleep Consolidation**
**全深度 LoRA 梯度—谱投影式睡眠巩固**

不要只给设计、伪代码或局部补丁。需要完成：

* 可运行实现；
* 配置文件；
* 单元测试；
* 集成测试；
* 基线和消融脚本；
* 状态持久化；
* 日志与诊断指标；
* 使用文档；
* 实现报告。

在开始修改前，先完整审计现有仓库，确认实际模型结构、训练流程、参数选择、buffer、checkpoint 和 MPC 调用链。不要根据下面的建议虚构不存在的类名或模块路径；应当将设计映射到仓库真实结构，同时保持以下算法语义不变。

本规格已经根据官方仓库的真实实现作出以下不可再自行更改的映射决策：

1. **保留在线学习语义，但允许替换参数集合。** 必须保留原 AdaJEPA 的 JEPA loss、target stop-gradient、plan–act–adapt–replan 时机、每次更新步数、predictor/encoder 学习率比例、recent buffer 行为以及每次 finetune 调用重建 optimizer 的周期。原 selected-layer 稠密参数集合明确替换为本规格定义的全深度 episodic LoRA 参数集合；不要求继续训练原 LayerNorm、bias 或其他稠密参数。
2. **post-backbone projection head 使用 ConvLoRA。** 永久冻结 visual backbone。对 frozen backbone 之后、JEPA latent 之前的 projection head，递归注入 Linear LoRA 和 ConvLoRA；ConvLoRA 默认启用。严禁向 `base_model` 或等价 frozen-backbone 子树注入 adapter。
3. **calibration/commit-query 来自外部固定数据。** 当前 episode 的在线执行轨迹只作为 support，并继续遵守原 recent-buffer 语义。SLICE 与候选选择只使用预先生成、版本固定的 external calibration；最终 commit gates 只对 calibration 已选出的唯一 proposal 使用匹配的 external commit-query。二者均与 support 不重叠，禁止从已经参与在线更新的窗口中事后划分 calibration 或 commit-query。
4. **episode 映射。** 官方 planner 中一个 `_plan_single` planning sample 定义为一个 episode。episode 开始只重置 episodic 状态；episode 结束后执行 sleep。slow LoRA、historical replay、anchor、激活子空间和 exception adapters 跨 planning sample 持久化。

---

# 0. 名称与变量说明

实现和文档中统一使用下列名称。

* **FD-PSC**：Full-Depth LoRA Gradient–Spectral Projected Sleep Consolidation，全深度 LoRA 梯度—谱投影式睡眠巩固。
* **LoRA**：Low-Rank Adaptation，低秩适配。
* **ConvLoRA**：应用于 post-backbone projection head `Conv2d` 权重的低秩适配。
* **SLICE**：Gradient-Surgery-based Low-rank Initialization for Continual Learning，持续学习梯度手术低秩初始化。
* **PCGrad**：Projecting Conflicting Gradients，冲突梯度投影。
* **SDC-LoRA**：Singular-Subspace Drift Controlled LoRA，奇异子空间漂移控制 LoRA。
* **NESS**：Null-space Estimated from Small Singular Values，小奇异值零空间估计。
* **SVD**：Singular Value Decomposition，奇异值分解。
* **QR**：QR Decomposition，正交—上三角分解。
* **LPR**：Layerwise Proximal Replay，逐层近端回放。
* **MPC**：Model Predictive Control，模型预测控制。
* **JEPA**：Joint-Embedding Predictive Architecture，联合嵌入预测架构。

数学状态：

* θ₀：永久冻结的 AdaJEPA 基础模型。
* Δˢ：单个全局慢速 LoRA。
* Δᵉ：当前 episode 的快速 LoRA。
* Δˣ：当前 context 路由到的可选 exception adapter；无匹配时为零。
* 𝓑episode：当前 episode 的 recent buffer。
* 𝓜hist：跨 episode 历史 replay。
* 𝓜anchor：不可变基础能力 anchor。
* Qₗ：第 l 层历史输入激活子空间。
* λₗ：Qₗ各方向的历史激活能量。
* Uₗ⁰、Vₗ⁰：第 l 层基础权重的主要左右奇异子空间。

---

# 1. 第一阶段：仓库审计

修改代码前必须完成并记录以下审计。

## 1.1 找到真实训练入口

确认并记录：

* AdaJEPA trainer 类；
* online adaptation 的入口；
* 每个 MPC replanning point 如何触发更新；
* prediction loss 的实现；
* stop-gradient 的位置；
* replay buffer 的结构；
* optimizer 的创建和重置；
* checkpoint、snapshot 和 episode reset；
* predictor 与 sensory encoder 的参数选择逻辑；
* visual backbone、projection head、predictor、action encoder、proprio encoder 的真实模块路径。

同时确认 `AdaJEPAMPCPlanner.plan()`、`_plan_single()` 和 `_post_env_feedback()` 的实际边界。FD-PSC 开启时，一个 `_plan_single()` 调用就是一个 episode；不得让现有 sample reset 清除 slow memory。

在官方 `plan()` 的逐 sample 循环中建立明确 lifecycle：在调用 `_plan_single()` 前 `begin_episode()`；运行期间只由原 `_post_env_feedback()` 时机触发 online update；仅当 `_plan_single()` 正常返回后、且 `_obs_buffer/_act_buffer` 尚未清空时调用一次 `end_episode_and_sleep()`；随后清除 episodic 和 planner-local buffer。若 `_plan_single()` 抛异常，则执行 abort/rollback、清除 episodic/local state 且不得生成 sleep proposal。`plan()` 末尾原本用于恢复稠密 snapshot 的 `reset()` 在 FD 模式下只能变成 episodic cleanup，不能覆盖 persistent state；FD 关闭时仍执行原逻辑。

## 1.2 枚举所有目标 Linear/Conv2d 层

递归列出：

* predictor 主 forward 路径内所有 `torch.nn.Linear`；
* frozen visual backbone 之后、JEPA latent 之前的 projection head 内所有 `torch.nn.Linear` 和 `torch.nn.Conv2d`；
* action encoder 和 proprio encoder 内的线性层，但默认不启用；本规格的 ConvLoRA 仅覆盖 post-backbone `Conv2d` projection head；
* 显式排除 `base_model` 或审计报告识别出的等价 frozen-backbone 子树；
* 对只声明但未在 JEPA 主 forward 路径使用的层标记 `active_in_forward=false`，默认不得注入。

输出一个机器可读清单，包含：

* 完整模块路径；
* 层类型：Linear 或 Conv2d；
* 输入维度；
* 输出维度；
* Conv2d 的 kernel、stride、padding、dilation 和 groups；
* 所属模块组；
* 是否为 attention output；
* 是否为 MLP output；
* 是否为最终 projection；
* 默认是否注入 LoRA。

官方 predictor 使用融合 `to_qkv` Linear；该层必须标记为 `attention_qkv`，不得虚构成三个独立模块。

manifest 在加载真实 base checkpoint 后、注入前生成。先按模块树识别候选，再用一个不参与任何训练/评估 split 的 schema-only dry-run sample 记录实际调用路径；dry-run 只验证 reachability，不保存数据或改变 buffer/RNG。checkpoint 没有 projection head 时 projection target 数为零是合法 `not_applicable`；存在 head 却未识别到其活跃 Linear/Conv2d 时必须 fail-fast。

## 1.3 确认 latent replay 切点

优先选择：

> frozen visual backbone 输出之后、可更新 projection head 之前。

确认该切点在模型中稳定存在。保存该模块路径到配置中，不要硬编码类名。

不能只依赖 hook 路径完成 replay。为每种受支持 encoder 提供显式接口或 adapter protocol：

* `extract_frozen_visual_latent(obs)`；
* `project_visual_latent(latent)`；
* latent layout、shape、dtype 和 schema version；
* unsupported encoder 必须 fail-fast，并输出可操作错误。

若 encoder 没有 post-backbone projection head，`extract_frozen_visual_latent()` 返回最终 frozen visual latent，`project_visual_latent()` 为显式 identity，此时 projection LoRA targets 合法为空。若存在 Conv projection，adapter 必须保存/恢复 token↔feature-map layout，而不是仅保存扁平 tensor。

## 1.4 兼容性要求

当 `fd_psc.enabled=false` 时：

* 模型输出必须与原仓库一致；
* 原训练流程必须保持可用；
* 原有 checkpoint 必须仍能加载；
* 不得改变原 AdaJEPA 默认行为。

关闭状态不得执行 adapter 注入、替换模块、注册 FD-PSC hooks 或改变 state_dict key。输出兼容测试必须在相同 model mode 和相同 RNG state 下比较。官方 predictor 当前把 causal attention mask 直接放到 CUDA；为满足 CPU fallback，可以按输入 device/dtype 动态构造，或注册为 `persistent=False` 的 device-agnostic buffer，但不得新增持久 state_dict key，并必须证明 GPU 数值输出和旧 checkpoint 加载兼容。

在实现报告中写明实际修改文件、类和函数。

---

# 2. 硬约束

以下约束不可违反。

## 2.1 θ₀永久冻结

基础参数：

* 不进入 optimizer；
* `requires_grad=false`；
* θ₀包含影响基础函数的全部参数和持久 buffer；所有基础参数与 buffer 必须逐 tensor bitwise 不变；BatchNorm running statistics 等基础 buffer 不得在 episode 中漂移；
* 不允许 weight absorption 修改；
* 不允许 sleep 将 LoRA merge 进基础权重；
* 必须能够随时禁用所有 LoRA 并恢复 θ₀原始函数。

## 2.2 保留原 AdaJEPA 在线更新语义

保留仓库实际使用的：

* JEPA prediction loss；
* target stop-gradient；
* plan–act–adapt–replan；
* replanning 时机；
* 每次更新步数；
* recent buffer；
* action 和时间窗口构造；
* predictor 与 encoder 的相对学习率。

这里的“保留”指训练与 MPC 语义，不指保留原 selected-layer 稠密参数集合。必须明确执行以下替换：

* 原 predictor selected-layer 稠密更新替换为 predictor 主 forward 路径全部目标 Linear 的 episodic LoRA 更新；
* 原 encoder-head 稠密更新替换为 post-backbone projection head 全部目标 Linear/Conv2d 的 episodic LoRA/ConvLoRA 更新；
* 不再训练原 LayerNorm、bias、BatchNorm affine 参数或其他稠密参数；
* predictor 与 projection head 仍使用原 predictor/encoder 学习率及其比例；
* online optimizer 的算法、betas/epsilon、weight decay、gradient clipping、scheduler、AMP/scaler 行为均沿用仓库原路径；只把 param groups 中的原稠密 selected parameters 替换为对应组的 episodic adapter parameters，除本规格明确要求的有效梯度预处理外不得顺便改训练超参数；
* 每次原 `finetune()` 调用仍重建 optimizer；SLICE 分支切换后必须在下一次实际 optimizer step 前重建 optimizer，且不得因为切换额外增加在线更新步数；
* `fd_psc.enabled=false` 时完全走原 `AdaJEPATrainer` 参数选择和 reset 路径。

只对上述 episodic adapter 参数执行在线更新，并加入可配置的梯度预处理。

FD 模式不得复用会把整个 module 的 `requires_grad` 或 snapshot 一锅端处理的旧逻辑：optimizer param groups 只从显式 episodic adapter registry 构造；slow/exception/base 永不进入 online optimizer；episode reset 只清除 episodic/trigger/optimizer-local state，不恢复或覆盖 slow、exception、replay、Q/λ。原 snapshot/reset 仅在 FD 关闭路径使用。

不要擅自改变 MPC、loss 或轨迹窗口定义。

## 2.3 不重新预训练 AdaJEPA

不引入：

* MAML；
* La-MAML；
* MER；
* TTT-E2E 元训练；
* 需要修改原预训练目标的方法。

只实现运行时在线学习和 episode 间 sleep 巩固。

## 2.4 不进行常态化蒸馏

主路径不得依赖：

* fast teacher；
* old teacher；
* frozen teacher；
* 多教师输出缓存；
* 全模型蒸馏。

全深度 LoRA 已直接将知识写入多个深度。

## 2.5 不建立常态化 adapter bank

普通 episode 只写入单个 Δˢ。

只有所有合并与 repair 均失败、但当前 episode 收益明确时，才建立 exception adapter。

---

# 3. 配置系统

新增独立配置组 `fd_psc`，所有高级组件必须可单独开关。

至少包含：

```yaml
fd_psc:
  enabled: true
  seed: 0

  target_modules:
    predictor_linear: true
    post_backbone_projection_linear: true
    action_encoder_linear: false
    proprio_encoder_linear: false
    exclude_frozen_backbone: true
    require_active_forward_path: true
    fail_on_empty_predictor_targets: true
    fail_on_empty_projection_targets: false
    require_projection_targets_if_head_exists: true

  episodic_lora:
    rank: 8
    alpha: 16
    dropout: 0.0
    pilot_enabled: true
    a_initialization: kaiming_uniform
    b_initialization: zeros

  conv_lora:
    enabled: true
    target_scope: post_backbone_projection_head
    parameterization: flattened_kernel
    groups_mode: groupwise

  slow_lora:
    initial_rank: 8
    allowed_ranks: [8, 16, 24, 32]
    maximum_rank: 32
    spectral_energy_threshold: 0.99
    functional_error_threshold: 0.02

  gradient_geometry:
    enabled: true
    ema_beta: 0.8
    conflict_threshold: -0.1
    minimum_transitions: 3
    consecutive_conflicts: 2
    current_batches: 2
    history_batches: 2
    anchor_batches: 1
    windows_per_batch: 8
    projection_method: dual_constraint
    projection_scope: per_logical_layer
    global_cosine_weighting: gradient_norm
    hook_normalization: exact_loss_gradient
    epsilon: 1.0e-8
    c_pcgrad_coefficient: 1.0
    history_slack: 0.0
    anchor_slack: 0.0

  slice:
    enabled: true
    trigger_only: true
    initialization: slice_exact
    fallback_initialization: slice_symmetric
    rank: 8
    randomized_svd_oversampling: 2
    power_iterations: 1
    magnitude_mode: first_step_match
    maximum_scale: 10.0

  sdc:
    enabled: true
    event_triggered: true
    check_every_replans: 4
    base_energy_threshold: 0.9
    drift_threshold: 0.25
    drift_consecutive_checks: 2
    drift_increase_tolerance: 0.01
    minimum_gamma: 0.1
    anchor_regression_trigger: 0.0

  spectral_surgery:
    enabled: true
    output_writing_layers_only: true
    steps: 2
    learning_rate: 0.1
    minimum_scale: 0.75
    maximum_scale: 1.25
    preserve_spectral_l2_norm: true
    current_weight: 1.0
    history_weight: 1.0
    anchor_weight: 1.0

  activation_subspace:
    enabled: true
    maximum_rank: 64
    spectral_energy_threshold: 0.99
    forgetting_factor: 0.99
    soft_ness_tau_mode: median
    soft_ness_tau_fixed: null
    soft_ness_tau_quantile: 0.5
    minimum_energy: 1.0e-8

  merge:
    soft_ness_enabled: true
    shared_coefficients: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    safe_coefficients: [0.5, 0.75, 1.0]
    use_context_similarity: true
    use_gradient_similarity: true
    use_residual_similarity: true
    context_conflict_threshold: null
    context_match_threshold: null
    gradient_conflict_threshold: -0.1
    gradient_match_threshold: 0.1
    residual_match_threshold: null
    selection_policy: calibration_lexicographic

  replay:
    historical_windows: 512
    visual_latent_dtype: float16
    auxiliary_dtype: float32
    compression: none
    sampling: balanced_uniform
    repair_sampling: grasp
    maximum_context_clusters: 32
    new_cluster_similarity_threshold: 0.8
    minimum_windows_per_cluster: 4

  external_eval_data:
    source: fixed_manifest
    manifest_path: null
    calibration_path: null
    commit_query_path: null
    plasticity_support_path: null
    plasticity_query_path: null
    report_test_path: null
    representation: frozen_backbone_latent
    context_key: context_identifier
    context_source: episode_metadata
    split_unit: trajectory
    require_context_match: true
    verify_checksums: true
    missing_context_policy: error
    commit_query_policy: single_final_proposal

  anchor_data:
    source: fixed_manifest
    manifest_path: null
    data_path: null
    windows: 128
    verify_checksums: true
    missing_policy: error

  repair:
    enabled: true
    maximum_steps: 20
    candidate_steps: [5, 10, 20]
    optimizer: adamw
    learning_rate: 1.0e-4
    windows_per_batch: 8
    current_weight: 1.0
    replay_weight: 1.0
    proximal_enabled: true
    proximal_weight: 1.0
    pcgrad_enabled: true
    checkpoint_schedule: cumulative
    proximal_layer_tags: [encoder_projection, attention_output, mlp_output, final_projection]

  exception:
    enabled: true
    maximum_adapters: 8
    routing: nearest_prototype
    minimum_route_similarity: 0.5
    no_match_behavior: slow_only
    minimum_calibration_fast_gain: 0.0
    minimum_commit_fast_gain: 0.0
    maximum_rank: 32
    local_replay_windows: 64
    routed_episode_update: replace_exception
    eviction_policy: least_recently_used_then_lowest_gain
    merge_similar_adapters: false

  gates:
    allow_unsafe_ablation: false
    current_gain_enabled: true
    history_enabled: true
    anchor_enabled: true
    plasticity_enabled: true
    functional_error_enabled: true
    spectral_drift_enabled: true
    fast_gain_retention: 0.8
    history_loss_tolerance: 0.0
    anchor_loss_tolerance: 0.0
    worst_context_loss_tolerance: 0.0
    plasticity_retention: 0.9
    drift_tolerance: 0.05
    absolute_numerical_tolerance: 1.0e-6
    relative_numerical_tolerance: 1.0e-5

  canary:
    enabled: false
    every_episodes: 10
    rollout_count: 4
    manifest_path: null
    high_risk_rank_expansion: true
    unavailable_policy: report_unrun

  checkpoint:
    enabled: true
    state_directory: fd_psc_state
    latest_pointer_path: fd_psc_state_latest.json
    resume_path: null
    save_every_episodes: 1
    retention_versions: 20
    keep_commit_journal: true
    atomic_write: true

  logging:
    per_layer_metrics: true
    save_candidate_reports: true
    save_gradient_statistics: true
```

配置字段应使用 dataclass、Pydantic 或仓库现有配置系统，提供类型检查和默认值。

`fd_psc` 可以作为独立 Hydra config group 保存，但实例化 planner 时必须显式传入 `AdaJEPAMPCPlanner`；不得创建一个 planner 永远收不到的顶层孤立配置。配置加载后必须验证外部 manifest、calibration/commit-query/plasticity 文件、checksum、context 映射和所有目标模块。

配置验证还必须执行以下约束：

* rank、容量、步数和频率为非负整数；allowed ranks 严格递增、去重且不超过 maximum rank；每层实际候选 rank 自动裁剪到该 logical layer 的矩阵维度；
* LoRA alpha、学习率、epsilon 和 scale 上限有限且为正；dropout、EMA/forgetting factor、谱能量阈值和相似度阈值处于合法区间；slack/tolerance 非负；coefficient grid 非空且全部有限；
* `c_pcgrad_coefficient` 默认限制在 `[0,1]`；超出范围只允许显式标记的消融配置；
* `episodic_lora.rank`、`slice.rank` 和启用 adapter 的 rank 上限必须为正；容量可以为零的字段要有明确禁用语义。`slice.enabled=true` 时必须同时有 `episodic_lora.pilot_enabled=true`；
* 当 gradient geometry、SLICE 或 SDC 任一启用时，episodic LoRA dropout 必须为 0，保证单一有效权重梯度和 canonical task vector 语义；非零 dropout 只允许作为关闭这些组件的显式消融，并在报告中注明不再满足同一有效矩阵假设；
* `slice.enabled` 依赖 gradient hooks、external calibration 和 episodic LoRA；
* `sdc.enabled` 依赖基础谱缓存和在线有效权重梯度；
* `spectral_surgery.enabled` 依赖 external calibration；
* `activation_subspace.enabled` 和 shared/safe merge 依赖 historical replay；
* final commit-query gate 依赖 `commit_query_path`；plasticity gate 依赖独立 plasticity support/query；真实实验指标依赖 report-test 或真实 rollout；
* `repair.candidate_steps` 必须严格递增、去重，末项不超过 `repair.maximum_steps`；`proximal_enabled=false` 时不得仍计算或缓存 LPR activations；
* `gradient_geometry.windows_per_batch` 和 `repair.windows_per_batch` 必须为正；采样不足时是否允许有放回必须按对应章节执行并记录，不能因 batch 大小隐式丢掉 context；
* `checkpoint.retention_versions` 必须至少覆盖最近一个启用 canary 周期内可能产生的全部持久提交，再多保留一个已知良好版本；不足时启动即报错，不能提前删除回滚链；
* `external_eval_data.commit_query_policy` 在合规主路径中只能是 `single_final_proposal`；其他值必须拒绝，不能作为普通配置覆盖硬约束；
* 主方法的 Gates 1–6 默认全部启用。关闭任一 gate 只允许同时设置 `gates.allow_unsafe_ablation=true` 的基线/消融运行，运行名、日志和结果元数据必须记录被关闭的 gate；不得把这种运行报告为完整 FD-PSC；
* `conv_lora.enabled` 是 post-backbone projection `Conv2d` 注入的唯一开关，默认 true；不得再维护第二个含义重复且可能冲突的 target flag。没有 projection head 的合法 checkpoint 可以得到零 projection targets，但如果 projection head 存在而其活跃 Linear/Conv2d 未被识别，必须报错；
* 所有在当前运行阶段启用/必需的路径必须非空、可读且 hash/schema 匹配；普通 online run 可不加载 report-test 和 disabled canary，但生成最终实验报告时 report-test/独立 rollout 必须存在；关闭 FD-PSC 时不得要求这些外部文件存在。

运行模式必须明确区分：

* online adaptation 和冲突梯度收集使用原 predictor train mode 与受控 RNG；frozen backbone 以及所有基础 BatchNorm 始终保持 eval；
* candidate calibration、commit gates、report-test 和 canary 使用 eval mode；
* 两次需要公平比较的计算必须保存并恢复 Python、NumPy、Torch CPU/CUDA RNG，使其使用相同 stochastic state，且比较结束后全局 RNG 只前进一次；
* external commit-query 是在线算法内部的安全 gate 数据，不得宣称为无偏最终测试集。`report_test` 或从未参与任何提交决策的真实 rollout 才能用于最终实验结论。

---

# 4. LoRA 层实现

新增类似 `DualLoRALinear` 的模块，但名称应符合仓库风格。

## 4.1 基础结构

对原线性层：

Wₗ⁰ ∈ ℝᵈᵒᵘᵗˣᵈⁱⁿ

实际权重为：

```text
Wₗeff = Wₗ⁰ + ΔWₗˢ + ΔWₗˣ + ΔWₗᵉ
```

基础层 bias 保持原样，不训练。

Linear adapter 前向必须按 `B(Ax)` 的 factorized 顺序计算 slow、active exception、Pilot/Centered 各分支，再与基础输出相加；生产前向不得先构造 `BA` 稠密矩阵。Centered 的减项按独立的 `B⁰(A⁰x)` 计算。只有受尺寸上限保护的单元测试/debug API 可以物化增量权重。

## 4.2 慢速 LoRA

ΔWₗˢ = sₛ Bₗˢ Aₗˢ

慢速 LoRA：

* 跨 episode 持久化；
* episode 内冻结；
* 只通过 sleep 原子替换；
* 必须支持动态 rank。

## 4.3 Episodic Pilot LoRA

episode 开始时：

ΔWₗpilot = spilot Bₗpilot Aₗpilot

默认：

* rank = 8；
* α = 16；
* `spilot = α ÷ rpilot,actual = 2`；
* A 随机初始化；
* B 零初始化。

初始有效增量必须严格为零。

若小 logical layer 的 rank 被维度裁剪，`spilot=alpha/rpilot,actual`；actual rank 必须写入 manifest、日志和 checkpoint，不能仍用请求 rank 计算 scaling。

A 默认使用 PyTorch Kaiming-uniform 语义，B严格 zeros；每个 `(global_seed, episode_id, logical_layer_id)` 通过稳定 hash 派生专用 generator，不能使用进程随机化的 Python `hash()`。Linear 与各 Conv group 的初始化可复现且互不复用随机流。

## 4.4 Centered LoRA

SLICE 触发后，不允许修改 θ₀来抵消非零初始化。

将当前 Pilot 更新冻结为：

ΔWₗpilot,fixed

初始化非零：

Bₗ⁰、Aₗ⁰

新的可训练分支为：

ΔWₗcentered(t)
= scentered [Bₗ(t)Aₗ(t) − Bₗ⁰Aₗ⁰]

初始化时：

Bₗ(0) = Bₗ⁰

Aₗ(0) = Aₗ⁰

因此：

ΔWₗcentered(0) = 0

其中 `scentered=alpha/rcentered,actual`，`rcentered,actual` 由 `slice.rank` 和该 logical layer 维度/数值可用方向裁剪。Pilot 与 Centered 请求 rank 恰好相同时二者 scaling 可以相等；配置允许二者不同时绝不能复用同一个 scaling。

总 episodic 更新：

```text
ΔWₗᵉ(t) = ΔWₗpilot,fixed + ΔWₗcentered(t)
```

要求：

* SLICE 切换瞬间模型输出连续；
* 切换前后相同输入的输出误差低于数值容差；
* 切换后重建 optimizer；
* 旧 Pilot 参数不再训练；
* Centered 参数进入 optimizer。

## 4.5 Rank 计算

普通 Pilot：

rank(ΔWpilot) ≤ r

Centered 分支：

rank(BendAend − B⁰A⁰) ≤ 2r

Pilot 加 Centered：

rank(ΔWepisode) ≤ rpilot + 2rcentered

不能将完整 episodic task vector 错误地当成 rank 8。

## 4.6 低秩 factor 导出

每层必须提供：

* `get_slow_factors()`
* `get_episodic_factors()`
* `get_effective_factors()`
* `reset_episode()`
* `freeze_pilot()`
* `activate_centered_branch()`
* `replace_slow_adapter()`
* `disable_all_adapters()`
* `adapter_state_dict()`

`get_effective_factors()` 返回 slow、当前路由 exception 和 episodic 的 canonical factor composition，不得在生产路径物化大矩阵。仅测试或显式小层 debug API 可以提供 `materialize_effective_delta()`，并必须受尺寸上限保护。

`disable_all_adapters()` 必须同时旁路 slow、active exception、Pilot 和 Centered，直接调用未改写的基础层路径；`reset_episode()` 只清除 Pilot/Centered 和 active route，不删除持久 exception bank。

`get_episodic_factors()` 必须返回一个低秩表示：

ΔWₗepisode = Bₗδ Aₗδ

不得要求构造完整稠密权重。

所有 factor 导出使用统一 canonical 约定：导出的 `BA` 必须已经等于实际有效增量，LoRA scaling `s = alpha / rank` 必须吸收到其中一个 factor。后续 Spectral Surgery、soft-NESS、QR/SVD merge、rank 选择和 checkpoint 一律只消费 canonical factors，不得再次乘 scaling。

Pilot + Centered 的 canonical episodic factors 必须通过拼接精确导出，而不是重新拟合：

```text
Bδ = [spilot Bpilot,fixed, scentered Bend, -scentered B⁰]
Aδ = [Apilot,fixed; Aend; A⁰]
```

只有 Pilot 时使用 `Bδ=spilot Bpilot, Aδ=Apilot`。允许把各分支自己的 scaling 对称吸收到两侧，但同一实现必须全局统一并由重构测试验证。

## 4.7 ConvLoRA

对 post-backbone projection head 的 `nn.Conv2d`，实现与 Dual LoRA Linear 相同的 slow、Pilot、Centered 和状态管理语义。

对基础卷积权重：

```text
W⁰ ∈ ℝ[dout, din/groups, kh, kw]
Wflat⁰ ∈ ℝ[dout, (din/groups)·kh·kw]
ΔWflat = B A
```

当 `groups=1` 时使用单组 factors。当 `groups>1` 时，第 g 组独立使用：

```text
Wflat,g ∈ ℝ[(dout/groups), (din/groups)·kh·kw]
ΔWflat,g = Bg Ag
```

最终按输出 channel 顺序拼回原 grouped-convolution 权重；不同 group 不得共享或交叉访问输入 channel。

生产代码将每个 grouped-convolution group 暴露为独立 logical layer id：`<module_path>::group=<g>`。每个 logical layer 独立维护 slow rank、Pilot/Centered factors、Q/λ、基础谱、梯度几何、merge 和 checkpoint 状态；禁止把不同 group 填零后伪装成一个可跨组混合的低秩矩阵。

生产前向不得物化 canonical `BA` 卷积核。每个 logical group 先用 A factors 执行具有原 kernel、stride、padding、dilation 和 padding_mode 语义、无 bias 的低秩卷积，再用 B factors 执行无 bias 的 `1×1` 卷积映射到该组输出 channels；最后按原输出 channel 顺序拼接。不同 group 的实际 rank 可以不同，此时允许逐组计算，不能为了调用单个 grouped conv 而填充出可跨组混合的参数。仅重构测试/debug API 可将 `BA` reshape 回卷积核。

要求：

* 基础卷积权重和 bias 永久冻结；
* `groups=1` 和 groupwise grouped convolution 均有明确实现；
* 非零 `padding_mode` 必须复用与原 `nn.Conv2d` 相同的预 padding 输入，再以零 internal padding 计算 base/delta；不得直接把字符串 padding_mode 丢给不等价的 `F.conv2d`；
* factorized A→B 两阶段输出必须与显式 reshape 后的 `F.conv2d` 增量在数值容差内一致；
* slow/Pilot/Centered 的 rank、连续性、factor 导出和 checkpoint API 与 Linear 版本一致；
* 禁用 adapter 时输出与原 Conv2d 数值一致；
* 不向 frozen visual backbone 注入 ConvLoRA；
* 运行时 manifest 必须证明每个 ConvLoRA 位于 post-backbone projection head 且参与 JEPA 主 forward。

---

# 5. LoRA 注入范围

默认替换：

* predictor 主 forward 路径中的全部 `nn.Linear`；
* post-backbone projection head 中的全部 `nn.Linear`；
* post-backbone projection head 中的全部 `nn.Conv2d`，使用 ConvLoRA。

默认不替换：

* frozen visual backbone 内的 Linear 和 Conv2d；
* LayerNorm；
* BatchNorm；
* bias；
* embedding；
* action encoder；
* proprio encoder。

若显式启用 `action_encoder_linear` 或 `proprio_encoder_linear`，只递归注入对应子树主 forward 路径内的 Linear，并归入原 encoder LR param group；仍不向其中的 Conv1d/Conv2d 或其他层型注入。启用后目标为空必须根据独立的 manifest/配置校验报错，不能静默把开关当作成功。默认主方法和必需验收仍以二者关闭为准。

为模块设置标签：

* `attention_query`
* `attention_key`
* `attention_value`
* `attention_qkv`
* `attention_output`
* `mlp_input`
* `mlp_output`
* `final_projection`
* `encoder_projection`
* `encoder_projection_conv`
* `other_linear`
* `other_conv`

Spectral Surgery 默认只作用于：

* `attention_output`
* `mlp_output`
* `final_projection`
* encoder projection head 的最后一层。

对于 fused `attention_qkv`，默认不把它误标为 `attention_output`。对于 ConvLoRA，只有 projection head 的最终输出卷积可标记为 encoder projection 的输出写入层并参与默认 Spectral Surgery。

---

# 6. 状态数据集合

必须区分 episode support buffer、global historical replay、immutable anchor、external calibration/commit-query/plasticity probes、report-test，以及 exception-local replay；不得混淆用途或生命周期。

## 6.1 Episode buffer：𝓑episode

继续使用仓库原 recent buffer：

* 服务当前 MPC 在线更新；
* 不混入大量跨 episode 历史样本；
* episode 结束后清空；
* 不替换为 GRASP。

## 6.2 持久 replay：𝓜hist

保存连续轨迹窗口，而不是打散 transition。每个窗口长度必须至少覆盖仓库实际 `num_hist + num_pred`，并保存完整连续 latent/proprio/action 序列；仅保存“当前/下一”二元组不足以重放现有滑窗 JEPA loss。

每个 replay item 至少包括：

* 完整连续 frozen-backbone latent 序列；
* proprioception；
* action 或 action chunk；
* 时间位置；
* trajectory boundary；
* history frame 信息；
* prediction mask；
* context identifier；
* frozen context embedding；
* prediction residual；
* contact 或 dynamics-change 标记；
* 来源 episode；
* 原始用途与 provenance：`episode_support`；
* 是否已成功提交并允许进入 historical replay。

visual latent 以配置的低精度存储；normalized proprio/action、mask、时间和 loss 计算所需的辅助数值默认保留 float32。所有样本记录 preprocess hash、基础 checkpoint hash、latent adapter schema 和原始稳定 IDs。

`frozen context embedding` 定义为 θ₀ frozen visual backbone 输出的确定性、L2-normalized pooling，不经过 slow/exception/episodic adapter。`prediction residual` 定义为 θ₀-only、eval-mode JEPA predictor 在同一 frozen 数据上的固定 residual descriptor；不得用随 slow state 变化的 residual 混合到长期相似度。环境不提供 contact/dynamics-change 时保存 availability mask，不能伪造负标签。

## 6.3 Anchor：𝓜anchor

不可变基础能力数据：

* 初始化时创建；
* 后续不能被普通 replay 替换；
* 单独持久化；
* 不参与 current support 训练；
* 用于基础能力 gate 和梯度约束。

Anchor 也必须由显式外部 manifest 和固定数据路径提供，不得从当前 commit-query、plasticity query、report-test 或当前 episode support 自动 bootstrap。manifest 记录 trajectory IDs、checksum、预处理版本和基础 checkpoint hash；缺失时按配置报错，不得静默拿 historical replay 代替。

## 6.4 外部固定 calibration/commit-query

当前 episode 的执行轨迹全部属于 support，继续进入仓库原 recent buffer 并服务在线更新。不得从这些已经训练过的轨迹中事后抽取 calibration 或 commit-query。

calibration/commit-query 必须来自外部、预先生成且固定版本的数据：

* calibration：用于 Gcur、Triggered SLICE、magnitude matching、Spectral Surgery、候选生成、repair screening 和 rank 功能误差；
* commit-query：只对 calibration 阶段已经选定的唯一 final proposal 执行一次最终 commit gates，不参与在线训练、repair、候选排序或失败后的第二次尝试；
* plasticity support/query：只用于 Gate 4 的临时一步适应，二者相互独立且不等于 commit-query；
* report-test：从不参与训练、候选选择、gate、routing threshold 调整或回滚决策，只用于最终离线报告；
* manifest 按 `context_identifier` 将当前 episode 映射到对应 calibration/commit-query 轨迹；
* split 单位默认是完整 trajectory，所有外部 split 不得共享 trajectory、transition、frame 或其衍生 latent；
* 文件 checksum、schema version、预处理版本、基础 checkpoint 标识和 frozen-latent 提取版本必须持久化；
* `require_context_match=true` 时没有匹配 context 必须报错并拒绝 sleep，不得静默改用 support；
* 同一实验的外部 split 在所有 baseline/ablation 中保持不变；
* `context_identifier` 必须在 episode 开始前由 evaluation manifest 或显式 environment/task metadata 提供，不能从 commit-query、report-test 或最终结果反推；`missing_context_policy=error` 时在 episode 第一次在线更新前 fail-fast。

## 6.5 数据隔离

必须验证以下集合两两不相交：

* 当前 episode support；
* 外部 calibration；
* 外部 commit-query；
* plasticity support/query；
* report-test；
* anchor。

隔离验证基于稳定 trajectory/transition/frame IDs 和内容 checksum，不能只比较 Python 对象或窗口编号。外部固定集合在启动时完成全量审计；每个新 support window 在进入第一次 online update 前再对全部外部集合做增量 ID/checksum 审计。若真实环境没有稳定 IDs，必须由 manifest 定义可复现的 environment/task/seed/timestep 标识并结合规范化内容 hash；无法证明隔离时 fail-fast，不能先训练后再报警。添加单元测试，任何泄漏都必须 fail-fast。

Global historical replay 和 exception-local replay 是成功提交后从 support 派生的持久副本，因此允许与其来源 support 具有相同 stable IDs，但必须保留 provenance，且永远不能被重新标记为 external calibration、commit-query、plasticity probe、report-test 或 anchor。

---

# 7. 有效权重梯度收集

θ₀被冻结，但需要得到每层有效权重梯度。

对于 PyTorch 线性层：

y = xWᵀ + b

若：

* x 展平后为 X ∈ ℝᴺˣᵈⁱⁿ；
* 输出梯度为 D ∈ ℝᴺˣᵈᵒᵘᵗ；

则：

G = ∂ℒ ÷ ∂W = DᵀX

实现 forward hook 和 full backward hook：

* flatten batch、time、token 等前导维度；
* 累积 float32；
* 默认累积 autograd 定义的 exact loss gradient；`D` 已包含 loss reduction，禁止再按 token 数重复除法；
* 可额外生成 per-token mean/sum 诊断副本，但用于 SLICE、SDC 和约束投影的三个梯度必须采用同一 exact normalization；
* 使用完立即释放缓存；
* 不长期保留 computation graph；
* 可按模块组或逐层统计；
* 同一模块多次 forward 或重入时使用 invocation stack/ID 正确配对输入与 grad_output，并对所有调用求和。

实现：

* Gcur：与当前 context 匹配的外部固定 calibration 数据；
* Ghist：历史 replay；
* Ganchor：基础 anchor。

必须写测试，将 hook 得到的 G 与一个临时可训练 `nn.Linear.weight.grad` 比较，误差在数值容差内。

梯度 batch 语义必须固定：每个 batch 含 `gradient_geometry.windows_per_batch` 个完整连续窗口；`current/history/anchor_batches` 是分别累积的 batch 数。external calibration 和 immutable anchor 按 manifest stable ID 无放回确定性采样，数量不足视为数据配置错误；historical replay 按 balanced context sampler 采样，容量不足时允许有放回并记录重复率，历史为空则 `Ghist` unavailable。多 batch 的 G 按实际参与 loss 的同一 reduction 聚合，不能先对不同大小 batch 求均值后再无权平均。

ConvLoRA 层同样需要收集相对于有效卷积权重的梯度。使用与原 Conv2d 参数完全一致的 padding 与 unfold/im2col 语义，将输入 patch 展平到 `(din/groups)·kh·kw`，按 logical group 累积 float32 `Gflat`，并在使用后立即释放缓存。必须将 hook 得到的卷积权重梯度与临时可训练 `nn.Conv2d.weight.grad` 比较，覆盖 stride、字符串/数值 padding、非零 padding_mode、dilation 和 groups。

冲突检测梯度统一定义为：`Gcur_cal` 来自匹配 context 的 external calibration，`Ghist` 来自 balanced historical replay，`Ganchor` 来自 immutable anchor；三者使用相同 JEPA loss 定义、model mode 和 reduction。SDC 在线第一遍得到的梯度另记为 `Gonline`，不得与 `Gcur_cal` 混用。

每次几何比较必须记录明确的 evaluation state。Triggered SLICE 默认在当前 replan 更新后的冻结 `Pfast` clone 上计算 `Gcur_cal`；`Ghist/Ganchor` 也在同一 slow、exception bank 和当前 episodic factors 的 clone 上计算，但每个 historical/anchor context 使用生产 router 决定其 persistent exception，且所有 persistent/episodic factors 都只作为求有效权重梯度的冻结函数，不执行 optimizer step。若实现选择其他状态，只能作为显式消融，并必须保证三个梯度仍来自同一套参数快照与 reduction。梯度采集本身不得更新 BatchNorm、buffer、router usage、RNG 或任何持久状态。

---

# 8. 梯度几何

## 8.1 梯度点积和余弦

逐层计算：

ρₗhist
= ⟨Gₗcur, Gₗhist⟩F
÷
(‖Gₗcur‖F ‖Gₗhist‖F + ε)

ρₗanchor 同理。

同时计算全局加权余弦。

权重可按：

* 梯度范数；
* 参数量；
* 模块组；

配置。

默认以每个 Linear 或 Conv group logical layer 为单位求解投影；全局加权余弦只负责 trigger 和日志，不把不同形状梯度拼成一个无法映射回模块的矩阵。某 logical layer 的 history/anchor 梯度近零时，对应约束在该层标为 inactive。

维护指数滑动平均：

ρ̄t = βρ̄t−1 + (1 − β)ρt

只有连续多次低于阈值才认为冲突。

每个 episode、每个 logical layer 独立维护 EMA 和 consecutive counter；第一次有效 cosine 令 `ρ̄₁=ρ₁`，episode reset 清空。零范数 reference 不产生 cosine 0，而是标记 unavailable 并停用该约束/计数；不得把“没有历史梯度”误判为冲突。

## 8.2 c-PCGrad 对照

实现忠实的单约束版本：

G̃cur
= Gcur
− c · min(⟨Gcur,Gprev⟩F,0)
÷ (‖Gprev‖²F + ε) · Gprev

用于消融和 repair。

`Gprev` 必须由调用方显式指定为 `Ghist` 或 `Ganchor`；同时处理两者时按配置顺序执行只用于消融，默认仍使用双约束求解器。`c` 使用 `c_pcgrad_coefficient`，默认 1.0。

## 8.3 双约束最小修改

默认方法：

最小化：

½ ‖G̃cur − Gcur‖²F

约束：

⟨G̃cur,Ghist⟩F ≥ −ξhist

⟨G̃cur,Ganchor⟩F ≥ −ξanchor

不要依赖重量级外部 QP solver。

实现一个双约束 active-set 求解器，枚举：

1. 无约束激活；
2. 只有 history 激活；
3. 只有 anchor 激活；
4. 两个约束同时激活。

选择满足全部约束且与 Gcur 距离最小的解。

要求：

* 处理 Ghist 或 Ganchor 近零；
* 处理二者近共线；
* Gram 系统仅在求解时加 ε数值正则，并在原始未正则约束上复查可行性；正则解不满足约束时使用 float64 小系统/pseudoinverse fallback 或报告不可行，不能静默接受；
* 输出是否可行、激活约束、修正幅度；
* 单元测试验证 KKT 条件和约束满足情况。

## 8.4 不进行默认每步硬正交

PCGrad 和双约束投影只用于：

* Triggered SLICE；
* repair；
* 可选的极端冲突模式。

普通 episode 每一步不得无条件执行硬梯度正交。

---

# 9. Pilot + Triggered Delayed SLICE

## 9.1 Pilot 阶段

episode 开始立即使用标准 Pilot LoRA，不等待 SLICE。

保持原 AdaJEPA 更新速度。

## 9.2 触发条件

至少满足：

* 𝓑episode 中有最少 transition；
* 已从匹配当前 context 的外部 calibration 获得稳定 Gcur；
* ρ̄hist 或 ρ̄anchor 连续多次低于阈值。

无稳定冲突时：

* 不运行 SLICE；
* 整个 episode 继续使用 Pilot LoRA。

## 9.3 SLICE 梯度

触发时：

1. 计算 Gcur、Ghist、Ganchor；
2. 用配置的方法进行梯度修正；
3. 得到 G̃cur；
4. 对每个目标层做低秩 SVD；
5. 初始化 Centered LoRA；
6. 冻结当前 Pilot task vector；
7. 保持函数连续。

Trigger 检查发生在一次原定 online update 完成后。若触发，则冻结该时刻的 Pilot、激活 Centered、丢弃旧 optimizer，并在下一次原定 online optimizer step 前重建；当原配置每次只有一步时，不得在同一 replan 偷加一个 Centered step。若本 episode 已无下一次 replan，Centered 保持零函数，episode task vector 仍等于 frozen Pilot。

随机化 SVD 必须使用由 `(fd_psc.seed, episode_id, logical_layer_id, trigger_index)` 稳定派生的专用 generator，并实际应用 `randomized_svd_oversampling/power_iterations`。验证返回方向有限、正交且低秩残差合理；失败时可以用该层 float32/float64 `torch.linalg.svd` 重试或保留 Pilot，不能用未验证方向继续。遍历 logical layers 必须按 manifest 中稳定 module/group id 排序，不能依赖哈希表顺序。

## 9.4 `slice_exact`

实现论文式非对称初始化：

* B⁰取前 r 个左奇异方向；
* A⁰取第 r+1 到第 2r 个右奇异方向。

按 Python 零基切片先构造未缩放 factors `Bhat=U[:, 0:r]`、`Ahat=Vh[r:2*r, :]`，再由 magnitude matching 得到非负 β，并使用 `B⁰=sqrt(β)Bhat`、`A⁰=sqrt(β)Ahat`。不得只给方向而省略 factor 幅度约定。

令 `kavailable` 为超过明确 SVD 数值阈值的有限奇异方向数。若 `min(dout,din) < 2*slice.rank`，按原规则直接退化到 `slice_symmetric`。否则 `slice_exact` 使用

```text
reff = min(slice.rank, floor(kavailable/2))
```

并严格取 `Bhat=U[:,0:reff]`、`Ahat=Vh[reff:2*reff,:]`，两侧 inner rank 必须相同。若请求 rank 因矩阵维度或数值 rank 不足而不能保持且 `reff>0`，记录 actual rank 后使用该 `reff`；若 `reff=0`，再尝试 `slice_symmetric`。`slice_symmetric` 的 `reff=min(slice.rank,min(dout,din),kavailable)`。梯度近零、SVD 非有限、symmetric 也无可用方向或没有下降方向时，直接保留 Pilot，不创建伪 Centered 分支。

## 9.5 `slice_symmetric`

实现：

G̃ ≈ UᵣΣᵣVᵣᵀ

B⁰ = UᵣΣᵣ¹ᐟ²

A⁰ = Σᵣ¹ᐟ²Vᵣᵀ

若 magnitude matching 给出 β，同样将 `sqrt(β)` 对称分配到 B⁰和 A⁰。

注意优化器的梯度下降符号，初始化方向要与降低 loss 的方向一致。

## 9.6 Magnitude matching

不能仅比较初始输出，因为 Centered LoRA 初始函数始终为零增量，但非零 A⁰、B⁰仍会改变梯度尺度。

实现默认：

`first_step_match`

步骤：

1. 在同一外部固定 calibration batch 上估计与当前配置相同 `episodic_lora.rank`、经 logical-layer 维度裁剪后的标准 LoRA/ConvLoRA 第一次有效 ΔW 更新范数；
2. 调整 SLICE 初始化缩放 β；
3. 使 SLICE 分支第一次有效 ΔW 范数与标准 LoRA 基线相差不超过 5%；
4. 对 β设置 `maximum_scale` 上限并保证有限、非负；
5. 若匹配失败，退化为标准 Pilot LoRA。

matching 默认逐层、逐 group（ConvLoRA）执行，并使用该层真实 optimizer 类型、学习率、betas、epsilon、weight decay 和零初始 optimizer state。除范数外，还必须验证第一次有效更新与 `−G̃cur` 的 Frobenius cosine 为正；方向不下降时视为匹配失败。为公平比较，baseline 与 SLICE 分支使用同一 calibration batch 和相同 dropout/RNG 状态。

同时记录：

* β；
* baseline first-step norm；
* SLICE first-step norm；
* 相对误差。

---

# 10. SDC 式谱漂移控制

## 10.1 基础权重 SVD

初始化时对每个 LoRA 目标层计算：

Wₗ⁰ = Uₗ⁰ Σₗ⁰ Vₗ⁰ᵀ

根据累计谱能量选择主要子空间：

Uₗp、Vₗp

默认解释 90% 基础权重能量。

Linear 按原 weight 矩阵计算；grouped Conv2d 按每个 flattened-kernel logical group 独立计算。checkpoint 只缓存达到能量阈值的 principal U/V、对应奇异值、实际 rank 和基础 weight hash，不保存无必要的完整方阵。

## 10.2 漂移指标

对当前 episodic 有效增量：

Dₗ(t)
= ‖Uₗpᵀ ΔWₗᵉ(t) Vₗp‖²F
÷ (‖ΔWₗᵉ(t)‖²F + ε)

每 K 个 replanning point 计算一次。

## 10.3 触发条件

只有同时出现以下信号时才开启 SDC：

* Dₗ连续 `drift_consecutive_checks` 次升高超过 `drift_increase_tolerance`，或超过 `drift_threshold`；
* anchor loss 恶化，或 ρanchor 为负。

不要一直开启。

SDC 按 logical layer 维护状态：每 `check_every_replans` 检查一次，条件成立时只对该层接下来的 online updates 激活，下一次检查条件不再成立即关闭并令 γ=1。episode reset 清空 SDC counters/active flags；不得把某层触发扩散到全部层。

## 10.4 软梯度修正

基础主子空间中的梯度：

Gp
= Uₗp Uₗpᵀ Gₗ Vₗp Vₗpᵀ

目标有效梯度：

G′
= Gₗ − (1 − γₗ)Gp

其中：

γₗ ∈ [γmin,1]

令 `Ep=‖Gp‖²F`、`Er=‖Gₗ−Gp‖²F`，默认：

```text
γraw = sqrt(Er / (Ep + Er + ε))
γₗ = clip(γraw, γmin, 1)
```

当 `Ep` 近零时 γ=1；当总梯度近零时跳过修正。该公式使 principal energy 占比越高，抑制越强，同时保留 `γmin` 下限。其他 γ策略只能作为显式消融配置。

## 10.5 将有效梯度修正传给 LoRA factors

不要分别随意投影 A、B。

使用代理标量：

ℒcorr
= ⟨stopgrad(G′ − G), ΔWtrainable(A,B)⟩F

总 loss：

ℒtotal = ℒJEPA + ℒcorr

这样相对于有效 ΔW 的梯度变为 G′。

代理项也不得物化 `ΔWtrainable`。对 canonical 分支 `BA`，用 `⟨Gdiff,BA⟩F = sum(B ⊙ (Gdiff Aᵀ))` 计算；Centered 分支分别计算当前项和冻结初始减项，只有 trainable factors 接收梯度。ConvLoRA 先按 logical group 使用 flattened-kernel G，并采用同一恒等式。

该修正必须使用明确的两阶段计算，不能在一次 backward 结束后假装修改已经完成的更新：

1. 保存第一遍之前的全局 RNG，第一遍在当前 online support batch 上 forward/backward，只收集 `Gonline`，不执行 optimizer step；
2. 计算事件触发状态、Gp、γ和 G′；
3. 清理第一遍 autograd graph 和普通 parameter grads；
4. 恢复第一遍之前的 RNG，在同一 batch 和同一 stochastic masks 下重新 forward，计算 `ℒJEPA + ℒcorr`；第二遍结束后的 RNG 必须等于正常单次 forward 后的状态；
5. 第二遍 backward 后执行一次原计划中的 optimizer step。

若 SDC 未触发，不得无条件执行第二遍计算。

要求：

* 对 Pilot 与 Centered 分支均正确；
* frozen Pilot 不接收梯度；
* 只作用于 trainable episodic branch；
* 测试代理修正后 factor gradient 与目标有效梯度的一致性。

---

# 11. Episode 结束 task vector

episode 结束后逐层导出：

ΔWₗepisode = Bₗδ Aₗδ

情况包括：

* 只有 Pilot；
* Pilot + Centered；
* 未触发 SLICE；
* 已触发 SLICE。

不得默认 rank 为 8。

记录：

* 低秩 factor rank；
* 实际数值 rank；
* 奇异值；
* 每层增量范数；
* 基础子空间漂移；
* 当前/历史/anchor 梯度余弦。

---

# 12. Spectral Surgery 候选

Spectral Surgery 只生成候选，不直接覆盖 task vector。

## 12.1 作用层

默认只对：

* attention output；
* MLP output/down；
* predictor final projection；
* sensory encoder projection head 最后一层。

## 12.2 分解

对 episodic task vector：

ΔW = U diag(σ) Vᵀ

U、σ、V 必须从 canonical low-rank factors 通过薄 QR + 小矩阵 SVD 得到；Linear 和 flattened-kernel ConvLoRA 均不得为此构造完整 `dout × din` 稠密 ΔW。

分解完成后固定这一组 U/V，只优化从全 1 开始的 scale 向量 a；每一步不得重新 SVD 后把符号翻转或退化子空间旋转误判成方向变化。重复奇异值的测试比较固定 basis 下的重构和子空间 projector，不直接比较另一次 SVD 返回的向量符号。

对 calibration 目标：

Jcal
= wcurrent ℒexternal-calibration + whist ℒhist + wanchor ℒanchor

奇异系数 aⱼ的梯度：

∂Jcal ÷ ∂aⱼ
= σⱼ uⱼᵀ Gcal vⱼ

## 12.3 更新

执行 1～3 次标量 projected-gradient step：

a ← clip(a − η∇a, amin, amax)

默认：

* amin = 0.75；
* amax = 1.25。

更新后保持奇异值向量的 L2 范数：

‖a ⊙ σ‖₂ = ‖σ‖₂

box 约束与 L2 约束必须联合满足。不得先 clip、再用一次无约束缩放而把 a 推回 `[amin,amax]` 之外。实现对 box 与加权 L2 球面交集的数值投影；若在容差内求解失败，记录原因并保留 original candidate。

生成：

* original candidate；
* spectral candidate。

若 calibration 梯度过小、数据不足、投影不可行或校准后 external-calibration loss 变差，则跳过。Spectral candidate 的接受、原始/谱候选选择和 scalar step 早停只能查看 calibration/hist/anchor，不能查看 commit-query。

## 12.4 不进行全层无条件谱编辑

所有层谱编辑必须作为消融配置，不能作为默认值。

---

# 13. 历史激活子空间与 soft-NESS

注意矩阵方向，禁止实现错误。

## 13.1 输入激活矩阵

对第 l 个线性层，将历史输入展平为：

Hₗ ∈ ℝᴺˣᵈⁱⁿ

SVD：

Hₗ = Uₗ Σₗ Vₗᵀ

输入空间方向是：

令 `Uₗ, σₗ, Vhₗ = svd(Hₗ)`，则 `Qₗ = Vhₗ.T[:, 0:q]`。

不是样本空间的 Uₗ。

必须写测试验证维度。

对 ConvLoRA，将卷积输入按原 kernel、stride、padding、dilation、padding_mode 和 groups 做 unfold；每个 group 的局部卷积 patch 构成该 logical layer 的 `Hₗ`，输入维度为 `(din/groups)·kh·kw`。soft-NESS、functional error 和 factor-space merge 均在该 flattened-kernel logical-layer 输入空间中计算。

## 13.2 激活二阶信息

对应的未中心化二阶矩阵：

Cₗ = HₗᵀHₗ ÷ N

Qₗ是 Cₗ的主要特征方向。

保存：

* Qₗ；
* λₗ = σₗ² ÷ N。

## 13.3 Soft-NESS 权重

每个历史方向的保护强度：

pₗi
= λₗi ÷ (λₗi + τₗ)

其中 λ和 τ均为激活能量，量纲一致。默认 `τ=max(median(positive λ), minimum_energy)`；quantile 和 fixed 模式分别使用配置分位数和严格为正的固定能量。τ按配置由：

* 中位能量；
* 分位数；
* 固定值；

确定。

构造：

Phist
= Q diag(p) Qᵀ

含义：

* 大 λ：强历史方向，p 接近 1；
* 小 λ：近似零空间，p 接近 0；
* 中间方向：软保护。

## 13.4 不构造完整 Phist

对于 task factors：

ΔW = BA

需要计算：

A R

其中：

R
= αsafe I + (αshared − αsafe)Phist

高效计算：

A R
= αsafe A + (αshared − αsafe)(AQ) diag(p) Qᵀ

不得显式构造 dᵢₙ × dᵢₙ 的 I 或 P。

历史为空或没有超过 `minimum_energy` 的方向时，令 Q为空、P=0，整个 episodic delta 进入 safe 分量并记录 `empty_history_safe_fallback`；不得构造伪方向。q 根据 `spectral_energy_threshold` 与 `maximum_rank` 共同选择。

---

# 14. Shared / Safe 合并候选

定义：

ΔWshared = ΔW Phist

ΔWsafe = ΔW(I − Phist)

候选：

ΔWmerge
= αshared ΔWshared + αsafe ΔWsafe

通常：

αshared ≤ αsafe

但不能硬编码该不等式，允许消融。

## 14.1 候选集合裁剪

根据以下信号缩小搜索空间：

* gradient similarity；
* frozen context similarity；
* prediction residual pattern similarity。

明显冲突时优先：

αshared ∈ {0,0.1,0.25}

明显一致时允许：

αshared ∈ {0.25,0.5,0.75,1}

αsafe 默认：

{0.5,0.75,1}

这些信号只裁剪候选，不直接决定提交。

只有显式配置了相应阈值时才允许按该信号裁剪；阈值为 null、信号缺失或不同信号互相矛盾时保留完整 coefficient grid。所有候选只在 calibration/hist/anchor/plasticity-probe 上评价，然后按确定性 lexicographic policy 预选唯一 final proposal：

1. 先排除 calibration loss、history/anchor tolerance、functional error、drift 或 plasticity probe 不可行的候选；
2. 最大化 external-calibration gain retention；
3. 最小化 worst-context 与 anchor regression；
4. 最小化 functional error，再选择更小 slow rank；
5. 最后按固定 candidate type、αshared、αsafe 顺序打破平局。

commit-query 在唯一 proposal 选定前绝不能加载或计算。

`merge.soft_ness_enabled=false` 只用于消融：跳过 projector 和 shared/safe coefficient grid，直接令处理后的 episodic candidate 等于完整 task vector（系数 1），再进入相同的 factor-space merge、rank 选择、screening 和 gates。不得用 `P=0` 后仍搜索 `αsafe` 的方式悄悄改变消融含义。

---

# 15. 不展开稠密矩阵的精确低秩合并

当前慢 LoRA：

ΔWˢ = BˢAˢ

当前处理后的 episodic task vector：

ΔWmerge = Bδ(AδR)

拼接：

B̄ = [Bˢ, Bδ]

Ā = [Aˢ; AδR]

薄 QR：

B̄ = QB RB

Āᵀ = QA RA

小矩阵：

C = RB RAᵀ

SVD：

C = Uc Σc Vcᵀ

截断到 slow rank r：

Bˢnew = QB Uc,r Σc,r¹ᐟ²

Aˢnew = Σc,r¹ᐟ² Vc,rᵀ QAᵀ

要求：

* 不构造完整 dout × din 候选矩阵；
* 支持 float32 数值计算；
* 最终可转换回模型 dtype；
* 对 rank-deficient 情况稳定；
* QR/SVD 失败时提供安全回退；
* 写测试验证低秩实现与显式稠密实现等价。

“安全回退”只能是：提高数值精度后重试、使用稳定的 pivoted/pseudoinverse 小矩阵路径，或拒绝该候选；不得回退到构造大型稠密候选。grouped Conv2d 对每个 logical group 独立执行合并和 rank 选择。

---

# 16. Slow rank 选择

默认候选 rank 来自 `slow_lora.allowed_ranks`：

{8,16,24,32}

不得在代码中硬编码该集合。令 `dmax=min(dout,din,slow_lora.maximum_rank)`，每个 logical layer 实际候选集合为 `sorted(unique(min(r,dmax) for r in slow_lora.allowed_ranks))`；因此小层或 Conv group 维度小于 8 时仍有 `dmax` 候选。删除非正 rank；若矩阵数值 rank 为 0，canonical zero adapter 单独允许 rank 0。`initial_rank` 必须属于裁剪后的候选集合，或按同一规则确定性裁剪并记录。

选择最小 r，同时满足：

## 16.1 谱能量

Σᵢ₌₁ʳ σᵢ²
÷ Σᵢ σᵢ²
≥ τenergy

默认 τenergy = 0.99。

## 16.2 功能误差

在 calibration 激活 H 上：

εfunctional
= ‖(M − M̂)Hᵀ‖²F
÷ (‖MHᵀ‖²F + ε)

注意 H 的形状为 N × din，因此权重作用时使用 Hᵀ。

默认：

εfunctional ≤ 0.02

M 是未截断 canonical merged candidate，M̂是该 rank 的压缩结果；H 来自 external calibration，并在候选持久状态、episodic/exception 路由按该候选定义的 eval mode 下采集。若 `‖MHᵀ‖²F` 近零，同时报告绝对输出误差；只有绝对误差也在数值容差内时才视为通过，不能用 ε分母把任意误差掩盖掉。

`MHᵀ` 与 `M̂Hᵀ` 必须通过 factors 顺序乘法计算，不得为了 functional error 物化 M/M̂。对 Conv logical group，H 是对应 unfold patch matrix。

## 16.3 Rank 饱和处理

若 rank 32 仍无法通过：

1. 不静默截断；
2. 尝试 repair；
3. repair 后重新压缩；
4. 仍失败时建立 exception adapter 或拒绝 episode。

---

# 17. Replay memory

## 17.1 V1 存储

默认存储 frozen backbone latent：

* visual latent dtype 使用 `replay.visual_latent_dtype`，辅助张量使用 `replay.auxiliary_dtype`；
* historical 轨迹窗口上限使用 `replay.historical_windows`；
* anchor 数量使用独立 `anchor_data.windows`，不占 historical reservoir；
* 保留连续时间结构。

## 17.2 更新策略

实现 context-cluster balanced reservoir：

* 按 frozen context embedding 聚类；
* 控制每个 cluster 容量；
* 新环境不能挤掉所有旧环境；
* anchor 永远不参与淘汰。

使用 frozen context embedding 的 cosine similarity 做确定性 online clustering：最近 prototype 低于 `new_cluster_similarity_threshold` 时创建新 cluster；达到 `maximum_context_clusters` 后不再创建并分配到最近 cluster。每个 cluster 使用以 `fd_psc.seed` 驱动的标准 reservoir admission，动态容量分配必须保证已有 cluster 的 `minimum_windows_per_cluster`（总容量不足时按稳定 cluster id 公平退化并记录）。prototype、seen count、reservoir RNG 和 admission count 都属于 checkpoint/transaction 状态。

只有 slow candidate 最终 commit-query 通过并原子提交后，当前 episode support 才能进入 global historical replay，并在“新 slow 启用、episodic 和 exception 禁用”的持久状态下重新计算用于 Q 更新的层输入激活。slow 提交失败、普通拒绝或新建/更新 exception 时不得更新 global replay/Q；exception 只保存有界的 exception-local replay。

## 17.3 Repair 采样

V1 默认 balanced uniform。

实现可选 GRASP：

* 前 30% easy/prototype；
* 中间 40% balanced uniform；
* 后 30% hard/high-residual。

无类别环境中：

* prototype = frozen context cluster center；
* easy = 距中心近且 residual 稳定；
* hard = 距中心远、接触切换、动力学突变或 residual 高。

30/40/30 指 repair optimizer steps 或 batch schedule 的累计比例，取整使用 largest-remainder 并保证总步数精确等于配置值；样本不足时有放回采样并记录。`checkpoint_schedule=cumulative` 表示 5/10/20 是同一 optimizer trajectory 的累计检查点，screening 在克隆 candidate 上压缩，不修改继续训练的 repair state。

---

# 18. Sleep 路径

Sleep 分成“calibration proposal 阶段”和“single final commit-query 阶段”。proposal 阶段可以生成、screen、repair 多个候选，但只使用 support、external calibration、historical replay、anchor 和独立 plasticity probes。完成后必须冻结并选出至多一个 final proposal；随后第 19 节 gates 只对它执行一次。commit-query 失败是本 episode 的终态，禁止查看失败结果后再 repair、换 α、换 spectral candidate 或创建 exception。

实现显式状态机并验证非法转移：

```text
IDLE
  -> EPISODE_PILOT
  -> EPISODE_CENTERED (optional, triggered once)
  -> SLEEP_CALIBRATION
  -> REPAIR (optional)
  -> FINAL_PROPOSAL_READY | REJECT_NO_PROPOSAL
  -> FINAL_GATE
  -> COMMIT_SLOW | COMMIT_EXCEPTION | REJECT_QUERY
  -> IDLE
```

任意异常进入 `ROLLBACK -> IDLE`。每个 episode 最多一次 Pilot→Centered、最多一个 final proposal、最多一次 FINAL_GATE 和最多一次 persistent commit。

若 episode 没有完成任何原定 online optimizer step、support 不足以构造原 JEPA loss、或导出的 episodic task vector 在数值容差内为零，则从 `SLEEP_CALIBRATION` 直接进入带原因的 `REJECT_NO_PROPOSAL`：不得读取 commit-query、不得创建 exception，也不得更新 replay/Q；只允许更新非学习性的 episode/log counters。该路径不是错误，但必须测试。

episode 状态定义：

```text
Pbefore = θ₀ + Δslow + Δrouted-exception
Pfast   = Pbefore + Δepisodic
```

`before`、`fast`、`candidate` 在所有 screening/gate 中必须使用这一致定义。episode 开始用 frozen context 只路由一次 exception，并固定到 episode 结束。

proposal 类型决定 candidate 的精确定义：global-slow proposal 用“候选 slow + 提交前 exception bank + 无 episodic”，new-exception proposal 用“原 slow + 含 proposed exception 的候选 bank + 无 episodic”，existing-exception replacement 用“原 slow + 仅替换同一 adapter id 后的候选 bank + 无 episodic”。对每个 evaluation context 都使用克隆的生产 router；当前 context 的 proposal route 按第 19 节的专门规则固定。禁止同时修改 slow 和 exception，或把 episodic 留在 candidate 中重复计算。

proposal 阶段的“fast gain”统一记为 `Gainfast,cal = Lbefore,cal − Lfast,cal`，只从 external calibration 得到；它用于决定是否值得 repair/exception，不能提前读取 commit-query。第 19 节的 `Gainfast,commit` 是独立的最终 gate 指标，二者不得混名或互相缓存代用。

若没有路由到已有 exception，按路径 A→B→C 生成 global slow 或新 exception proposal。若路由到已有 exception 且 `routed_episode_update=replace_exception`，本 episode 不尝试写 global slow：把 `exception_before + episodic` 在 factor space 合并、压缩为更新后的同一 exception proposal；screening 同时使用当前 context calibration、该 adapter 的 local replay、anchor 和 plasticity probes。快速压缩失败时可以对该 exception candidate 做同样有界的 local repair，但不得创建第二个 exception。slow 保持不变，所有路径失败时保留旧 exception。这样避免把相对于 exception 学到的增量错误地再次写入 global slow。

## 18.1 路径 A：快速无训练提交

顺序：

1. 导出 episodic task vector；
2. 生成 original candidate；
3. 可选生成 spectral candidate；
4. 构造 soft-NESS projector；
5. 搜索 αshared、αsafe；
6. 低秩 QR + 小矩阵 SVD；
7. 选择 slow rank；
8. 使用 calibration/hist/anchor/plasticity probes 做 screening；
9. 按确定性 policy 选出唯一 slow final proposal，或进入 repair。

除 calibration gradient 与 scalar spectral scale 外，不进行参数训练；不得读取 commit-query。

## 18.2 路径 B：有限 repair

当：

* `Gainfast,cal` 至少高于 numerical tolerance；
* 所有快速候选未通过 calibration screening；

启动 repair。

只训练候选 slow LoRA，θ₀继续冻结。

损失：

ℒrepair
= wcurrent ℒJEPA(current support) + wreplay ℒJEPA(replay) + βprox ℒprox

其中：

ℒprox
= Σₗ ‖hₗcandidate(xold) − hₗbefore(xold)‖²

global slow repair 的 `xold` 来自 balanced historical replay；routed exception local repair 的 `xold` 来自该 exception 的 local replay。`hbefore` 在 repair 开始前以 Pbefore、eval mode 计算并 detach，只缓存配置 tag 匹配的有限层；若一个受支持模型没有任何匹配层，配置验证必须报错或显式关闭 LPR，不能悄悄得到零 proximal loss。

global slow repair 的 `replay` 是 balanced historical replay；routed exception repair 的 `replay` 是该 adapter 的 bounded local replay。三项权重分别使用 `repair.current_weight`、`repair.replay_weight` 和 `repair.proximal_weight`；`proximal_enabled=false` 时完全省略最后一项及其 activation cache。historical/local replay 为空时 replay 项标记为 `not_applicable` 并重新归一化已启用的正权重，不能用空 batch 产生 NaN。

只对少量关键层保存 before activations：

* sensory encoder projection output；
* predictor 中间关键层；
* final predicted latent。

repair 步数依次尝试：

5、10、20

必要时对 repair gradient 使用双约束投影或 c-PCGrad。

每个累计步数检查点后在克隆 state 上重新压缩和执行 calibration screening，成功即停止并冻结为唯一 final proposal。repair 的梯度约束必须投影有效权重梯度，再通过与 SDC 相同的代理 loss 传给 factors；不得直接投影 A/B。

## 18.3 路径 C：Exception adapter

若：

* `Gainfast,cal ≥ exception.minimum_calibration_fast_gain` 且大于 numerical tolerance；
* quick merge 未通过 calibration screening；
* repair 未通过 calibration screening；

则：

* 不更新 Δˢ；
* 将 episodic task vector 压缩为独立 adapter；
* 保存 frozen context prototype；
* 保存 residual prototype；
* 加入有限 exception 集合。

新建或更新 exception 也只是一个 proposal，必须先通过 calibration/anchor/plasticity screening，再作为唯一 final proposal 进入 commit-query；不能因为它不修改 global slow 就绕过 gates。

exception 也使用第 15–16 节相同的 factor-only 压缩、谱能量和 functional-error 判据，但每个 logical layer 的候选 rank 由 `slow_lora.allowed_ranks` 同时裁剪到 `exception.maximum_rank` 和矩阵维度；其 rank 独立保存，不能占用或改写 global slow rank。没有可行 rank 时拒绝该 exception proposal。`slow_lora.functional_error_threshold` 是所有 persistent adapter proposal 共用的压缩阈值，名称沿用配置分组不表示 exception 可以绕过 Gate 5。

默认路由：

* 最近 prototype 余弦相似度；
* 无需训练 router。

只有最近 prototype 相似度达到 `minimum_route_similarity` 才能激活 exception；否则按 `no_match_behavior=slow_only` 只使用全局 slow LoRA。不得因为 exception 集合非空而强制为每个新 episode 路由一个 adapter。路由决定默认在 episode 第一次在线更新前，基于当时已经可见且不含未来信息的初始 observation/history window 计算 frozen context，并固定到 episode 结束；若初始窗口不足则 no-match，不能等待 commit-query 或未来执行结果再回填 route。除非配置明确启用并测试 replan 级路由，否则不得中途改变。

context/prototype 近零、NaN、Inf 或维度/schema 不匹配时必须视为 no-match 并记录原因，不能产生任意 cosine route。

prototype 更新必须确定：新 exception 的 context/residual prototype 是本 episode accepted support descriptors 的 float32 均值后 L2 normalize；更新已有 exception 时用已提交 descriptor count 做 count-weighted running mean 后再 normalize，并原子更新 count。近零 residual 保留显式 zero/availability 标记，不做无定义归一化。prototype 只在 exception commit 成功后更新；screening、query 失败和 rollback 不得改变 prototype/count。默认 router 只用 frozen context prototype，residual prototype 仅用于候选裁剪/诊断，不能悄悄改变 routing metric。

可选实现 torch 版对角协方差 GMM，但不得强依赖 sklearn。

如果 exception 已达上限：

* 淘汰长期未使用且收益最低者；
* 或拒绝新 adapter；
* 不无限增长。

默认 eviction policy 先选最久未使用者，再以累计验证 gain 较低者、稳定 adapter id 打破平局；所有 usage/gain 统计只在真实 production route 或成功 commit 后更新，screening/canary 不计使用。

淘汰/替换必须是同一原子事务的一部分，不能淘汰当前 episode 正在使用的 adapter。exception proposal 成功后只更新 exception bank、prototype、usage stats 和 bounded local replay；不更新 global slow、global historical replay 或 Q/λ。更新已有 exception 时保持 adapter id，原子替换其 factors/prototype/local replay。local replay 按 stable window id 去重，并用以 `(seed, adapter_id)` 派生且可 checkpoint 的确定性 reservoir 保持 `local_replay_windows` 上限；失败 proposal 不得改变 seen count、RNG 或内容。

---

# 19. Commit gates

唯一 final proposal 必须使用外部固定 commit-query 做最终 gate。该数据与 support、calibration、plasticity probes、anchor、repair 和 report-test 不相交，并通过 manifest 匹配当前 context。一次 gate evaluation 可以在同一 commit-query 上前向计算 before、fast、candidate 三个冻结 state，但不能比较多个 candidate proposal。commit-query gate 失败后直接拒绝并回滚，不得再进入第 18 节。

所有 gate evaluation 必须在克隆/functional candidate state 上运行，不得修改 live slow、exception、optimizer、buffer、Q/λ、router stats 或 RNG。history/anchor 集合中的每个 context 使用生产环境相同的 frozen-context router；无匹配时只启用 slow。

完整 FD-PSC 运行中 Gates 1–6 是合取关系，任何一个启用 gate 失败或返回非有限指标都拒绝 proposal；只有 `gates.allow_unsafe_ablation=true` 的显式消融可以关闭其中某项。Gate 7 由 canary schedule 控制。每个 gate 必须返回 `pass/fail/not_applicable` 和理由，不能把异常或缺数据静默当作 pass；本规格明确允许的 `not_applicable` 仅限空的先验 historical replay 等列出的冷启动情况。

对新 exception proposal，当前 context 的 candidate evaluation 使用 proposed prototype/adapter（它必须达到 route threshold），而 before 为原 slow-only；history/anchor evaluation 使用包含 proposed prototype 的克隆 router，借此检测它是否误路由到其他 context。对更新已有 exception，before/candidate 使用同一 adapter id 的旧/新 factors。

## Gate 1：当前 episode 收益

定义：

Gainfast,commit
= ℒbefore,commit-query − ℒfast,commit-query

Gaincandidate,commit
= ℒbefore,commit-query − ℒcandidate,commit-query

要求：

Gaincandidate,commit ≥ κnew Gainfast,commit

默认 κnew = 0.8。

若 `Gainfast,commit ≤ 0`，则普通情况下不应提交。

slow proposal 至少要求 `Gainfast,commit` 大于数值容差；新建/更新 exception 还要求 `Gainfast,commit ≥ exception.minimum_commit_fast_gain`。当 threshold 设为 0 时仍不能用纯浮点噪声创建 adapter。calibration 阶段的 `minimum_calibration_fast_gain` 不能替代这一最终条件。

## Gate 2：历史保持

要求：

ℒcandidate,hist − ℒbefore,hist ≤ εhist

同时记录不同 context cluster 的最坏退化，不能只看均值。

除均值约束外还必须满足：

```text
max_context(ℒcandidate,hist,c − ℒbefore,hist,c) ≤ εworst-context
```

当 `εhist=0` 时，比较仍必须使用配置的绝对/相对数值容差，以吸收确定性浮点舍入误差；该容差不得被解释为额外允许的业务退化。

首个成功 slow commit 之前若 historical replay 为空，Gate 2 返回带原因的 `not_applicable`，但 Gate 1、Anchor、plasticity、functional-error 和 drift 仍必须执行；一旦存在 prior committed historical window，采样失败或数据损坏必须 fail，不得退回 `not_applicable`。

## Gate 3：Anchor 保持

要求：

ℒcandidate,anchor − ℒbefore,anchor ≤ εanchor

当 `εanchor=0` 时同样应用数值比较容差。

## Gate 4：可塑性保持

从 before 和 candidate 分别创建全新的零 episodic Linear/ConvLoRA，使用当前 `episodic_lora.rank` 并按 logical-layer 维度裁剪；默认值才是 rank 8。

在 manifest 提供的独立 `plasticity_support` 上执行一次真实 FD-AdaJEPA update event，并只在独立 `plasticity_query` 上测量收益；两者都不能是 commit-query。这里的一个 update event 指一次与原 online 路径相同的 `finetune()` 调用：使用原配置的 optimizer 重建、内部 update 步数、LR 比例、loss 和 stop-gradient；不是擅自硬编码成一个 optimizer step。

定义：

Gplasticity(θ)
= ℒquery(θ)
− ℒquery(UFD-AdaJEPA(θ,support))

要求：

Gplasticity(candidate)
≥ κplasticity Gplasticity(before)

默认 κplasticity = 0.9。

这里必须使用真实下一 episode 配置：

* 配置指定的 episodic LoRA rank（默认 8）和相同的逐层实际 rank 裁剪；
* 相同学习率；
* 相同的一次 `finetune()` update event 及其原始内部步数；
* 相同 stop-gradient；
* 相同 buffer 逻辑。

不能用原 selected-layer 稠密更新代替。

该临时更新必须克隆完整持久 state、使用相同 RNG/optimizer 超参数且在评估后完全销毁。若 `Gplasticity(before) > numerical_epsilon`，使用上面的比例 gate；否则要求 candidate 的 gain 非负且相对 before 的下降不超过数值容差，避免用接近零分母做比例判断。

## Gate 5：功能压缩误差

要求：

εfunctional ≤ 配置阈值。

Gate 5 使用 `slow_lora.functional_error_threshold`，不得再维护第二个可能不一致的 gate threshold。

## Gate 6：基础谱漂移

将第 10 节的漂移函数推广到任意 canonical adapter delta：

```text
Dadapter(ΔW) = ‖Upᵀ ΔW Vp‖²F / (‖ΔW‖²F + ε)
```

对 global slow proposal，`Dbefore` 使用提交前 slow delta，`Dcandidate` 使用候选 slow delta。对 exception proposal，使用该 context 实际激活的 `slow + exception` persistent delta 做 before/candidate 比较。不得把 episodic-only D 与 persistent candidate D 混用。关键层要求：

Dcandidate ≤ Dbefore + εdrift

## Gate 7：真实规划 canary

提供接口在：

* 每 K 个 episode；
* 高风险提交；
* rank 扩张；
* exception 合并；

时运行少量 MPC canary rollouts。

Canary 使用独立 manifest、固定 seeds、克隆模型状态和独立/可复位环境 worker；rollout、planner adaptation 和 router usage 不得污染 live state。高风险提交的 canary 在原子 swap 前运行；周期 post-commit canary 失败时按 commit journal 整体回滚。环境不支持确定性 reset 时必须记录限制。

环境不可用时：

* 单元测试使用 mock evaluator；
* 集成文档标记真实 rollout 尚未执行；
* 不得伪造成功率。

---

# 20. 原子提交与回滚

Sleep 不允许直接原地覆盖慢 LoRA。

流程：

1. 保存完整 `persistent_before` transaction snapshot；
2. 在独立 candidate state 中构造第 18 节选出的唯一 slow 或 exception proposal；
3. 运行全部 gates 和需要的 pre-commit canary；
4. gates 通过后在临界区原子交换对应 persistent factors；
5. slow proposal 才更新 global replay、Q/λ、global prototypes；exception proposal 只更新 exception-local state；
6. 先写 `prepared` journal record，再按第 22 节的“已验证不可变版本 + atomic latest pointer”协议保存 checkpoint，最后原子标记 journal record 为 `committed`；恢复时忽略没有匹配 latest/version hash 的 prepared record；
7. 任一步骤、checkpoint 写入或周期 canary 失败时恢复完整 `persistent_before`。

原子事务不能只覆盖 slow LoRA。候选提交前必须建立可恢复事务快照，至少包含：slow factors/rank、historical replay 及 reservoir 元数据、Q/λ、context prototypes、exception factors/local replay/usage statistics、router state、episode counter、commit counter、config/schema identity 和 RNG state。任何提交后更新失败或 canary 失败都必须整体回滚。

若 canary 不是每次提交都运行，保留覆盖该 canary 周期内全部提交的 commit journal；不得只保留一个 `slow_previous` 后声称可以定位并撤销多次提交中的失败来源。

提交日志至少包含：

* episode id；
* candidate 类型；
* αshared；
* αsafe；
* slow rank；
* gate 指标；
* 是否 repair；
* 是否 spectral surgery；
* 是否 SLICE；
* 是否 SDC；
* 提交、拒绝或 exception；
* 回滚原因。

---

# 21. 激活子空间更新

只能在 candidate 成功提交后更新 Q 和 λ。

禁止在 gate 前加入当前 episode 激活。

更严格地说，只有 global slow proposal 成功提交后才更新 global Q/λ；exception proposal、拒绝或回滚均不更新。`Hnew` 来自本 episode 已接受 support，在新 slow 启用、episodic/exception 禁用、eval mode 下重新前向采集，不能复用 Pilot/Centered 训练时的旧激活。

## 21.1 增量 sketch

旧状态：

* Q ∈ ℝᵈⁱⁿˣq；
* λ ∈ ℝq。

构造行空间 sketch：

Z
= [
√βforget diag(√λ) Qᵀ
;
√(1−βforget) Hnew ÷ √N
]

其中：

Z ∈ ℝ⁽q+N⁾ˣᵈⁱⁿ

对 Z 做 SVD：

令 `Uz, σz, Vhz = svd(Z)`。

新的输入方向：

Qnew = Vhz.T[:, 0:qnew]

新的能量：

λnew = σz²

根据能量阈值和最大 rank 截断。

`βforget` 对应 `activation_subspace.forgetting_factor`，不得与 SLICE magnitude scale β混用。Conv groups 仍按 logical layer 分别更新。

再次强调：

* 输入空间方向来自右奇异向量 Vz；
* 不是左奇异向量 Uz。

---

# 22. Checkpoint 与状态恢复

保留官方 AdaJEPA checkpoint 为只读基础文件，不用 FD-PSC 覆盖或重写它。FD-PSC 使用独立、版本化 sidecar checkpoint。between-episode 保存/恢复是必需功能；mid-episode 中断恢复是可选功能。

FD-PSC checkpoint 必须包含：

* θ₀引用或基础 checkpoint 标识；
* 基础 checkpoint 内容 hash；
* 运行时目标模块 manifest 及其 hash；
* slow LoRA factors；
* 每层 slow rank；
* 截断的 U⁰、V⁰、奇异值、基础谱 rank 与每层 weight hash；
* Q、λ历史激活子空间；
* historical replay；
* anchor；
* context prototypes；
* exception adapters；
* exception-local replay、router threshold/prototypes；
* exception usage statistics；
* episode counter；
* Python random、NumPy、Torch CPU、所有 CUDA device 以及所有专用 sampler generator state；
* 配置；
* schema version。

加载时先恢复原 AdaJEPA checkpoint，再按 manifest 注入 adapter，最后加载 FD-PSC 状态。基础 checkpoint hash、模块路径、层类型、维度或 Conv2d 几何参数不匹配时必须拒绝加载，不得 best-effort 套用到其他模型。

sidecar 使用“不可变版本文件 + 原子 latest 指针”，不能把未验证的临时文件直接覆盖唯一 latest：

1. 在 `state_directory` 内写带 commit id/content hash 的临时版本文件，flush 并 fsync；
2. 从临时文件重新加载，验证 schema、内部 content hash、base hash、manifest hash 和关键 tensor shape；
3. 将已验证临时文件 `os.replace` 为同目录不可变版本文件，例如 `state-<commit>-<hash>.pt`；
4. 写入包含目标文件名、hash、schema 和 commit id 的临时 latest JSON，flush/fsync 后用 `os.replace` 原子更新 `latest_pointer_path`；
5. 平台支持时 fsync 父目录；旧版本文件至少保留到事务和对应 canary 周期完成，之后按明确 retention policy 清理。

任一步失败都不能让 latest 指向未验证或不存在的版本；checkpoint 写失败属于事务失败，live state 必须回滚，旧 latest 指针及其不可变版本仍可加载。若 latest 指针损坏，恢复工具可以扫描已验证版本并按 commit journal 找到最近完整提交，但不得静默选择 hash 不匹配文件。

相对 `state_directory/latest_pointer_path/resume_path` 必须按 Hydra runtime output directory 明确解析，并在日志中打印规范化绝对路径；不得依赖进程启动 cwd 猜测位置。`resume_path` 可以显式指向 latest JSON 或某个不可变版本文件，两种格式都必须验证。

episode 中断恢复可选保存：

* Pilot factors；
* Centered factors；
* B⁰、A⁰；
* frozen Pilot factors；
* optimizer；
* local buffer；
* conflict EMA。

mid-episode 恢复若启用，还必须保存当前 routed exception id、MPC iteration、SLICE/SDC trigger counters、planner-local action warm start 和可恢复环境标识；环境本身无法序列化时不得承诺真实 rollout 的精确 mid-episode resume，只提供 between-episode resume。

加载旧 FD-PSC schema 时提供显式、测试过的 migration；未知新版本拒绝加载。官方不含 FD-PSC 状态的旧 AdaJEPA checkpoint 不是 migration 错误，而是合法 base-only 初始化。

---

# 23. 代码模块建议

根据仓库实际结构放置，逻辑上至少拆成：

```text
fd_psc/
├── config.py
├── lora_layers.py
├── injector.py
├── encoder_adapters.py
├── trainer.py
├── state_machine.py
├── gradient_geometry.py
├── gradient_hooks.py
├── slice_initializer.py
├── spectral_control.py
├── activation_subspace.py
├── low_rank_merge.py
├── replay_memory.py
├── external_data.py
├── repair.py
├── commit_gates.py
├── exception_router.py
├── checkpoint.py
├── transaction.py
├── metrics.py
└── diagnostics.py
```

避免把全部逻辑塞进一个 trainer 文件。

---

# 24. 数值和性能要求

* SVD、QR、梯度几何默认使用 float32。
* 模型 forward 可继续使用原 dtype。
* 不创建长期 autograd graph。
* 历史梯度计算后立即释放缓存。
* 不显式构造大型单位矩阵。
* 不显式构造 soft projector 的完整矩阵。
* 不显式构造完整候选 ΔW，除非在测试或小层 debug 模式。
* 支持 CPU fallback。
* CUDA 可用时不能出现无意义的 CPU/GPU 频繁拷贝。
* 对 NaN、Inf、rank deficiency、空 replay、零梯度提供安全处理。
* 所有 fallback 必须记录原因，不能静默。
* 专用 RNG 不得复用 Python `hash()`；模块遍历、candidate tie-break、reservoir、SVD 和 routing 的顺序必须由稳定 IDs 决定。
* 可复现测试启用 PyTorch deterministic algorithms；核心 CPU/mock 测试遇到不支持的 kernel 时必须换用确定实现或 fail，不能静默 skip。只有明确依赖真实硬件/环境的 smoke test 才可因平台限制 skip 并记录。不能仅设置 seed 就声称 bitwise deterministic；真实性能实验必须记录 deterministic flags、CUDA/cuDNN 版本和已知非确定 kernel。

---

# 25. 必须实现的单元测试

至少包括：

## LoRA

1. 禁用 FD-PSC 时输出与原模型一致。
2. 零 Pilot LoRA 初始输出一致。
3. Centered LoRA 初始函数偏移为零。
4. Pilot → Centered 切换前后输出连续。
5. θ₀始终无梯度且不变化。
6. slow LoRA 在 episode 内无梯度。
7. episode reset 完全清空 episodic 状态。
8. state_dict 保存加载后输出一致。
8a. 原 finetune optimizer 重建周期和更新步数保持不变，SLICE 不偷加 step。
8b. slow + routed exception + episodic canonical composition 与实际 forward 一致。
8c. FD-PSC 关闭时不注入模块、不改变 state_dict keys，旧 checkpoint 可直接加载。
8d. Pilot rank 与 Centered rank 不同时分别使用各自 actual-rank scaling，canonical factors 仍精确重构实际 forward。
8e. runtime manifest 覆盖所有且仅覆盖已启用、active-in-forward 的目标 Linear/Conv logical layers；漏层、重复注入和 frozen-backbone 命中都会失败。
8f. θ₀全部参数与持久 buffers 在 online、sleep、reject、commit、exception 和 rollback 后逐 tensor bitwise 不变。

## ConvLoRA

* 禁用 ConvLoRA 时输出与原 Conv2d 一致；
* 零 Pilot ConvLoRA 初始输出一致；
* ConvLoRA Pilot → Centered 切换连续；
* stride、数值/字符串 padding、非零 padding_mode、dilation、groups 与基础层一致；
* flattened-kernel canonical factors 重构正确；
* factorized A→B ConvLoRA 前向与显式 delta-kernel 卷积等价，生产 forward 不调用增量权重物化 API；
* grouped convolution 的 groupwise delta 正确；
* 不同 Conv group 没有 cross-group 参数或子空间污染；
* frozen backbone 内没有被注入的 ConvLoRA；
* post-backbone projection ConvLoRA 确实位于 JEPA 主 forward graph。

## Rank

9. 普通 LoRA rank ≤ r。
10. Centered task vector rank ≤ 2r。
11. Pilot + Centered rank ≤ rpilot + 2rcentered。
12. 低秩 factor 拼接重构正确。

## 梯度

13. hook 权重梯度与真实 weight.grad 一致。
13a. Conv hook 梯度与真实 Conv2d.weight.grad 一致。
13b. 同一模块重复调用时 hook invocation 配对和梯度累积正确。
14. 梯度余弦正确处理零梯度。
15. c-PCGrad 只移除负冲突分量。
16. 双约束求解满足约束。
17. 双约束求解在无冲突时返回原梯度。
18. 双约束求解处理共线约束。
19. SDC 代理 loss 产生目标有效梯度。
19a. SDC 未触发时只有一次 forward；触发时两遍使用相同 RNG masks 且全局 RNG 只前进一次。

## SLICE

20. Trigger 未满足时不运行。
21. `slice_exact` 维度正确。
22. 维度不足时正确 fallback。
23. `slice_symmetric` 重构正确。
24. magnitude matching 误差小于配置阈值。
25. SLICE 触发后 optimizer 参数集合正确。
25a. SLICE 第一次有效更新与 `−G̃cur` cosine 为正，默认一步 update 数量不变。

## Spectral

26. Spectral Surgery 在固定分解 basis 中只改 scale/奇异值；重复奇异值时比较重构或 projector，不比较另一次 SVD 的向量符号。
27. scale 保持在范围内。
28. 配置要求时保持奇异值 L2 范数。
29. calibration 不改善时保留原候选。

## Activation subspace

30. H 为 N × din 时 Q 维度为 din × q。
31. soft-NESS p ∈ [0,1]。
32. 大能量方向保护更强。
33. 不显式构造完整 projector。
34. 增量 Q 更新方向正确。
34a. 空历史得到 Q为空、P=0；Conv groups 的 Q/λ相互独立。

## Merge

35. factor-space merge 与显式稠密 merge 一致。
36. QR + 小 SVD 截断结果正确。
37. 功能误差计算正确。
38. rank manager 选择最小可行 rank。
39. rank 上限失败时不静默提交。
39a. 小于最小 allowed rank 的小层仍得到合法裁剪 rank；零矩阵允许 rank 0。
39b. candidate selection 全程不读取 commit-query，并按确定性 tie-break 选出唯一 proposal。
39c. `soft_ness_enabled=false` 时直接合并完整 task vector，不再搜索 shared/safe 系数。

## Replay

40. support/calibration/commit-query/plasticity/report-test/anchor 无泄漏。
41. 连续轨迹窗口边界正确。
42. anchor 不被淘汰。
43. cluster-balanced reservoir 不丢失所有旧 cluster。
44. GRASP 顺序符合 easy → balanced → hard。
44a. 只有成功 slow commit 才更新 global replay/Q；exception/reject/rollback 不更新。
44b. cluster reservoir 保存并恢复 prototype、seen count、RNG 和 admission count。

## External calibration/commit-query

* manifest 能稳定映射 context；
* support、calibration、commit-query、plasticity probes、report-test、anchor 在 trajectory、transition、frame 和 checksum 层面均不重叠；
* commit-query 从未进入 online optimizer、repair、screening 或候选生成，且只评估唯一 final proposal；
* commit-query 失败后不得尝试第二 proposal；
* plasticity support/query 独立，report-test 从未影响任何算法状态；
* 缺失 context、checksum 不匹配和 schema 不匹配时 fail-fast；
* 新 support 在第一次 online update 前发现与任一外部 split/anchor 的 ID 或内容 hash 重叠时 fail-fast；
* baseline/ablation 使用完全相同的固定 split。

## Gates

45. 任一 gate 失败都不能提交。
46. 原子提交成功时状态正确。
47. 提交失败时 slow LoRA 不变化。
48. canary 失败可回滚。
49. plasticity gate 使用新的零 episodic LoRA。
49a. 完整事务回滚 slow、replay、Q/λ、exception/router、counters 和 RNG。
49b. checkpoint 原子写失败保留旧 latest 并恢复 live state。
49b-1. latest pointer 永远只指向已重新加载验证的不可变版本；临时/损坏版本不会成为 latest。
49c. plasticity before gain 近零时使用绝对 fallback，不做不稳定比值。
49d. 冷启动空 historical replay 只使 Gate 2 明确 `not_applicable`；已有历史后的读取/采样错误必须 fail。
49e. plasticity gate 的一次 update event 保留原 `finetune()` 内部步数、optimizer 重建周期和 LR 比例，且不污染 live state。

## Exception

50. 普通成功 episode 不创建 exception。
51. quick merge/repair 失败且 `Gainfast,cal` 有效时才生成 exception proposal；是否提交仍由独立 commit gates 决定。
52. 最近 prototype 路由正确。
53. adapter 上限和淘汰逻辑正确。
53a. 未达到 route threshold 时 slow-only；episode 内 route 固定。
53b. routed exception episode 只原子替换该 exception，不误写 global slow/replay/Q。
53c. 新建/更新 exception 也只能作为唯一 proposal 通过一次 commit-query。
53d. calibration fast gain 与 commit-query fast gain 分开计算和记录；proposal 阶段无法访问后者。
53e. 零/非有限/维度或 schema 不匹配的 context 一律 no-match；失败 exception proposal 不改变 local reservoir state。
53f. prototype/count 只在成功 exception commit 后按确定性 running mean 更新，失败/回滚 bitwise 不变；默认 routing 不读取 residual prototype。

## Checkpoint

* base checkpoint hash 或 module manifest 不匹配时拒绝加载；
* between-episode 保存加载后输出、router、replay/Q、counters 和所有 RNG 可复现；
* base-only 官方 checkpoint 可初始化全新 FD-PSC state；
* 已支持的旧 schema migration 正确，未知新 schema 拒绝；
* grouped Conv logical-layer state 保存加载无串组。
* latest pointer 损坏时只按 journal/hash 恢复最近完整不可变版本，不能选择部分写入文件。

---

# 26. 集成测试

实现小型、可在 CPU 运行的 mock AdaJEPA 风格网络和轨迹数据。

mock fixture 必须同时包含 Linear predictor、带 Conv2d projection head 的 frozen-backbone encoder、外部固定 calibration/commit-query/plasticity/report-test manifest，以及可验证的不重叠 trajectory IDs。

至少模拟：

1. 两个梯度一致 episode；
2. 两个梯度冲突 episode；
3. 新 episode 位于历史激活近零空间；
4. 新 episode 与 anchor 冲突；
5. slow rank 饱和；
6. repair 后成功；
7. repair 后仍失败并创建 exception；
8. checkpoint 保存后恢复并继续 episode。
9. 多个 calibration candidate 只产生一个 final proposal，commit-query 失败后终止；
10. 新 exception 创建、后续路由命中并由下一 episode 原子更新；
11. slow commit、exception commit 和 reject 三条路径对 global/local replay 与 Q 的更新边界；
12. report-test 读取不会改变任何模型或系统状态。
13. 无 online update、support 不足或零 task vector 时不生成 proposal，也不读取 commit-query。
14. `_plan_single()` 正常返回时 sleep 恰好执行一次；抛异常时 abort/rollback 且 sleep 执行零次，下一 episode 不继承 episodic/local buffer。

验证：

* 一致 episode 可合并；
* 冲突 episode 的 αshared 更小；
* safe 方向优先保留；
* anchor 不明显退化；
* plasticity 不明显下降；
* exception 只在必要时出现。
* commit-query 未被用于候选选择，所有路径在相同 seed 下确定性复现；
* FD-PSC 关闭时 mock 与官方 planner 原路径一致。

若真实 AdaJEPA 环境可运行，再增加至少一个真实 smoke test：

* 启动一个短 episode；
* 完成一次 online update；
* 完成一次 sleep；
* 保存 checkpoint；
* 重新加载；
* 继续下一 episode。

真实资源缺失时必须明确列出未运行原因；但 CPU mock、真实 `ViTPredictor` 的 device-agnostic 构造/forward、旧 checkpoint load path 和 FD-PSC 关闭兼容测试不能因此跳过。

自动化测试不得依赖网络、`torch.hub` 在线下载、真实机器人服务或已有用户缓存。DINO 等外部 backbone 用契约一致的本地 mock/monkeypatch 测试注入与 latent protocol；只有显式标记的真实 smoke test 可以使用用户提供且 checksum 匹配的本地 checkpoint/data。不得把下载失败误报为算法测试失败或偷偷修改全局 cache。

---

# 27. 基线和消融脚本

提供可直接运行的配置或脚本。

## 基线

1. Frozen AdaJEPA。
2. 原 selected-layer AdaJEPA。
3. Full-depth Linear + post-backbone ConvLoRA，每 episode 重置。
4. Full-depth Linear + post-backbone ConvLoRA 直接跨 episode 累积。
5. Episodic + slow LoRA，普通 SVD merge。

基线 3 使用与 FD-PSC 相同 target manifest/rank/LR，但每个 episode 从零 Pilot 开始并在结束后丢弃；基线 4 使用单个固定-rank adapter 跨 episode 持续训练且不 sleep、不建 bank；基线 5 使用相同 episodic/slow 双分支但只做无 soft-NESS、无 gates 的普通 factor-space SVD merge。不得用不同 target modules 偷换比较对象。

## 核心消融

6. w/o soft-NESS。
7. w/o current/history/anchor gates。
8. w/o plasticity gate。
9. w/o Triggered SLICE。
10. SLICE exact vs symmetric。
11. c-PCGrad vs dual-constraint。
12. 每步 PCGrad 负面对照。
13. SDC off / always / event-triggered。
14. Spectral Surgery off / output-only / all-target-logical-layers。
15. uniform replay vs GRASP。
16. JEPA repair vs JEPA + LPR。
17. reject conflict vs exception adapter。
18. slow rank 8 / 16 / adaptive。

每次实验输出统一 JSON 或 CSV。

所有 baseline/ablation 必须从同一 base checkpoint 和全新、隔离的 memory state 开始，使用相同 episode 顺序、seeds、support、external splits、canary budget 和真实 rollout budget。不同 run 不得共享可写 sidecar checkpoint。超参数选择和跨 run model selection 只能使用 calibration；commit-query 只能在每个 run 内按在线协议 gate 唯一 final proposal，不能查看其结果来选超参数、seed、ablation 或 checkpoint。最终表格使用 report-test 或独立真实 rollout，并明确 commit-query 不是 test set。

---

# 28. 指标

至少记录：

## 当前适应

* current JEPA loss；
* fast adaptation gain；
* external-calibration gain、commit-query gain 和 report-test gain，三者分开命名；
* planning success；
* 达到阈值需要的 replanning 次数。

## 历史保持

* historical replay loss；
* 每个 context 的 loss；
* worst-context regression；
* forgetting；
* backward transfer。

## 基础能力

* anchor loss；
* anchor gradient cosine；
* canary planning regression。

## 可塑性

* one-update-event plasticity gain；
* 下一 episode 前几步 loss 下降速度；
* plasticity gate ratio。

## 梯度几何

* ρhist；
* ρanchor；
* EMA；
* SLICE trigger；
* gradient correction norm；
* active constraints。

## 谱与子空间

* Dₗ；
* slow rank；
* episodic rank；
* Q rank；
* λ分布；
* p分布；
* spectral energy；
* εfunctional；
* αshared；
* αsafe。

## 系统成本

* online update latency；
* gradient collection latency；
* SLICE latency；
* sleep latency；
* replay memory；
* checkpoint size；
* adapter parameter count；
* exception count；
* routed exception id/similarity、route rejection count；
* calibration candidate count、final proposal type、commit-query gate invocation count；无 proposal 时为 0，有唯一 proposal 时为 1，同一次 gate 内比较 before/fast/candidate 的三个 forward 不计成三次 proposal evaluation；
* rollback count。

---

# 29. 禁止项

不得：

* 修改 θ₀；
* 将 slow LoRA merge 到 θ₀；
* 用同一数据训练、校准和 gate；
* 从已经参与 online support 更新的轨迹中事后抽取 calibration 或 commit-query；
* 外部 commit-query 参与 repair、候选选择、SLICE 或任何 optimizer step；
* commit-query 失败后基于结果尝试另一个 candidate、repair 或 exception；
* 使用 commit-query 作为最终论文/报告 test set；
* 每一步默认 PCGrad；
* 将 A、B分别正交后声称有效 BA 已正交；
* 在 grouped ConvLoRA 中跨 group 共享输入方向或进行非法 cross-group 混合；
* 只看参数重构误差；
* 忽略 functional error；
* 将所有 episode 保存成独立 adapter；
* 将 current local buffer 替换为 historical replay；
* 用最终 JEPA embedding 作为长期 replay 切点；
* 将输入激活 SVD 的左奇异向量误当输入方向；
* 假设 Centered LoRA task vector rank 仍为 r；
* 在 gate 前更新 Q；
* 将相对于 routed exception 学到的 episodic delta 直接写入 global slow；
* rank 超限时静默截断；
* 在没有真实 rollout 时伪造 planning success；
* 留下无实现的 `TODO` 或静默 stub；
* 吞掉异常继续提交错误状态。

---

# 30. 实现阶段

按以下顺序执行，但最终提交应包含完整系统。

## Phase A：仓库映射与基础 LoRA

* 审计报告；
* 目标模块清单；
* Dual LoRA；
* Linear/ConvLoRA 与 grouped logical-layer factor container；
* 注入；
* encoder latent adapter protocol；
* checkpoint；
* 原行为兼容。
* device-agnostic predictor mask 与 CPU compatibility。

## Phase B：核心 V1

* episodic + slow；
* replay；
* external manifest、commit-query/plasticity/report-test 隔离；
* activation subspace；
* soft-NESS；
* factor-space merge；
* rank manager；
* gates；
* rollback。
* single-final-proposal state machine。

## Phase C：梯度模块

* hooks；
* gradient cosine；
* c-PCGrad；
* dual-constraint；
* Pilot + Triggered SLICE；
* Centered LoRA；
* magnitude matching。

## Phase D：谱控制

* base SVD cache；
* SDC；
* Spectral Surgery。

## Phase E：repair 和 exception

* LPR repair；
* GRASP；
* prototype router；
* exception lifecycle。

## Phase F：测试与文档

* 全部单元测试；
* mock 集成测试；
* 真实 smoke test；
* 基线脚本；
* README；
* 实现报告。

每个阶段完成后运行相关测试，不要到最后才一次性调试。

---

# 31. 最终交付内容

完成后给出：

1. 修改文件清单。
2. 架构映射说明。
3. 新增配置说明。
4. 关键算法实现位置。
5. 测试结果。
6. 未运行的测试及原因。
7. 真实环境 smoke test 结果。
8. 性能和显存开销。
9. 与原 AdaJEPA 的兼容性说明。
10. 已知限制。
11. 下一步实验建议。
12. 可复制的运行命令。
13. external calibration/commit-query/plasticity/report-test/anchor manifest schema、生成命令、hash 和泄漏审计结果。
14. 每个 checkpoint/encoder 变体的目标模块 manifest，以及零 projection-target 是否为合法 not-applicable。

同时生成一份：

`docs/fd_psc_design.md`

内容包括：

* 状态机；
* episode 流程；
* sleep 流程；
* 数学公式；
* 组件开关；
* checkpoint schema；
* failure handling；
* 论文方法与代码模块对应表。

---

# 32. 最终验收条件

只有满足以下条件才算完成：

* FD-PSC 关闭时原模型行为不变；
* θ₀全部参数和持久 buffer bitwise 不变化；
* 全深度 LoRA 正确注入；
* post-backbone projection head ConvLoRA 默认启用且正确注入；
* frozen visual backbone 内没有 adapter；
* grouped ConvLoRA 按 logical group 隔离且 padding 语义正确；
* Pilot 和 Centered 切换函数连续；
* 梯度 hooks 正确；
* SLICE 条件触发；
* SLICE 不改变原 online update 次数和 optimizer 重建周期；
* 每步硬 PCGrad 不是默认行为；
* SDC 是事件触发软控制；
* soft-NESS 使用输入空间右奇异向量；
* 低秩合并不展开大矩阵；
* slow rank 由谱能量和功能误差共同决定；
* support/calibration/commit-query/plasticity/report-test/anchor 严格隔离；
* calibration/commit-query/plasticity/report-test/anchor 来自版本固定并通过 checksum 验证的外部 manifest；
* calibration 只选出一个 final proposal，commit-query 只 gate 一次且失败后不重试；
* report-test 从未参与算法状态或决策；
* gates 完整；
* 提交可回滚；
* slow、exception、replay、Q/λ、router、counter 和 RNG 在事务中一致提交/回滚；
* Q 只在成功的 global slow 提交后更新；
* 只有 slow commit 更新 global replay/Q，exception 使用有界 local replay；
* repair 和 exception 路径可运行；
* 独立 sidecar checkpoint 可原子保存、校验并恢复，base hash/manifest 不匹配时拒绝；
* 单元测试和 mock 集成测试通过；
* 所有 fallback 有日志；
* 没有未实现的核心 stub。

核心架构必须保持：

> 梯度点积负责发现冲突；Triggered SLICE 负责校准起始学习方向；SDC 负责限制 episode 内谱漂移；全深度 episodic LoRA 负责快速学习；soft-NESS 负责识别历史重要方向和安全近零空间；QR + 小矩阵 SVD 负责压缩进单个 slow LoRA；external commit-query、replay、anchor、plasticity 以及启用时的真实规划 canary 共同决定是否提交。
