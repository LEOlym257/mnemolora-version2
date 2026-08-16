# MnemoLoRA: Closed-Loop Trust-Region Consolidation and Capacity Recycling

MnemoLoRA is a two-timescale memory system for continual test-time adaptation of latent world models. It keeps a pretrained model bitwise frozen, learns a temporary low-rank adapter during each episode, and persists only a guarded, bounded-memory update between episodes.

> **Research implementation.** The repository validates the FSD V2 mathematical and systems contracts on deterministic fixtures and toy models. The recorded paper snapshot reports 237/237 repository tests, 57/57 focused V2 tests, and a 105-case FP32/FP16/BF16 RTX 4060 operator audit. Released-checkpoint, multi-seed continual-control performance is not established by these tests.

## What problem does MnemoLoRA solve?

An adaptive world model can use self-supervised transitions to correct prediction errors while an agent is acting. The next question is what should survive the episode. Resetting the adapter discards useful corrections; retaining every optimizer update accumulates interference, rank, and recovery state.

MnemoLoRA separates immediate adaptation from persistent memory:

| Timescale | State | Role |
|---|---|---|
| Wake, inside an episode | Episodic LoRA `E_t` | Fast JEPA updates from the current trajectory. It is reset at the episode boundary and never mutates persistent memory. |
| Sleep, between episodes | Slow LoRA `S_t` | Bounded-rank memory produced by fresh replay geometry, trust-region projection, and QR–SVD compression. |
| Lifetime | Dense core `C_t` plus `S_t` | Fixed-size long-term memory written only by Deep Sleep; the official base `theta_0` remains immutable. |

For each adaptable Linear or grouped-Conv2d module, the effective weight is

```text
W_t,l = W_0,l + C_t,l + S_t,l + E_t,l
```

The implementation preserves the original AdaJEPA wake objective and MPC loop. It changes how the completed episode update is measured, compressed, persisted, and recovered.

## Wake–sleep protocol

The FSD V2 state machine is:

```text
idle
  -> wake: train episodic LoRA on the ordinary online JEPA loss
  -> geometry: re-embed committed raw replay under the persistent model
  -> RTRC: project the task update under one shared relative drift budget
  -> compress: merge factors with thin QR and a small-core SVD
  -> [Deep Sleep: write core and refit residual LoRA when compression overflows]
  -> commit: atomically publish memory, replay, metrics, controllers, and RNG
  -> idle
```

Every sleep state has a rollback edge. A failed Deep Sleep therefore returns to the already-computed ordinary compressed candidate; a failed transaction restores the complete pre-sleep state.

### 1. Isolated episodic plasticity

At episode start, the system verifies the base, target manifest, configuration, and preprocessing identities, resets `E_t`, and freezes `C_t` and `S_t`. The existing online JEPA loss and optimizer schedule then update only episodic parameters. RTRC is applied after the episode, not inside every wake optimizer step, so wake learning keeps its original adaptation semantics.

### 2. Fresh geometry from internal raw replay

Committed replay stores raw model inputs—observations, actions, preprocessing identity, and context metadata—not cached latents treated as truth. At sleep, replay is re-embedded through `theta_0 + C_t + S_t` with episodic adapters disabled. For each target layer, the system builds a compact second-moment geometry from the current inputs and conservatively completes the discarded spectrum. Re-embedding keeps the geometry current after persistent memory changes.

Deep Sleep has the same data boundary: its corpus is restricted to internal raw replay plus current-episode support. It does not read external calibration, commit-query, anchor, or report-test data.

### 3. Shared-dual representation trust region (RTRC)

For task factors `T_l = B_l A_l`, RTRC minimizes the factor change subject to one model-level constraint:

```text
min_A'  1/2 sum_l ||B_l (A'_l - A_l)||_F^2
        s.t. sum_l omega_l tr(B_l A'_l Sigma_hat_l A'_l^T B_l^T) <= delta_t
```

The budget is `delta_t = beta_t D_hat(T)`. A single dual variable couples heterogeneous layers. The production path stays in factor space: it does not materialize dense task matrices or dense covariance matrices, and it recomputes feasibility after the adapter storage-dtype cast.

### 4. Closed-loop budget adaptation

The scalar `beta_t` is updated from losses measured at three internal states: persistent-before-wake, fast episodic, and persistent-plus-uncompressed-task. Current-support loss measures how much useful wake gain consolidation removes; replay loss measures historical regression. The controller increases the next budget when plasticity is being erased and decreases it when replay stability is being lost. Missing signals contribute zero rather than a fabricated measurement, and controller state is checkpointed.

### 5. Fixed-rank compression and Deep Sleep

The normal path concatenates old slow factors with the accepted task factors and recompresses them to the configured slow rank. The system records both factor-space truncation error and storage-cast error.

Deep Sleep is triggered only when the maximum relative compression error exceeds `0.05` for three consecutive commits and at least 64 historical raw windows are available. It fixes the teacher at the start of sleep:

```text
teacher = theta_0 + core_old + slow_uncompressed
```

With slow LoRA temporarily disabled, only the dense core is optimized. For internal replay and current support, Deep Sleep distills:

- output residuals between the fixed teacher and the core-only reference;
- residuals from a bounded set of critical hidden layers; and
- the current JEPA auxiliary loss.

It then fits a deterministic rank-8 parameter residual to the difference between the uncompressed teacher update and the new core. The default optimizer budget is 100 Adam steps with output/hidden/current weights `1 / 0.25 / 0.25`. The resulting `core_new + residual` candidate is checked against the complete internal corpus. If relative output error exceeds `0.02` (with an absolute near-zero fallback), the core, residual memory, controller counters, sampler RNG, and global RNG are restored and the normal compressed candidate is committed instead.

### 6. Transactional persistence and resume

The transaction covers the dense core, slow residual memory, raw replay, adaptive-budget and Deep Sleep controllers, state machine, metrics, version counters, and CPU/CUDA RNG state. A schema-2 immutable sidecar is written first, followed by an atomically replaced latest pointer. Checkpoint load is legal only between episodes and revalidates the base hash, target manifest, preprocessing identity, configuration identity, commit sequence, and payload hash. A write, validation, or publication failure cannot leave a partially committed memory state.

## Implementation defaults

The shipped V2 configuration is [`conf/fd_psc/fsd_v2.yaml`](conf/fd_psc/fsd_v2.yaml):

| Component | Default |
|---|---|
| Episodic LoRA | rank 8, `alpha=16`, dropout 0 |
| Slow memory | persistent and maximum rank 32 |
| RTRC | `beta_0=0.20`, range `[0.02, 1.0]`, one shared dual, 80 bisections |
| Fresh geometry | up to 128 replay windows, 64 directions, energy `0.999`, conservative tail |
| Raw replay | 512 windows, at most 32 context clusters, at least 4 windows per cluster |
| Budget controller | learning rate `0.25`, plasticity target `0.10`, history target `0.01` |
| Deep Sleep | trigger `0.05 × 3`, minimum 64 replay windows, max 100 Adam steps, residual rank 8 |
| Deep Sleep fidelity | relative squared output error `<= 0.02` |
| Checkpoint | schema 2, immutable versions, atomic latest pointer, every episode |

The V2 path deliberately disables legacy gradient-surgery, external commit-query gates, exception banks, canaries, and repair search. Those components remain available only for the legacy FD-PSC run modes; they are not part of the V2 algorithm described here.

## Installation

```bash
git clone https://github.com/LEOlym257/mnemolora-version2.git
cd mnemolora-version2
conda env create -f environment.yaml
conda activate ts
```

MuJoCo is required only for the maze environments. The protocol tests below use synthetic fixtures and do not download checkpoints, datasets, or external manifests.

## Run FSD V2 with AdaJEPA

FSD V2 is selected as a Hydra config group. Supply a real AdaJEPA checkpoint and evaluation data when running a planner:

```bash
python plan.py --config-name adajepa_plan_cem_maze \
    fd_psc=fsd_v2 \
    seed=100 \
    ckpt_base_path=./checkpoints \
    model_name=mediummaze_dynamics_shift \
    eval_data_path=./data/point_maze_medium \
    +wandb_logging=false
```

The same `fd_psc=fsd_v2` override can be used with the PushT/PushObj and diverse-maze planner configs. FSD V2 creates its own raw replay and sidecar state; it does not require the legacy six-split external manifest. Use an isolated state directory for every checkpoint/seed run, for example:

```text
fd_psc.checkpoint.state_directory=fsd_v2_state
fd_psc.checkpoint.latest_pointer_path=fsd_v2_state/latest.json
```

For the full checkpoint identity, target enumeration, resume, and failure semantics, see the [FD-PSC usage guide](docs/fd_psc.md) and [design/state-machine reference](docs/fd_psc_design.md).

## Tests and evidence boundary

Run the focused V2 suite with the standard-library runner:

```bash
python -m unittest \
    tests.test_fsd_v2_config \
    tests.test_fsd_v2_raw_replay \
    tests.test_fsd_v2_rtrc \
    tests.test_fsd_v2_no_external_data \
    tests.test_fsd_v2_budget_controller \
    tests.test_fsd_v2_deep_sleep \
    tests.test_fsd_v2_residual_distillation -v
```

Run the repository suite with:

```bash
python -m unittest discover -s tests -v
```

The paper's recorded snapshot reports 57 focused V2 tests and 237 repository tests passing. These fixtures cover configuration identity, raw replay and re-embedding, shared-dual RTRC, adaptive budgets, Deep Sleep residual math, checkpoint/resume, transaction rollback, and RNG restoration. They establish software and protocol behavior; they do not establish planning success, backward transfer, long-run forgetting, throughput, or production memory growth.

The implementation report records the remaining boundary explicitly: real released-checkpoint loading, target manifests for every encoder variant, real MPC rollouts, multi-seed baselines, and end-to-end performance benchmarks are `UNRUN` unless a report says otherwise. See [`docs/fd_psc_implementation_report.md`](docs/fd_psc_implementation_report.md).

## Repository layout

| Path | Purpose |
|---|---|
| `fd_psc/v2/` | FSD V2 system, state machine, RTRC, replay geometry, Deep Sleep, and checkpoint store |
| `fd_psc/config.py` | Typed configuration and V2 fail-fast validation |
| `fd_psc/injector.py`, `fd_psc/lora_layers.py` | Active target discovery and Linear/ConvLoRA injection |
| `planning/adajepa.py` | AdaJEPA wake-loop integration |
| `conf/fd_psc/fsd_v2.yaml` | Shipped V2 defaults |
| `tests/test_fsd_v2_*.py` | Focused protocol and mathematical tests |
| `docs/fd_psc.md` | Operational usage, checkpoint, and experiment guidance |
| `docs/fd_psc_design.md` | Design-to-code mapping and state-machine details |

## Relationship to AdaJEPA

This repository contains the AdaJEPA-based latent world-model and its test-time adaptation loop, built on the [temporal-straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) and [DINO-WM](https://github.com/gaoyuezhou/dino_wm) code lineages. With `fd_psc=disabled`, the original AdaJEPA checkpoint, optimizer path, and state-dict behavior are preserved. With `fd_psc=fsd_v2`, the base model remains frozen and only the episodic LoRA is trainable during wake.

Released checkpoints and task data are not included in this repository. Obtain them from the project release location and keep checkpoint hashes and run manifests with every experiment.

## Citation

If you use the method or implementation, please cite the paper and the AdaJEPA foundation:

```bibtex
@article{mnemolora2026,
  title={MnemoLoRA: Closed-Loop Trust-Region Consolidation and Capacity Recycling for Continual World Models},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2026}
}

@article{wang2026adajepa,
  title={AdaJEPA: An Adaptive Latent World Model},
  author={Wang, Ying and Bounou, Oumayma and LeCun, Yann and Ren, Mengye},
  journal={arXiv preprint arXiv:2606.32026},
  year={2026}
}
```

## Acknowledgement

We build on [temporal-straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) and [DINO-WM](https://github.com/gaoyuezhou/dino_wm), and thank their authors for making the implementations available.
