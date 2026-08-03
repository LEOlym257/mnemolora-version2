# AdaJEPA → FD-PSC repository audit

Audit baseline: Git commit `a29975964f966f2836a2c7e26f464367c795c333`.

This audit records the concrete integration map used by the implementation. It
is intentionally separate from the algorithm design: no class or path below is
hypothetical.

## Online adaptation call chain

The planning entry point is `PlanWorkspace.perform_planning()` in `plan.py`.
Hydra constructs `planning.adajepa_mpc.AdaJEPAMPCPlanner`, whose `plan()` loops
over planning samples. One call to its `_plan_single()` delegates to the
inherited `planning.mpc.MPCPlanner.plan()` and is therefore one FD-PSC episode.

At every MPC replan the inherited loop executes actions, receives the
cumulative environment trajectory, and invokes `_post_env_feedback()`. The
AdaJEPA override extracts exactly `T + 1` model-stride observations aligned to
the `T` executed actions, appends that segment to `_obs_buffer/_act_buffer`,
applies the configured `recent<N>` or `hard<N>` policy, and calls
`AdaJEPATrainer.finetune()` at the existing `finetune_every` cadence.

`planning.adajepa.AdaJEPATrainer.finetune()` is the only online model-update
entry point. It preserves these semantics in FD mode:

- one optimizer is rebuilt for every `finetune()` invocation;
- configured internal `steps` share that optimizer until a completed step
  activates Centered/SLICE; when another real step remains, the optimizer is
  rebuilt immediately with the same class and predictor/encoder learning-rate
  groups so the new Centered factors are trainable on the next step;
- Adam, AdamW, and SGD use PyTorch defaults, exactly as before;
- predictor and encoder-head parameters remain separate groups with `lr` and
  `encoder_lr`;
- contiguous recent segments are merged; hard-buffer segments are not;
- the prediction target is detached according to `wm.stop_grad`;
- no extra optimizer step is introduced when Pilot switches to Centered.
- conflict triggering is evaluated once after every actual optimizer step,
  while SDC event bookkeeping and the replan index advance only once per
  `finetune()` event.

The loss is `_prediction_loss()`: sliding-window, one-step latent MSE over all
valid windows. For token concatenation it excludes the action token; for
feature concatenation it excludes the action feature dimensions. This is the
JEPA loss used by online adaptation, replay, repair, and probes.

Before constructing an enabled FD-PSC trainer, `PlanWorkspace` hashes the
versioned `Preprocessor` contract (all normalization statistics, observation
transform, encoder transform, frameskip, and history/prediction window
geometry). `AdaJEPAMPCPlanner` passes that runtime identity through
`AdaJEPATrainer` to the memory system. External-data and canary manifests are
both compared with it at startup; online support records store the runtime
hash rather than copying an unverified manifest declaration. The planner also
exposes an injectable isolated canary evaluator; when none is configured, the
manifest-backed runner records the configured unavailable policy instead of
touching the live MPC worker.

Each online support segment freezes its episode/context/preprocess/schema and
trajectory/transition/frame/content identity at registration. A single short
replan is not silently discarded: consecutive segments may form one replay
window only when they belong to the same trajectory, share an identical
stable boundary frame and tensor, and pass the external-split audit again as a
composed identity. Non-contiguous, ambiguous, or forged boundaries fail
closed. The history cold-start decision is backed by the persistent successful
slow-commit count, not by whether a bounded replay sample happens to be empty.

## Existing reset and checkpoint behavior

The original trainer snapshots selected dense parameters plus every registered
predictor/encoder buffer in `_take_snapshot()`, then `reset()` copies those
tensors back. The original planner calls reset before each sample and after the
sample loop. That path is retained verbatim when FD-PSC is disabled. It is not
used in FD mode because it would overwrite persistent adapter state and can
also conflate episodic cleanup with train-mode buffer restoration.

Official training checkpoints serialize complete encoder/predictor/action and
proprio modules. `plan.load_ckpt()` loads those base modules before the world
model is constructed. FD-PSC injection therefore occurs only after the base
checkpoint is loaded, and persistent memory is stored in a separate sidecar.

## Real model paths

The world model is `models.visual_world_model.VWorldModel`:

- visual encoder: `wm.encoder`;
- predictor: `wm.predictor`;
- action encoder: `wm.action_encoder`;
- proprio encoder: `wm.proprio_encoder`.

The released DINO configuration instantiates `models.dino.DinoV2Encoder`.
Its permanently frozen visual backbone is `wm.encoder.base_model`. Its
post-backbone projection head, when present, is `wm.encoder.projector` and is
one of:

- `ChannelProjector`: `projector.conv_layers.*` (`Conv2d`);
- `GlobalProjector`: `projector.mix`, Conv2d entries of
  `projector.down_blocks`, and `projector.head`.

GroupNorm, BatchNorm, LayerNorm, activations, pooling, and bias are never FD
targets. DINO `agg_mlp` is used only by the optional aggregation call, not by
the JEPA `VWorldModel.encode_obs()` path, and is therefore inactive for the
default online loss.

The predictor is `models.vit.ViTPredictor`. Every transformer depth contains:

- `transformer.layers.<i>.0.to_qkv`: fused attention QKV Linear, tagged
  `attention_qkv` (it is not three imaginary modules);
- `transformer.layers.<i>.0.to_out.0`, when projection is not Identity, tagged
  `attention_output`;
- `transformer.layers.<i>.1.net.1`: MLP input Linear;
- `transformer.layers.<i>.1.net.4`: MLP output Linear.

All of these active Linear layers are targets. The final transformer norm and
positional embedding are frozen. There is no separate predictor final Linear
in this implementation.

Action/proprio encoders are disabled targets by default. If enabled, only
their active `nn.Linear` descendants are eligible. The official
`ProprioceptiveEmbedding` uses Conv1d and therefore has no eligible default
Linear. Dummy encoder Linear members are declared but unused; the runtime
reachability pass marks them inactive and rejects their injection.

## Latent replay cut

For `DinoV2Encoder`, the stable replay cut is the tensor returned by
`base_model.forward_features(...)[feature_key]`, before `encoder.projector`.
The explicit adapter protocol records token layout and, for projection heads,
the square token-to-feature-map conversion needed to replay Conv2d exactly.
The protocol exposes `extract_frozen_visual_latent()` and
`project_visual_latent()`; replay does not depend on a transient hook.

An encoder with no projection head may use an explicit identity projection
only when its final visual latent is fully frozen. An encoder whose internal
backbone/head boundary cannot be identified is rejected with an actionable
error instead of guessing a module path.

Historical replay stores a `theta0_jepa_pattern_v1` residual descriptor from
that exact frozen latent.  Extraction disables slow, exception, and episodic
adapters and runs the world model in eval mode.  For the unchanged sliding
one-step JEPA windows and observation-only prediction mask, it concatenates:

1. signed mean and RMS for each `(history position, feature)`, pooling batch,
   window offset, and observation-token axes; and
2. signed mean and RMS for each `(history position, observation token)`,
   pooling batch, window offset, and feature axes.

The row-major float32 CPU vector therefore has a fixed shape for a model while
retaining temporal, feature, and token/spatial residual patterns.  Scalar JEPA
MSE remains separate `difficulty_score` metadata.  The stored descriptor stays
raw.  Candidate-grid pruning may form a transient raw-mean-then-normalize
prototype for cosine comparison; persistent normalization happens only in the
atomic exception-prototype update.  Both paths preserve the explicit
zero/unavailable case.

## Lifecycle hazards found and corrected

- The historical attention mask was allocated with `.to('cuda')` in every
  `Attention` constructor. It is now a device-moving, `persistent=False`
  boolean buffer, so CPU construction works and no checkpoint key is added.
- The original per-sample loop had no outer rollback boundary. FD mode wraps
  `_plan_single()` so an exception aborts the episode, restores persistent
  state, clears local buffers, and never creates a sleep proposal.
- The original final `reset()` restores dense snapshots. FD mode replaces it
  with episodic cleanup only; slow memory, historical replay, subspaces, and
  exception state survive across samples.
- Calling `encoder.train()` can advance BatchNorm statistics in a projection
  head. FD mode freezes every base parameter and persistent buffer and keeps
  frozen backbone/normalization modules in eval mode while exposing only
  episodic adapter parameters to the online optimizer.
- The original `finetune()` lacked exception-safe mode/gradient restoration.
  The FD path uses `try/finally`, and the unchanged legacy path remains
  available when the feature is disabled.

## Runtime target manifest

`fd_psc.injector` creates the machine-readable manifest after loading the base
checkpoint and before replacing modules. Candidate paths come from the real
module tree. A schema-only zero dry run, with Python/NumPy/Torch RNG saved and
restored, records reachability without using or retaining any experiment
split. Each entry records path, type, dimensions, Conv2d geometry, group,
logical group ID, semantic tag, output-writing status, active-forward status,
and default injection decision. The sidecar stores both the manifest and its
SHA-256; load fails on any path/type/shape/geometry/hash mismatch.

Zero projection targets are `not_applicable` when a supported encoder has no
projection head. A present head with no discovered active Linear/Conv2d is a
configuration error. Any target below `base_model` is a hard error.
