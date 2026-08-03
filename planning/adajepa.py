import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from utils import move_to_device

log = logging.getLogger(__name__)


class AdaJEPATrainer:
    """AdaJEPA trainer: test-time adaptation of the predictor (and optionally
    the encoder) on trajectory segments observed from the environment during
    planning, using the same sliding-window 1-step latent prediction loss as
    training. Defaults follow the paper: one adaptation step on the predictor's
    last transformer layer and the encoder's head."""

    def __init__(
        self,
        wm,
        lr: float,
        steps: int = 1,
        optimizer_name: str = "adam",
        finetune_encoder: bool = True,
        last_layer_only: bool = True,
        encoder_lr: float = None,
        encoder_last_layer_only: bool = True,
        fd_psc=None,
        runtime_output_dir: Optional[str] = None,
        fd_psc_canary_evaluator=None,
        fd_psc_runtime_preprocess_hash: Optional[str] = None,
    ):
        self.wm = wm
        self.lr = lr
        self.encoder_lr = encoder_lr if encoder_lr is not None else lr
        self.steps = steps
        self.optimizer_name = optimizer_name.lower()
        self.finetune_encoder = finetune_encoder
        self.last_layer_only = last_layer_only
        self.encoder_last_layer_only = encoder_last_layer_only
        self.device = next(wm.parameters()).device
        self.criterion = nn.MSELoss()
        # Import and construct the memory system only on the enabled path.  A
        # disabled configuration therefore performs no replacement, hook
        # registration, or state_dict mutation.
        from fd_psc.config import FDPSCConfig

        self.fd_psc_config = FDPSCConfig.from_mapping(fd_psc)
        self.fd_psc_system = None
        if self.fd_psc_config.enabled:
            from fd_psc.trainer import FDPSCSystem

            self.fd_psc_system = FDPSCSystem(
                wm=self.wm,
                config=self.fd_psc_config,
                runtime_output_dir=Path(runtime_output_dir or "."),
                canary_evaluator=fd_psc_canary_evaluator,
                runtime_preprocess_hash=fd_psc_runtime_preprocess_hash,
            )
            self._ada_predictor_params = []
            self._ada_encoder_params = []
            self._snapshot = []
        else:
            self._ada_predictor_params = self._select_predictor_params()
            self._ada_encoder_params = self._select_encoder_params() if self.finetune_encoder else []
            self._snapshot = self._take_snapshot()

    @property
    def fd_psc_enabled(self) -> bool:
        return self.fd_psc_system is not None

    def _take_snapshot(self):
        """Snapshot the tensors adaptation can touch: selected params + train-mode buffers (e.g. BN stats)."""
        tensors = list(self._ada_predictor_params)
        tensors += [b for _, b in self.wm.predictor.named_buffers()]
        if self.finetune_encoder:
            tensors += self._ada_encoder_params
            tensors += [b for _, b in self.wm.encoder.named_buffers()]
        return [(t, t.detach().clone()) for t in tensors]

    @torch.no_grad()
    def reset(self):
        """Restore the pre-adaptation values of the adapted tensors."""
        if self.fd_psc_system is not None:
            self.fd_psc_system.reset_episode()
            return
        for tensor, saved in self._snapshot:
            tensor.copy_(saved)

    def finetune(self, obs_seqs: list, act_seqs: list, merge: bool = True) -> list:
        """Run `steps` optimization steps on the given trajectory segments.

        merge=True concatenates temporally contiguous segments into one long
        sequence; use merge=False for non-contiguous segments.
        Returns the per-step prediction losses.
        """
        if self.fd_psc_system is not None:
            return self._finetune_fd_psc(obs_seqs, act_seqs, merge=merge)
        if not obs_seqs or self.steps <= 0:
            return []
        if merge and len(obs_seqs) > 1:
            obs_seqs, act_seqs = self._merge_segments(obs_seqs, act_seqs)
        segments = [self._prepare_segment(o, a) for o, a in zip(obs_seqs, act_seqs)]
        if not self.finetune_encoder:
            # Encoder frozen: embeddings can be precomputed once.
            with torch.no_grad():
                all_z = [self.wm.encode(o, a).detach() for o, a in segments]

        self._set_requires_grad(self.wm.predictor, self._ada_predictor_params, True)
        self.wm.predictor.train()
        if self.finetune_encoder:
            self._set_requires_grad(self.wm.encoder, self._ada_encoder_params, True)
            self.wm.encoder.train()
            base_model = getattr(self.wm.encoder, "base_model", None)
            if base_model is not None:
                base_model.eval()  # keep the frozen backbone in eval mode

        optimizer = self._make_optimizer()
        detach_src = not self.finetune_encoder
        detach_tgt = True if not self.finetune_encoder else bool(getattr(self.wm, "stop_grad", True))
        step_losses = []
        for step in range(self.steps):
            optimizer.zero_grad()
            if self.finetune_encoder:
                # Re-encode each step so gradients reach the encoder.
                all_z = [self.wm.encode(o, a) for o, a in segments]
            loss = torch.stack(
                [
                    self._prediction_loss(z, detach_src=detach_src, detach_tgt=detach_tgt)
                    for z in all_z
                ]
            ).mean()
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
            log.info("AdaJEPA step %d/%d  pred_loss=%.6f", step + 1, self.steps, step_losses[-1])

        self.wm.predictor.eval()
        self._set_requires_grad(self.wm.predictor, self._ada_predictor_params, False)
        if self.finetune_encoder:
            self.wm.encoder.eval()
            self._set_requires_grad(self.wm.encoder, self._ada_encoder_params, False)
        return step_losses

    def _finetune_fd_psc(self, obs_seqs: list, act_seqs: list, merge: bool = True) -> list:
        """Run the unchanged AdaJEPA update schedule on episodic adapters only."""
        if not obs_seqs or self.steps <= 0:
            return []
        system = self.fd_psc_system
        system.require_active_episode()
        if merge and len(obs_seqs) > 1:
            obs_seqs, act_seqs = self._merge_segments(obs_seqs, act_seqs)
        segments = [self._prepare_segment(o, a) for o, a in zip(obs_seqs, act_seqs)]
        if not self.finetune_encoder:
            with torch.no_grad():
                all_z = [self.wm.encode(o, a).detach() for o, a in segments]

        system.prepare_online_mode(
            predictor_train=True,
            encoder_train=self.finetune_encoder,
        )
        optimizer = self._make_optimizer()
        detach_src = not self.finetune_encoder
        detach_tgt = True if not self.finetune_encoder else bool(getattr(self.wm, "stop_grad", True))
        step_losses = []
        try:
            for step in range(self.steps):
                optimizer.zero_grad()
                def loss_closure():
                    step_z = (
                        [self.wm.encode(o, a) for o, a in segments]
                        if self.finetune_encoder
                        else all_z
                    )
                    return torch.stack(
                        [
                            self._prediction_loss(
                                z,
                                detach_src=detach_src,
                                detach_tgt=detach_tgt,
                            )
                            for z in step_z
                        ]
                    ).mean()

                forward_rng = system.capture_update_rng()
                loss = loss_closure()
                # Event-triggered SDC owns the exact two-pass backward.  When
                # inactive this is precisely the original single backward;
                # the JEPA scalar, target detach semantics, optimizer, and
                # optimizer recreation schedule remain unchanged.
                system.backward_with_sdc(
                    loss,
                    optimizer,
                    loss_closure=loss_closure,
                    forward_rng=forward_rng,
                )
                optimizer.step()
                step_losses.append(float(loss.detach()))
                system.note_optimizer_step(step + 1, step_losses[-1])
                centered_activated = system.after_optimizer_step(
                    self,
                    segments,
                    step_losses,
                )
                log.info(
                    "FD-AdaJEPA step %d/%d  pred_loss=%.6f",
                    step + 1,
                    self.steps,
                    step_losses[-1],
                )
                if centered_activated and step + 1 < self.steps:
                    # SLICE creates new Centered Parameter objects.  Rebuild
                    # with the unchanged optimizer class and LR split before
                    # the next real step so those parameters participate.
                    optimizer.zero_grad(set_to_none=True)
                    optimizer = self._make_optimizer()
            system.after_finetune_event(
                self,
                segments,
                step_losses,
                conflict_evaluated_per_step=True,
            )
            return step_losses
        finally:
            optimizer.zero_grad(set_to_none=True)
            system.finish_online_mode()

    def begin_fd_psc_episode(self, episode_id, context_identifier, initial_obs=None, metadata=None):
        if self.fd_psc_system is None:
            raise RuntimeError("FD-PSC is disabled")
        return self.fd_psc_system.begin_episode(
            episode_id=str(episode_id),
            context_identifier=str(context_identifier),
            initial_obs=initial_obs,
            metadata=metadata or {},
        )

    def abort_fd_psc_episode(self, reason="planner_exception"):
        if self.fd_psc_system is not None:
            self.fd_psc_system.abort_episode(str(reason))

    def end_fd_psc_episode(self, obs_seqs, act_seqs):
        if self.fd_psc_system is None:
            raise RuntimeError("FD-PSC is disabled")
        return self.fd_psc_system.end_episode_and_sleep(self, obs_seqs, act_seqs)

    @torch.no_grad()
    def score_segments(self, obs_seqs: list, act_seqs: list) -> list:
        """Prediction loss per segment under the current weights (higher = harder)."""
        return [
            float(self._prediction_loss(self.wm.encode(*self._prepare_segment(o, a))))
            for o, a in zip(obs_seqs, act_seqs)
        ]

    @staticmethod
    def _tensorize_payload(value: Any, *, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {
                key: AdaJEPATrainer._tensorize_payload(item, device=device)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            # Numeric nested JSON arrays become tensors; structural lists are
            # recursively retained for metadata such as leading_shape.
            try:
                tensor = torch.as_tensor(value)
                if tensor.dtype != torch.bool and not tensor.is_floating_point():
                    tensor = tensor.float()
                return tensor.to(device)
            except (TypeError, ValueError):
                return [
                    AdaJEPATrainer._tensorize_payload(item, device=device)
                    for item in value
                ]
        return value

    def _external_payload_to_z(self, payload: Mapping[str, Any]) -> torch.Tensor:
        payload = self._tensorize_payload(payload, device=self.device)
        if "z" in payload:
            z = payload["z"]
            if not isinstance(z, torch.Tensor) or z.ndim != 4:
                raise ValueError("external payload z must be [batch,time,patch,dim]")
            return z
        if "obs" in payload and ("act" in payload or "actions" in payload):
            obs = payload["obs"]
            act = payload.get("act", payload.get("actions"))
            if not isinstance(obs, Mapping) or not isinstance(act, torch.Tensor):
                raise ValueError("raw external payload requires tensor obs/actions")
            obs, act = self._prepare_segment(obs, act)
            return self.wm.encode(obs, act)
        latent_payload = payload.get("frozen_visual_latent", payload.get("visual_latent"))
        if latent_payload is None:
            raise ValueError(
                "external payload requires z, raw obs/actions, or frozen_visual_latent"
            )
        if not isinstance(latent_payload, Mapping):
            raise ValueError("frozen_visual_latent must be a mapping")
        from fd_psc.encoder_adapters import FrozenVisualLatent

        latent = FrozenVisualLatent.from_payload(latent_payload)
        proprio = payload.get("proprio")
        actions = payload.get("actions", payload.get("act"))
        if not isinstance(proprio, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise ValueError("frozen-latent payload requires proprio and actions tensors")
        projected_t = int(latent.metadata.get("leading_shape", (actions.shape[0], actions.shape[1]))[-1])
        if actions.shape[1] + 1 == projected_t:
            actions = torch.cat([actions, torch.zeros_like(actions[:, :1])], dim=1)
        if proprio.shape[1] != projected_t or actions.shape[1] != projected_t:
            raise ValueError(
                "external latent/proprio/action time dimensions do not form one continuous window"
            )
        return self.wm.encode_from_frozen_visual_latent(latent, proprio, actions)

    def external_loss_tensor(
        self,
        records: Sequence[Any],
        *,
        detach_src: Optional[bool] = None,
        detach_tgt: Optional[bool] = None,
    ) -> torch.Tensor:
        """Evaluate the unchanged JEPA loss on versioned external records."""
        if not records:
            raise ValueError("external loss requires at least one record")
        payloads = []
        for record in records:
            if self.fd_psc_system is not None:
                payloads.append(self.fd_psc_system.materialize_external_payload(record))
            elif isinstance(record, Mapping):
                payloads.append(record)
            else:
                payloads.append(record.payload)
        source_detach = (not self.finetune_encoder) if detach_src is None else bool(detach_src)
        target_detach = (
            True if not self.finetune_encoder else bool(getattr(self.wm, "stop_grad", True))
        ) if detach_tgt is None else bool(detach_tgt)
        losses = [
            self._prediction_loss(
                self._external_payload_to_z(payload),
                detach_src=source_detach,
                detach_tgt=target_detach,
            )
            for payload in payloads
        ]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def evaluate_external_records(self, records: Sequence[Any]) -> float:
        return float(self.external_loss_tensor(records).detach())

    def _select_predictor_params(self):
        predictor = self.wm.predictor
        if self.last_layer_only:
            params = list(predictor.transformer.layers[-1].parameters())
            params += list(predictor.transformer.norm.parameters())
            log.info(
                "AdaJEPA predictor adaptation restricted to last transformer layer (%d tensors).",
                len(params),
            )
            return params
        return list(predictor.parameters())

    def _select_encoder_params(self):
        """Mirror train-time freezing: never adapt a frozen backbone."""
        encoder = self.wm.encoder
        if hasattr(encoder, "base_model"):
            if hasattr(encoder, "projector") and getattr(encoder, "projector_name", None) in (
                "channel",
                "global",
            ):
                params = list(encoder.projector.parameters())
            else:
                params = [
                    p for name, p in encoder.named_parameters()
                    if not name.startswith("base_model.")
                ]
            log.info("AdaJEPA encoder adaptation uses non-backbone params (%d tensors).", len(params))
            return params
        if self.encoder_last_layer_only:
            children = list(encoder.named_children())
            if children:
                name, module = children[-1]
                params = list(module.parameters())
                log.info(
                    "AdaJEPA encoder adaptation restricted to last submodule '%s' (%d tensors).",
                    name, len(params),
                )
                return params
        params = list(encoder.parameters())
        log.info("AdaJEPA encoder adaptation uses full encoder params (%d tensors).", len(params))
        return params

    @staticmethod
    def _set_requires_grad(module, ada_params, enabled: bool):
        """Freeze all params of `module`; if enabled, re-enable the adapted subset."""
        for p in module.parameters():
            p.requires_grad_(False)
        if enabled:
            for p in ada_params:
                p.requires_grad_(True)

    def _make_optimizer(self):
        if self.fd_psc_system is not None:
            predictor_params, encoder_params = self.fd_psc_system.online_parameter_groups(
                include_encoder=self.finetune_encoder
            )
            if not predictor_params:
                raise RuntimeError("FD-PSC found no trainable episodic predictor adapters")
            param_groups = [{"params": predictor_params, "lr": self.lr}]
            if self.finetune_encoder and encoder_params:
                param_groups.append({"params": encoder_params, "lr": self.encoder_lr})
                log.info(
                    "FD-AdaJEPA optimizer: predictor_lr=%.2e  encoder_lr=%.2e",
                    self.lr,
                    self.encoder_lr,
                )
            optimizers = {
                "adam": torch.optim.Adam,
                "adamw": torch.optim.AdamW,
                "sgd": torch.optim.SGD,
            }
            if self.optimizer_name not in optimizers:
                raise ValueError(f"Unknown AdaJEPA optimizer: {self.optimizer_name!r}")
            return optimizers[self.optimizer_name](param_groups)
        param_groups = [{"params": self._ada_predictor_params, "lr": self.lr}]
        if self.finetune_encoder:
            param_groups.append({"params": self._ada_encoder_params, "lr": self.encoder_lr})
            log.info("AdaJEPA optimizer: predictor_lr=%.2e  encoder_lr=%.2e", self.lr, self.encoder_lr)
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        if self.optimizer_name not in optimizers:
            raise ValueError(f"Unknown AdaJEPA optimizer: {self.optimizer_name!r}")
        return optimizers[self.optimizer_name](param_groups)

    def _prepare_segment(self, obs, act):
        """Move one (obs, act) segment to device and pad a dummy action for the
        last frame so encode() sees T+1 (frame, action) pairs. The dummy action
        never enters the prediction loss: action dims are excluded from the loss
        and the last frame never appears in a source window."""
        obs = move_to_device({k: v.clone() for k, v in obs.items()}, self.device)
        act = act.to(self.device)
        act = torch.cat([act, torch.zeros_like(act[:, :1])], dim=1)
        return obs, act

    @staticmethod
    def _merge_segments(obs_seqs, act_seqs):
        """Concatenate contiguous segments; the last frame of segment i equals
        the first frame of segment i+1, so the duplicate frame is dropped."""
        obs = {k: v.clone() for k, v in obs_seqs[0].items()}
        act = act_seqs[0]
        for obs_i, act_i in zip(obs_seqs[1:], act_seqs[1:]):
            for k in obs:
                obs[k] = torch.cat([obs[k], obs_i[k][:, 1:]], dim=1)
            act = torch.cat([act, act_i], dim=1)
        return [obs], [act]

    def _prediction_loss(
        self,
        z: torch.Tensor,
        detach_src: bool = True,
        detach_tgt: bool = True,
    ) -> torch.Tensor:
        """Sliding-window 1-step MSE on obs tokens (visual+proprio, no action).

        z: (b, T+1, p, d) embeddings of T+1 frames.
        """
        T = z.shape[1] - 1
        if T < 1:
            return torch.tensor(0.0, device=self.device)
        window = min(self.wm.num_hist, T)
        losses = []
        for t in range(T - window + 1):
            z_src = z[:, t : t + window]
            z_tgt = z[:, t + 1 : t + 1 + window]
            if detach_src:
                z_src = z_src.detach()
            if detach_tgt:
                z_tgt = z_tgt.detach()
            z_pred = self.wm.predict(z_src)
            if self.wm.concat_dim == 0:
                loss = self.criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :])
            else:
                drop = self.wm.action_dim
                loss = self.criterion(z_pred[:, :, :, :-drop], z_tgt[:, :, :, :-drop])
            losses.append(loss)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _prediction_residual_descriptor(self, z: torch.Tensor) -> torch.Tensor:
        """Return a fixed-shape descriptor of the JEPA prediction residual.

        This uses exactly the same sliding one-step predictor windows and
        observation-only mask as :meth:`_prediction_loss`.  The descriptor
        retains two complementary residual marginals instead of collapsing to
        scalar MSE:

        * signed mean and RMS for every ``(history position, feature)`` after
          pooling batch, sliding-window offset, and observation-token axes;
        * signed mean and RMS for every ``(history position, observation
          token)`` after pooling batch, sliding-window offset, and feature
          axes.

        Components are concatenated in the order above and flattened in
        row-major order.  Pooling over batch and offset makes the shape
        independent of trajectory length, while retaining temporal,
        token/spatial, and feature residual patterns.  The result is detached
        float32 CPU data; normalization is deliberately deferred to the
        exception prototype update, where zero residuals have an explicit
        unavailable state.

        FD-PSC calls this only with every adapter disabled and the world model
        in eval mode.  A replayable window must contain the configured history
        length so descriptors remain shape-consistent across episodes.
        """

        if not isinstance(z, torch.Tensor) or z.ndim != 4:
            raise ValueError("JEPA residual descriptor requires z=[batch,time,token,feature]")
        total_steps = int(z.shape[1]) - 1
        history = int(self.wm.num_hist)
        if history <= 0 or total_steps < history:
            raise ValueError(
                "JEPA residual descriptor requires at least num_hist prediction steps"
            )

        residual_windows = []
        for offset in range(total_steps - history + 1):
            source = z[:, offset : offset + history].detach()
            target = z[:, offset + 1 : offset + 1 + history].detach()
            prediction = self.wm.predict(source)
            if self.wm.concat_dim == 0:
                prediction = prediction[:, :, :-1, :]
                target = target[:, :, :-1, :]
            else:
                action_width = int(self.wm.action_dim)
                if action_width <= 0 or action_width >= int(prediction.shape[-1]):
                    raise ValueError(
                        "concat_dim=1 residual descriptors require a valid action feature width"
                    )
                prediction = prediction[..., :-action_width]
                target = target[..., :-action_width]
            if prediction.shape != target.shape or prediction.numel() == 0:
                raise ValueError("JEPA residual descriptor encountered an invalid observation mask")
            residual_windows.append(
                prediction.to(dtype=torch.float32) - target.to(dtype=torch.float32)
            )

        # [offset,batch,history,observation-token,feature]
        residual = torch.stack(residual_windows, dim=0)
        feature_mean = residual.mean(dim=(0, 1, 3))
        feature_rms = residual.square().mean(dim=(0, 1, 3)).sqrt()
        token_mean = residual.mean(dim=(0, 1, 4))
        token_rms = residual.square().mean(dim=(0, 1, 4)).sqrt()
        descriptor = torch.cat(
            (
                feature_mean.reshape(-1),
                feature_rms.reshape(-1),
                token_mean.reshape(-1),
                token_rms.reshape(-1),
            )
        )
        if descriptor.numel() == 0 or not torch.isfinite(descriptor).all():
            raise ValueError("JEPA residual descriptor must be finite and non-empty")
        return descriptor.detach().cpu().contiguous()
