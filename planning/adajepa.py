import logging

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
        self._ada_predictor_params = self._select_predictor_params()
        self._ada_encoder_params = self._select_encoder_params() if self.finetune_encoder else []
        self._snapshot = self._take_snapshot()

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
        for tensor, saved in self._snapshot:
            tensor.copy_(saved)

    def finetune(self, obs_seqs: list, act_seqs: list, merge: bool = True) -> list:
        """Run `steps` optimization steps on the given trajectory segments.

        merge=True concatenates temporally contiguous segments into one long
        sequence; use merge=False for non-contiguous segments.
        Returns the per-step prediction losses.
        """
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

    @torch.no_grad()
    def score_segments(self, obs_seqs: list, act_seqs: list) -> list:
        """Prediction loss per segment under the current weights (higher = harder)."""
        return [
            float(self._prediction_loss(self.wm.encode(*self._prepare_segment(o, a))))
            for o, a in zip(obs_seqs, act_seqs)
        ]

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
