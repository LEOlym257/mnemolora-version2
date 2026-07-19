"""MPC planner with AdaJEPA: test-time adaptation of the world model between replans.

Samples are planned one at a time by the inherited MPC loop, each starting from
the pre-adaptation weights (independent adaptation). The adaptation plugs into
the loop via the _post_env_feedback hook: after each replan, the executed
environment segment is added to a replay buffer and the world model is
finetuned on it (see AdaJEPATrainer).
"""

import csv
import json
import os
import re
import time

import numpy as np
import torch

from .adajepa import AdaJEPATrainer
from .mpc import MPCPlanner, _cuda_sync


class AdaJEPAMPCPlanner(MPCPlanner):
    def __init__(self, adapt=None, **kwargs):
        super().__init__(**kwargs)
        adapt = dict(adapt) if adapt else {}
        self.adajepa_finetune_every = adapt.get("finetune_every", 1)
        self.replay_mode, self.replay_size = self._parse_replay_buffer(adapt.get("replay_buffer", False))
        self.adajepa_trainer = AdaJEPATrainer(
            wm=self.wm,
            lr=adapt.get("lr", 5e-4),
            steps=adapt.get("steps", 1),
            optimizer_name=adapt.get("optimizer", "adam"),
            finetune_encoder=adapt.get("finetune_encoder", True),
            last_layer_only=adapt.get("last_layer_only", True),
            encoder_lr=adapt.get("encoder_lr", None),
            encoder_last_layer_only=adapt.get("encoder_last_layer_only", True),
        )
        self._adajepa_loss_records = []
        self._records = []
        self._sample_idx = 0
        self._obs_buffer, self._act_buffer, self._segment_scores = [], [], []

    @staticmethod
    def _parse_replay_buffer(cfg):
        """false -> keep all segments; 'recent<N>' keeps the last N;
        'hard<N>' keeps the N with the highest prediction loss."""
        if not cfg:
            return None, 0
        m = re.match(r"(recent|hard)(\d+)$", str(cfg))
        if not m:
            raise ValueError(
                f"Invalid adapt.replay_buffer: {cfg!r}; expected false, 'recent<N>', or 'hard<N>'"
            )
        return m.group(1), int(m.group(2))

    def plan(self, obs_0, obs_g, actions=None):
        """Plan one sample at a time, resetting the weights before each sample."""
        n_evals = obs_0["visual"].shape[0]
        all_actions, all_action_lens = [], []
        self._records = []

        for i in range(n_evals):
            self.adajepa_trainer.reset()
            self._sample_idx = i
            self._obs_buffer, self._act_buffer, self._segment_scores = [], [], []
            self.plan_suffix = f"_s{i}"
            actions_i, action_len_i = self._plan_single(
                obs_0_i={k: v[i : i + 1] for k, v in obs_0.items()},
                obs_g_i={k: v[i : i + 1] for k, v in obs_g.items()},
                env_i=_EnvWorkerProxy(self.env, i),
                seed_i=[self.evaluator.seed[i]],
                state_0_i=self.evaluator.state_0[i : i + 1],
                state_g_i=self.evaluator.state_g[i : i + 1],
            )
            all_actions.append(actions_i)
            all_action_lens.append(action_len_i)

        self.plan_suffix = ""
        self.adajepa_trainer.reset()
        self._dump_per_iter_logs(self._records, n_evals)
        self._dump_adajepa_loss_csv()

        # Pad per-sample action sequences to a common length before concatenating;
        # action_len keeps the true number of actions per sample.
        max_T = max(a.shape[1] for a in all_actions)
        padded = []
        for a in all_actions:
            if a.shape[1] < max_T:
                pad = torch.zeros(
                    a.shape[0], max_T - a.shape[1], a.shape[2],
                    device=a.device, dtype=a.dtype,
                )
                a = torch.cat([a, pad], dim=1)
            padded.append(a)
        return torch.cat(padded, dim=0), np.concatenate(all_action_lens)

    def _plan_single(self, obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i):
        """Point the shared evaluator at one sample, run the inherited MPC loop, restore."""
        saved = (
            self.evaluator.env,
            self.evaluator.seed,
            self.evaluator.obs_0,
            self.evaluator.obs_g,
            self.evaluator.state_0,
            self.evaluator.state_g,
        )
        self.evaluator.env = env_i
        self.evaluator.seed = seed_i
        self.evaluator.obs_0 = obs_0_i
        self.evaluator.obs_g = obs_g_i
        self.evaluator.state_0 = state_0_i
        self.evaluator.state_g = state_g_i
        try:
            return super().plan(obs_0_i, obs_g_i)
        finally:
            (
                self.evaluator.env,
                self.evaluator.seed,
                self.evaluator.obs_0,
                self.evaluator.obs_g,
                self.evaluator.state_0,
                self.evaluator.state_g,
            ) = saved

    def _post_env_feedback(self, taken_actions, e_obses):
        """Adapt the world model on the newly executed segment (MPC-loop hook)."""
        obs_seq, act_seq = self._extract_adajepa_data(e_obses, taken_actions, self.iter)
        self._obs_buffer.append(obs_seq)
        self._act_buffer.append(act_seq)
        self._segment_scores = self._trim_buffer(
            self._obs_buffer, self._act_buffer, self._segment_scores
        )
        rec = {
            "sample_idx": self._sample_idx,
            "step": self.iter + 1,
            "success": bool(self.is_success[0]),
            "t_plan_s": self.t_plan_s,
        }
        extra_logs = None
        if (self.iter + 1) % self.adajepa_finetune_every == 0:
            _cuda_sync()
            t0 = time.perf_counter()
            self._segment_scores, pred_loss, step_losses = self._finetune_with_buffer(
                self._obs_buffer, self._act_buffer, self._segment_scores
            )
            _cuda_sync()
            rec["t_adajepa_s"] = time.perf_counter() - t0
            rec["adajepa/pred_loss"] = pred_loss
            self._record_adajepa_losses(self._sample_idx, self.iter + 1, step_losses)
            extra_logs = {"adajepa/pred_loss": pred_loss}
        self._records.append(rec)
        return extra_logs

    def _extract_adajepa_data(self, e_obses, taken_actions, iter_idx: int):
        """Slice this iteration's T*frameskip+1 env frames from the cumulative
        rollout at the model stride: T+1 frames aligned with the T taken actions."""
        fs = self.evaluator.frameskip
        T = taken_actions.shape[1]
        start = iter_idx * T * fs
        obs_chunk = {k: v[:, start : start + T * fs + 1 : fs] for k, v in e_obses.items()}
        return self.preprocessor.transform_obs(obs_chunk), taken_actions.cpu()

    def _trim_buffer(self, obs_buffer, act_buffer, scores):
        """Prune the replay buffer in place. Returns the updated scores list."""
        if not self.replay_mode or len(obs_buffer) <= self.replay_size:
            return scores
        if self.replay_mode == "recent" or not scores:
            del obs_buffer[: -self.replay_size]
            del act_buffer[: -self.replay_size]
            return scores[-self.replay_size :]
        # hard: keep top-N by previous scores; newest (unscored) segments always kept.
        scores = scores + [float("inf")] * (len(obs_buffer) - len(scores))
        top_idx = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.replay_size]
        )
        obs_buffer[:] = [obs_buffer[i] for i in top_idx]
        act_buffer[:] = [act_buffer[i] for i in top_idx]
        return [scores[i] for i in top_idx]

    def _finetune_with_buffer(self, obs_buffer, act_buffer, scores):
        """Finetune on the buffer. Returns (updated_scores, final_pred_loss, step_losses)."""
        # hard-buffer segments are non-contiguous, so they cannot be merged.
        merge = self.replay_mode != "hard"
        step_losses = self.adajepa_trainer.finetune(obs_buffer, act_buffer, merge=merge)
        pred_loss = step_losses[-1] if step_losses else 0.0
        if self.replay_mode == "hard":
            scores = self.adajepa_trainer.score_segments(obs_buffer, act_buffer)
        return scores, pred_loss, step_losses

    def _record_adajepa_losses(self, sample_idx, mpc_iter, step_losses):
        for s, loss in enumerate(step_losses):
            self._adajepa_loss_records.append(
                {
                    "sample_idx": int(sample_idx),
                    "mpc_iter": int(mpc_iter),
                    "adajepa_step": int(s + 1),
                    "loss": float(loss),
                }
            )

    def _dump_adajepa_loss_csv(self):
        if not self._adajepa_loss_records or self.log_filename is None:
            return
        path = os.path.join(os.path.dirname(self.log_filename) or ".", "adajepa_losses.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_idx", "mpc_iter", "adajepa_step", "loss"])
            writer.writeheader()
            for rec in self._adajepa_loss_records:
                writer.writerow(rec)

    def _dump_per_iter_logs(self, records, n_evals):
        if self.log_filename is None or not records:
            return
        out_dir = os.path.dirname(self.log_filename) or "."
        with open(os.path.join(out_dir, "per_iter_logs.json"), "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        # Per-step success rate across samples (success carries forward once reached).
        success_step = {}
        for rec in records:
            if rec["success"] and rec["sample_idx"] not in success_step:
                success_step[rec["sample_idx"]] = rec["step"]
        for step in sorted({rec["step"] for rec in records}):
            n_success = sum(
                1 for i in range(n_evals) if success_step.get(i, np.inf) <= step
            )
            self.dump_logs({"mpc/success_rate": n_success / n_evals, "step": step})


class _EnvWorkerProxy:
    """Single-worker view of a SubprocVectorEnv or SerialVectorEnv, exposing the
    vectorized-env API with batch-1 arrays."""

    def __init__(self, parent_env, env_idx: int):
        from env.serial_vector_env import SerialVectorEnv

        if isinstance(parent_env, SerialVectorEnv):
            self._env = parent_env.envs[env_idx]
            self._mode = "serial"
        else:
            self._worker = parent_env.workers[env_idx]
            self._mode = "subproc"

    def rollout(self, seed, init_state, actions):
        if self._mode == "serial":
            obs, state = self._env.rollout(seed[0], init_state[0], actions[0])
        else:
            self._worker.rollout(seed[0], init_state[0], actions[0])
            obs, state = self._worker.recv()
        return {k: v[np.newaxis] for k, v in obs.items()}, state[np.newaxis]

    def eval_state(self, goal_state, cur_state):
        if self._mode == "serial":
            result = self._env.eval_state(goal_state[0], cur_state[0])
        else:
            result = self._worker.eval_state(goal_state[0], cur_state[0])
        return {k: np.expand_dims(np.asarray(v), 0) for k, v in result.items()}

    def prepare(self, seed, init_state):
        if self._mode == "serial":
            obs, state = self._env.prepare(seed[0], init_state[0])
        else:
            obs, state = self._worker.prepare(seed[0], init_state[0])
        return {k: v[np.newaxis] for k, v in obs.items()}, state[np.newaxis]

    def update_env(self, env_info):
        if self._mode == "serial":
            self._env.update_env(env_info[0])
        else:
            self._worker.update_env(env_info[0])
