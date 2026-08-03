import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from fd_psc.config import FDPSCConfig, FDPSCConfigError, minimal_test_config
from models.vit import ViTPredictor
from planning.adajepa_mpc import AdaJEPAMPCPlanner
from planning.mpc import MPCPlanner


class ConfigValidationTest(unittest.TestCase):
    def test_disabled_configuration_never_requires_external_files(self):
        cfg = FDPSCConfig.from_mapping({"enabled": False})
        cfg.validate(require_files=True)

    def test_minimal_enabled_configuration_is_typed(self):
        cfg = minimal_test_config(episodic_lora={"rank": 3, "alpha": 6})
        self.assertEqual(cfg.episodic_lora.rank, 3)
        self.assertEqual(cfg.episodic_lora.alpha, 6)
        self.assertEqual(len(cfg.identity_hash()), 64)

    def test_invalid_rank_grid_fails_fast(self):
        cfg = minimal_test_config()
        cfg.slow_lora.allowed_ranks = [8, 8, 4]
        with self.assertRaises(FDPSCConfigError):
            cfg.validate(require_files=False)

    def test_nonzero_dropout_with_geometry_is_rejected(self):
        with self.assertRaises(FDPSCConfigError):
            minimal_test_config(
                episodic_lora={"dropout": 0.1},
                gradient_geometry={"enabled": True},
            )

    def test_disabled_gate_requires_explicit_unsafe_ablation(self):
        cfg = minimal_test_config()
        cfg.gates.allow_unsafe_ablation = False
        with self.assertRaises(FDPSCConfigError):
            cfg.validate(require_files=False)


class PredictorCPUCompatibilityTest(unittest.TestCase):
    def test_predictor_constructs_and_runs_on_cpu_without_persistent_mask_key(self):
        torch.manual_seed(11)
        predictor = ViTPredictor(
            num_patches=2,
            num_frames=3,
            dim=8,
            depth=2,
            heads=2,
            mlp_dim=16,
            dim_head=4,
            dropout=0.0,
            emb_dropout=0.0,
        ).cpu().eval()
        before_keys = set(predictor.state_dict())
        mask_names = [name for name, value in predictor.named_buffers() if value.dtype == torch.bool]
        self.assertTrue(mask_names)
        self.assertTrue(all(name not in before_keys for name in mask_names))
        x = torch.randn(2, 6, 8)
        with torch.no_grad():
            actual = predictor(x)
            clone = copy.deepcopy(predictor)
            clone.load_state_dict(predictor.state_dict(), strict=True)
            expected = clone(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class DisabledOfficialPlannerCompatibilityTest(unittest.TestCase):
    def test_official_planner_plan_has_no_fd_psc_side_effects_when_disabled(self):
        class WorldModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor = ViTPredictor(
                    num_patches=2,
                    num_frames=2,
                    dim=8,
                    depth=1,
                    heads=2,
                    mlp_dim=16,
                    dim_head=4,
                    dropout=0.0,
                    emb_dropout=0.0,
                )
                self.encoder = nn.Sequential(nn.Linear(2, 2))

        torch.manual_seed(37)
        wm = WorldModel().eval()
        predictor_input = torch.randn(1, 4, 8)
        with torch.no_grad():
            expected_predictor_output = wm.predictor(predictor_input).clone()
        state_before = copy.deepcopy(wm.state_dict())
        keys_before = tuple(state_before)
        rng_before = torch.get_rng_state().clone()
        evaluator = SimpleNamespace(
            seed=np.asarray([101, 202]),
            state_0=np.zeros((2, 1), dtype=np.float32),
            state_g=np.ones((2, 1), dtype=np.float32),
        )

        def fake_mpc_init(instance, **kwargs):
            instance.wm = kwargs["wm"]
            instance.evaluator = kwargs["evaluator"]
            instance.env = kwargs.get("env")
            instance.log_filename = kwargs.get("log_filename")

        def fake_plan_single(
            instance, obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i
        ):
            del instance, obs_g_i, env_i, seed_i, state_0_i, state_g_i
            marker = int(obs_0_i["visual"].reshape(-1)[0])
            length = marker + 1
            return (
                torch.full((1, length, 2), float(marker)),
                np.asarray([length], dtype=np.float64),
            )

        with mock.patch.object(MPCPlanner, "__init__", new=fake_mpc_init), mock.patch.object(
            AdaJEPAMPCPlanner, "_plan_single", new=fake_plan_single
        ):
            planner = AdaJEPAMPCPlanner(
                wm=wm,
                evaluator=evaluator,
                env=SimpleNamespace(workers=[object(), object()]),
                log_filename=None,
                adapt={"finetune_encoder": False},
                fd_psc={"enabled": False},
            )
            obs_0 = {
                "visual": torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1, 1)
            }
            obs_g = {"visual": torch.zeros_like(obs_0["visual"])}
            actions, action_lens = planner.plan(obs_0, obs_g)

        self.assertIsNone(planner.fd_psc_system)
        self.assertFalse(planner.adajepa_trainer.fd_psc_enabled)
        self.assertEqual(tuple(wm.state_dict()), keys_before)
        self.assertFalse(hasattr(wm, "_fd_psc_encoder_adapter"))
        self.assertEqual(len(wm.encoder._forward_pre_hooks), 0)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        for name, value in state_before.items():
            self.assertTrue(torch.equal(value, wm.state_dict()[name]), name)
        with torch.no_grad():
            actual_predictor_output = wm.predictor(predictor_input)
        torch.testing.assert_close(
            actual_predictor_output, expected_predictor_output, rtol=0, atol=0
        )
        self.assertEqual(tuple(actions.shape), (2, 3, 2))
        self.assertTrue(torch.equal(actions[0, :2], torch.ones(2, 2)))
        self.assertTrue(torch.equal(actions[0, 2], torch.zeros(2)))
        self.assertTrue(torch.equal(actions[1], torch.full((3, 2), 2.0)))
        np.testing.assert_array_equal(action_lens, np.asarray([2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
