import copy
import unittest

import torch
from torch import nn

from fd_psc.encoder_adapters import FrozenVisualLatent, get_encoder_adapter
from fd_psc.injector import inject_fd_psc_adapters
from fd_psc.lora_layers import DualLoRAConv2d, DualLoRALinear


class MockDinoBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.forbidden_linear = nn.Linear(3, 3)
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward_features(self, x):
        self.calls.add_(1)
        pooled = x.mean(dim=(-2, -1))
        # Four 2x2 patch tokens.
        return {"x_norm_patchtokens": pooled.unsqueeze(1).repeat(1, 4, 1)}


class MockDinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = MockDinoBase()
        self.feature_key = "x_norm_patchtokens"
        self.projector_name = "channel"
        self.projector = nn.Sequential(
            nn.Conv2d(3, 4, 1),
            nn.GELU(),
            nn.Conv2d(4, 4, 1, groups=2),
        )
        self.emb_dim = 4

    def forward(self, x):
        tokens = self.base_model.forward_features(x)[self.feature_key]
        feature_map = tokens.reshape(x.shape[0], 2, 2, 3).permute(0, 3, 1, 2)
        return self.projector(feature_map).flatten(2).transpose(1, 2)


class MockAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.0))

    def forward(self, x):
        return self.to_out(self.to_qkv(x)[..., : x.shape[-1]])


class MockFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(dim * 2, dim), nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


class MockTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([MockAttention(dim), MockFeedForward(dim)])])

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x


class MockPredictor(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.transformer = MockTransformer(dim)

    def forward(self, x):
        return self.transformer(x)


class DummyRepeatActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # deliberately declared but unreachable

    def forward(self, x):
        return x


class MockWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MockDinoEncoder()
        self.predictor = MockPredictor()
        self.action_encoder = DummyRepeatActionEncoder()
        self.proprio_encoder = nn.Identity()
        self.encoder_transform = nn.Identity()

    def forward(self, obs, act=None):
        return self.predictor(self.encoder(obs["visual"]))


class MockDinoNoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = MockDinoBase()
        self.feature_key = "x_norm_patchtokens"
        self.projector_name = "none"
        self.emb_dim = 3

    def forward(self, x):
        return self.base_model.forward_features(x)[self.feature_key]


class NoHeadWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MockDinoNoHead()
        self.predictor = nn.Sequential(nn.Linear(3, 3))
        self.action_encoder = nn.Identity()
        self.proprio_encoder = nn.Identity()
        self.encoder_transform = nn.Identity()

    def forward(self, obs, act=None):
        return self.predictor(self.encoder(obs["visual"]))


def config(enabled=True):
    return {
        "enabled": enabled,
        "seed": 19,
        "target_modules": {
            "predictor_linear": True,
            "post_backbone_projection_linear": True,
            "action_encoder_linear": False,
            "proprio_encoder_linear": False,
            "fail_on_empty_predictor_targets": True,
            "fail_on_empty_projection_targets": False,
            "require_projection_targets_if_head_exists": True,
        },
        "episodic_lora": {"rank": 8, "alpha": 16.0, "dropout": 0.0},
        "conv_lora": {"enabled": True},
    }


class InjectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.sample = {"obs": {"visual": torch.randn(2, 3, 8, 8)}, "act": None}

    def test_disabled_is_strict_noop(self):
        wm = MockWM()
        keys = tuple(wm.state_dict())
        modules = tuple((name, id(module)) for name, module in wm.named_modules())
        result = inject_fd_psc_adapters(wm, config(False), schema_sample=self.sample)
        self.assertFalse(result.manifest.enabled)
        self.assertFalse(result.adapters)
        self.assertEqual(keys, tuple(wm.state_dict()))
        self.assertEqual(modules, tuple((name, id(module)) for name, module in wm.named_modules()))
        self.assertFalse(hasattr(wm, "_fd_psc_encoder_adapter"))

    def test_legacy_base_checkpoint_strict_load_then_fresh_fd_psc_init(self):
        """The official load order is base checkpoint first, adapters second."""
        torch.manual_seed(29)
        source = MockWM().eval()
        sample = {"obs": {"visual": torch.randn(2, 3, 8, 8)}, "act": None}
        legacy_state = copy.deepcopy(source.state_dict())
        expected = source(**sample).detach().clone()

        restored = MockWM().eval()
        incompatible = restored.load_state_dict(legacy_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(tuple(restored.state_dict()), tuple(legacy_state))
        self.assertTrue(torch.equal(expected, restored(**sample)))

        # A legacy checkpoint has no FD-PSC sidecar.  Injection therefore
        # starts a new zero-function memory state instead of treating the
        # missing sidecar as a migration failure.
        injected = inject_fd_psc_adapters(restored, config(), schema_sample=sample)
        injected.begin_episode(episode_id=0)
        self.assertTrue(injected.adapters)
        for adapter in injected.adapters.values():
            self.assertEqual(int(adapter.slow_A.shape[0]), 0)
            self.assertIsNotNone(adapter.pilot_B)
            self.assertTrue(torch.count_nonzero(adapter.pilot_B) == 0)
        self.assertTrue(torch.equal(expected, restored(**sample)))

    def test_manifest_full_depth_groupwise_and_base_exclusion(self):
        wm = MockWM().eval()
        expected = wm(**self.sample).detach().clone()
        rng_before = torch.get_rng_state().clone()
        calls_before = wm.encoder.base_model.calls.clone()
        result = inject_fd_psc_adapters(wm, config(), schema_sample=self.sample)
        # Dry-run buffer/RNG changes are restored; the expected forward above is
        # the only lasting base call.
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertTrue(torch.equal(calls_before, wm.encoder.base_model.calls))

        expected_ids = {
            "predictor.transformer.layers.0.0.to_qkv",
            "predictor.transformer.layers.0.0.to_out.0",
            "predictor.transformer.layers.0.1.net.1",
            "predictor.transformer.layers.0.1.net.4",
            "encoder.projector.0::group=0",
            "encoder.projector.2::group=0",
            "encoder.projector.2::group=1",
        }
        self.assertEqual(set(result.adapters), expected_ids)
        self.assertFalse(any("base_model" in item for item in result.adapters))
        self.assertIsInstance(wm.predictor.transformer.layers[0][0].to_qkv, DualLoRALinear)
        self.assertIsInstance(wm.encoder.projector[0], DualLoRAConv2d)
        self.assertIsInstance(wm.encoder.projector[2], DualLoRAConv2d)
        self.assertTrue(torch.equal(expected, wm(**self.sample)))
        self.assertFalse(any(p.requires_grad for p in wm.encoder.base_model.parameters()))

        manifest = result.manifest
        self.assertEqual(manifest.hash, manifest.manifest_hash)
        self.assertEqual(len(manifest.hash), 64)
        qkv = manifest.by_logical_id()["predictor.transformer.layers.0.0.to_qkv"]
        self.assertEqual(qkv.role, "attention_qkv")
        self.assertTrue(qkv.active_in_forward)
        final = manifest.by_logical_id()["encoder.projector.2::group=1"]
        self.assertTrue(final.final_projection)
        self.assertEqual(final.groups, 2)
        self.assertEqual(final.actual_rank, 2)
        self.assertEqual(len(result.predictor_parameters()), 8)
        self.assertEqual(len(result.encoder_parameters()), 6)

    def test_dino_explicit_latent_protocol_preserves_token_grid_and_time(self):
        wm = MockWM().eval()
        adapter = get_encoder_adapter(wm)
        visual = torch.randn(2, 3, 3, 8, 8)
        expected = wm.encoder(visual.reshape(6, 3, 8, 8)).reshape(2, 3, 4, 4)
        frozen = adapter.extract_frozen_visual_latent({"visual": visual})
        self.assertIsInstance(frozen, FrozenVisualLatent)
        self.assertEqual(frozen.layout, "tokens")
        self.assertEqual(frozen.metadata["token_grid"], (2, 2))
        self.assertTrue(torch.allclose(expected, adapter.project_visual_latent(frozen)))
        payload = FrozenVisualLatent.from_payload(frozen.to_payload())
        self.assertTrue(torch.equal(payload.tensor, frozen.tensor))

    def test_duplicate_injection_and_enabled_unreachable_optional_target_fail(self):
        wm = MockWM()
        inject_fd_psc_adapters(wm, config())
        with self.assertRaisesRegex(RuntimeError, "before adapter injection|duplicate"):
            inject_fd_psc_adapters(wm, config())

        wm = MockWM()
        cfg = config()
        cfg["target_modules"]["action_encoder_linear"] = True
        with self.assertRaisesRegex(RuntimeError, "action_encoder_linear"):
            inject_fd_psc_adapters(wm, cfg)

    def test_head_without_linear_or_conv_fails_fast(self):
        wm = MockWM()
        wm.encoder.projector = nn.Sequential(nn.ReLU())
        with self.assertRaisesRegex(RuntimeError, "projection head"):
            inject_fd_psc_adapters(wm, config())

    def test_episode_initialization_uses_stable_per_logical_generators(self):
        first = inject_fd_psc_adapters(MockWM(), config())
        second = inject_fd_psc_adapters(MockWM(), config())
        rng = torch.get_rng_state().clone()
        first.begin_episode(episode_id=12)
        second.begin_episode(episode_id=12)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for logical_id in sorted(first.adapters):
            self.assertTrue(
                torch.equal(
                    first.adapters[logical_id].pilot_A,
                    second.adapters[logical_id].pilot_A,
                )
            )

    def test_no_projection_head_is_explicit_identity_not_applicable(self):
        wm = NoHeadWM().eval()
        sample = {"obs": {"visual": torch.randn(2, 3, 8, 8)}, "act": None}
        expected = wm(**sample).detach().clone()
        result = inject_fd_psc_adapters(wm, config(), schema_sample=sample)
        self.assertEqual(
            result.manifest.metadata["projection_status"],
            "not_applicable_no_projection_head",
        )
        self.assertFalse(
            any(e.module_group == "encoder_projection" for e in result.manifest.entries)
        )
        self.assertTrue(torch.equal(expected, wm(**sample)))
        latent = result.encoder_adapter.extract_frozen_visual_latent(sample["obs"])
        self.assertTrue(
            torch.equal(
                wm.encoder.base_model.forward_features(sample["obs"]["visual"])[
                    "x_norm_patchtokens"
                ],
                result.encoder_adapter.project_visual_latent(latent),
            )
        )


if __name__ == "__main__":
    unittest.main()
