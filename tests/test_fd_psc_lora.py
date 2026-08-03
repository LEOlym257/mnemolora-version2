import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from fd_psc.lora_layers import CanonicalFactors, DualLoRAConv2d, DualLoRALinear


class DualLinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_pilot_actual_rank_scaling_and_base_freeze(self):
        base = nn.Linear(3, 2, bias=True)
        x = torch.randn(4, 3)
        expected = base(x).detach().clone()
        layer = DualLoRALinear(base, rank=8, alpha=12)
        self.assertEqual(layer.pilot_actual_rank, 2)
        self.assertEqual(layer.pilot_scaling, 6.0)
        self.assertTrue(torch.equal(expected, layer(x)))
        self.assertFalse(layer.weight.requires_grad)
        self.assertFalse(layer.bias.requires_grad)
        self.assertEqual(set(layer.trainable_episode_parameters()), {layer.pilot_A, layer.pilot_B})

    def test_slow_exception_pilot_canonical_composition(self):
        base = nn.Linear(5, 4, bias=False)
        layer = DualLoRALinear(base, rank=3, alpha=9)
        x = torch.randn(2, 6, 5)
        slow = CanonicalFactors(torch.randn(4, 2), torch.randn(2, 5))
        exception = CanonicalFactors(torch.randn(4, 1), torch.randn(1, 5))
        layer.replace_slow_adapter(slow)
        layer.set_active_exception(exception, adapter_id="x7")
        with torch.no_grad():
            layer.pilot_B.normal_()
        factors = layer.get_effective_factors()
        expected = base(x) + F.linear(x, factors.B @ factors.A)
        self.assertTrue(torch.allclose(layer(x), expected, atol=2e-6, rtol=2e-6))
        self.assertEqual(factors.rank, 2 + 1 + 3)
        self.assertEqual(layer.active_exception_id, "x7")

    def test_centered_switch_is_continuous_and_uses_its_actual_rank(self):
        base = nn.Linear(6, 5)
        layer = DualLoRALinear(base, rank=4, alpha=12)
        with torch.no_grad():
            layer.pilot_B.normal_()
        x = torch.randn(3, 6)
        before = layer(x).detach().clone()
        b0, a0 = torch.randn(5, 2), torch.randn(2, 6)
        layer.activate_centered_branch(b0, a0, alpha=10)
        self.assertTrue(torch.equal(before, layer(x)))
        self.assertTrue(layer.pilot_frozen)
        self.assertEqual(layer.pilot_scaling, 3.0)
        self.assertEqual(layer.centered_scaling, 5.0)
        with torch.no_grad():
            layer.center_A.add_(0.05)
            layer.center_B.sub_(0.02)
        factors = layer.get_episodic_factors()
        delta = F.linear(x, factors.B @ factors.A)
        self.assertTrue(torch.allclose(layer(x) - base(x), delta, atol=2e-5, rtol=2e-5))
        self.assertEqual(factors.rank, 4 + 2 + 2)

    def test_disable_reset_and_custom_state_round_trip(self):
        base = nn.Linear(4, 3)
        layer = DualLoRALinear(base, rank=2, alpha=4)
        x = torch.randn(2, 4)
        with torch.no_grad():
            layer.pilot_B.normal_()
        layer.replace_slow_adapter(torch.randn(3, 1), torch.randn(1, 4))
        layer.set_active_exception(torch.randn(3, 1), torch.randn(1, 4), adapter_id="e")
        expected = layer(x).detach().clone()
        state = layer.adapter_state_dict()

        restored = DualLoRALinear(copy.deepcopy(base), rank=2, alpha=4)
        restored.load_adapter_state_dict(state)
        self.assertTrue(torch.equal(expected, restored(x)))

        layer.disable_all_adapters()
        self.assertTrue(torch.equal(base(x), layer(x)))
        layer.enable_all_adapters()
        layer.reset_episode()
        self.assertEqual(layer.get_exception_factors().rank, 0)
        self.assertEqual(layer.get_episodic_factors().rank, 2)
        self.assertTrue(
            torch.allclose(
                base(x) + F.linear(F.linear(x, layer.slow_A), layer.slow_B),
                layer(x),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_regular_module_state_dict_loads_dynamic_ranks_and_centered_state(self):
        base = nn.Linear(5, 4)
        layer = DualLoRALinear(base, rank=3, alpha=9)
        layer.replace_slow_adapter(torch.randn(4, 2), torch.randn(2, 5))
        layer.set_active_exception(
            torch.randn(4, 1), torch.randn(1, 5), adapter_id="route-2"
        )
        with torch.no_grad():
            layer.pilot_B.normal_()
        layer.activate_centered_branch(torch.randn(4, 2), torch.randn(2, 5), alpha=7)
        with torch.no_grad():
            layer.center_A.add_(0.1)
        x = torch.randn(3, 5)
        expected = layer(x).detach().clone()
        payload = copy.deepcopy(layer.state_dict())

        restored = DualLoRALinear(copy.deepcopy(base), rank=3, alpha=9)
        restored.load_state_dict(payload)
        self.assertTrue(torch.equal(expected, restored(x)))
        self.assertTrue(restored.centered_active)
        self.assertTrue(restored.pilot_frozen)
        self.assertEqual(restored.active_exception_id, "route-2")
        self.assertEqual(restored.get_slow_factors().rank, 2)


def explicit_conv_delta(layer, x):
    kernel = layer.materialize_effective_delta()
    base = layer.base_layer
    if base.padding_mode != "zeros":
        x = F.pad(x, base._reversed_padding_repeated_twice, mode=base.padding_mode)
        padding = (0, 0)
    else:
        padding = base.padding
    return F.conv2d(x, kernel, None, base.stride, padding, base.dilation, base.groups)


class DualConvTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def _exercise(self, conv, shape):
        x = torch.randn(*shape)
        expected_base = conv(x).detach().clone()
        layer = DualLoRAConv2d(conv, rank=8, alpha=16)
        self.assertTrue(torch.equal(expected_base, layer(x)))
        for group in layer.logical_groups:
            self.assertLessEqual(group.pilot_actual_rank, min(group.in_features, group.out_features))
            with torch.no_grad():
                group.pilot_B.normal_()
        expected = expected_base + explicit_conv_delta(layer, x)
        self.assertTrue(torch.allclose(layer(x), expected, atol=3e-5, rtol=3e-5))
        return layer, x, expected_base

    def test_factorized_forward_preserves_conv_geometry(self):
        cases = [
            (nn.Conv2d(4, 6, 3, stride=2, padding=1, bias=True), (2, 4, 13, 11)),
            (
                nn.Conv2d(4, 8, 3, padding=2, dilation=2, groups=2, bias=False),
                (2, 4, 9, 10),
            ),
            (
                nn.Conv2d(4, 6, 3, padding=1, groups=2, padding_mode="reflect"),
                (2, 4, 10, 12),
            ),
            (nn.Conv2d(3, 5, 3, padding="same"), (2, 3, 8, 7)),
        ]
        for conv, shape in cases:
            with self.subTest(conv=conv):
                self._exercise(conv, shape)

    def test_groupwise_state_is_independent(self):
        conv = nn.Conv2d(4, 8, 3, padding=1, groups=2, bias=False)
        layer = DualLoRAConv2d(conv, rank=3, alpha=6, logical_id="encoder.projector.conv")
        self.assertEqual(
            [g.logical_id for g in layer.logical_groups],
            ["encoder.projector.conv::group=0", "encoder.projector.conv::group=1"],
        )
        x = torch.randn(1, 4, 7, 7)
        base = conv(x)
        with torch.no_grad():
            layer.logical_groups[0].pilot_B.normal_()
        delta = layer(x) - base
        self.assertGreater(float(delta[:, :4].abs().sum()), 0)
        self.assertEqual(float(delta[:, 4:].abs().sum()), 0.0)
        self.assertIsNot(layer.logical_groups[0].pilot_A, layer.logical_groups[1].pilot_A)

    def test_conv_centered_continuity_disable_and_no_materialize_in_forward(self):
        conv = nn.Conv2d(2, 3, 3, padding=1)
        layer = DualLoRAConv2d(conv, rank=2, alpha=4)
        group = layer.logical_groups[0]
        with torch.no_grad():
            group.pilot_B.normal_()
        x = torch.randn(2, 2, 6, 5)
        before = layer(x).detach().clone()
        group.activate_centered_branch(torch.randn(3, 1), torch.randn(1, 18))
        self.assertTrue(torch.equal(before, layer(x)))

        original = group.materialize_effective_delta
        group.materialize_effective_delta = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("production forward materialized BA")
        )
        layer(x)
        group.materialize_effective_delta = original
        layer.disable_all_adapters()
        self.assertTrue(torch.equal(conv(x), layer(x)))

    def test_grouped_conv_state_dict_round_trip_keeps_groups_separate(self):
        conv = nn.Conv2d(4, 8, 3, padding=1, groups=2)
        layer = DualLoRAConv2d(conv, rank=3, alpha=6, logical_id="proj")
        group0, group1 = layer.logical_groups
        group0.replace_slow_adapter(torch.randn(4, 1), torch.randn(1, 18))
        group1.replace_slow_adapter(torch.randn(4, 2), torch.randn(2, 18))
        group0.set_active_exception(torch.randn(4, 1), torch.randn(1, 18), adapter_id="e")
        with torch.no_grad():
            group0.pilot_B.normal_()
            group1.pilot_B.normal_()
        x = torch.randn(2, 4, 7, 6)
        expected = layer(x).detach().clone()

        restored = DualLoRAConv2d(copy.deepcopy(conv), rank=3, alpha=6, logical_id="proj")
        restored.load_state_dict(copy.deepcopy(layer.state_dict()))
        self.assertTrue(torch.equal(expected, restored(x)))
        self.assertEqual(restored.logical_groups[0].get_slow_factors().rank, 1)
        self.assertEqual(restored.logical_groups[1].get_slow_factors().rank, 2)
        self.assertEqual(restored.logical_groups[0].active_exception_id, "e")
        self.assertIsNone(restored.logical_groups[1].active_exception_id)


if __name__ == "__main__":
    unittest.main()
