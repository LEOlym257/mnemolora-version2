from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from fd_psc.canary import CanaryManifest, CanaryManifestError
from fd_psc.external_data import ExternalDataRegistry, ManifestSchemaError
from fd_psc.preprocess_identity import compute_preprocess_hash
from fd_psc.trainer import FDPSCIntegrationError
from planning.adajepa import AdaJEPATrainer
from preprocessor import Preprocessor
from tests.test_fd_psc_integration import (
    BASE_HASH,
    PREPROCESS_HASH,
    _ToyWorldModel,
    _config,
    _write_canary_manifest,
    _write_fixed_manifest,
)


class PreprocessIdentityTests(unittest.TestCase):
    @staticmethod
    def _preprocessor(offset: float = 0.0) -> Preprocessor:
        return Preprocessor(
            action_mean=torch.tensor([offset, 1.0]),
            action_std=torch.tensor([1.0, 2.0]),
            state_mean=torch.tensor([0.0, 1.0]),
            state_std=torch.tensor([2.0, 3.0]),
            proprio_mean=torch.tensor([4.0, 5.0]),
            proprio_std=torch.tensor([6.0, 7.0]),
            transform=nn.Identity(),
        )

    def test_identity_is_deterministic_and_covers_stats_and_window_schema(self) -> None:
        first = compute_preprocess_hash(
            self._preprocessor(),
            encoder_transform=nn.Identity(),
            frameskip=2,
            num_hist=3,
            num_pred=1,
        )
        second = compute_preprocess_hash(
            self._preprocessor(),
            encoder_transform=nn.Identity(),
            frameskip=2,
            num_hist=3,
            num_pred=1,
        )
        changed_stat = compute_preprocess_hash(
            self._preprocessor(0.5),
            encoder_transform=nn.Identity(),
            frameskip=2,
            num_hist=3,
            num_pred=1,
        )
        changed_window = compute_preprocess_hash(
            self._preprocessor(),
            encoder_transform=nn.Identity(),
            frameskip=1,
            num_hist=3,
            num_pred=1,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed_stat)
        self.assertNotEqual(first, changed_window)

    def test_external_and_canary_manifests_compare_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external, _ = _write_fixed_manifest(root)
            canary = _write_canary_manifest(root)
            ExternalDataRegistry(
                external,
                expected_base_checkpoint_hash=BASE_HASH,
                expected_preprocess_hash=PREPROCESS_HASH,
                require_all_splits=False,
            )
            CanaryManifest.load(
                canary,
                expected_base_checkpoint_hash=BASE_HASH,
                expected_preprocess_hash=PREPROCESS_HASH,
            )
            with self.assertRaisesRegex(
                ManifestSchemaError, "preprocess hash mismatch"
            ):
                ExternalDataRegistry(
                    external,
                    expected_preprocess_hash="c" * 64,
                    require_all_splits=False,
                )
            with self.assertRaisesRegex(
                CanaryManifestError, "preprocess hash mismatch"
            ):
                CanaryManifest.load(
                    canary,
                    expected_preprocess_hash="c" * 64,
                )

    def test_full_protocol_requires_explicit_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, split_paths = _write_fixed_manifest(root)
            model = _ToyWorldModel()
            del model._fd_psc_preprocess_hash
            with self.assertRaisesRegex(
                FDPSCIntegrationError,
                "explicit runtime preprocessing identity",
            ):
                AdaJEPATrainer(
                    wm=model,
                    lr=0.1,
                    steps=1,
                    optimizer_name="sgd",
                    finetune_encoder=True,
                    last_layer_only=False,
                    encoder_lr=0.1,
                    encoder_last_layer_only=False,
                    fd_psc=_config(manifest, split_paths),
                    runtime_output_dir=str(root),
                )


if __name__ == "__main__":
    unittest.main()
