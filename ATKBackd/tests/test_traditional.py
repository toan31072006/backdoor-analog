"""Small CPU-only operator checks; no training or external dataset needed.

Run directly with ``python -B ATKBackd/tests/test_traditional.py`` or via pytest.
"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack.traditional import (BadNetsTrigger, BlendedTrigger, WaNetTrigger,
                                build_traditional_trigger)


CLASSES = (BadNetsTrigger, BlendedTrigger, WaNetTrigger)
SHAPE = (3, 114, 10)


class TraditionalTriggerTests(unittest.TestCase):
    def setUp(self):
        self.csi = np.random.default_rng(7).uniform(0.0, 1.0, SHAPE).astype(np.float32)

    def test_zero_dose_and_relevant_zero_strength_are_exact_identity(self):
        for cls in CLASSES:
            with self.subTest(trigger=cls.__name__):
                trigger = cls(seed=42)
                cases = [(0.0, 0.185)]
                if cls is not BadNetsTrigger:
                    cases.append((1.0, 0.0))
                for dose, eps in cases:
                    result = trigger.inject(self.csi, dose=dose, eps=eps)
                    np.testing.assert_array_equal(result, self.csi)
                    self.assertFalse(np.shares_memory(result, self.csi))
                    self.assertEqual(result.dtype, np.float32)

    def test_shape_dtype_range_and_input_preservation(self):
        original = self.csi.copy()
        for cls in CLASSES:
            with self.subTest(trigger=cls.__name__):
                trigger = cls(seed=42)
                for dose, eps in ((0.2, 0.185), (1.0, 0.185), (3.0, 2.0)):
                    result = trigger.inject(self.csi.astype(np.float64), dose=dose, eps=eps)
                    self.assertEqual(result.shape, SHAPE)
                    self.assertEqual(result.dtype, np.float32)
                    self.assertTrue(np.isfinite(result).all())
                    self.assertGreaterEqual(float(result.min()), 0.0)
                    self.assertLessEqual(float(result.max()), 1.0)
                    self.assertFalse(np.array_equal(result, self.csi))
                np.testing.assert_array_equal(self.csi, original)

    def test_seeded_pattern_is_fixed_across_samples_and_calls(self):
        for cls in CLASSES:
            with self.subTest(trigger=cls.__name__):
                first, same, other = cls(seed=42), cls(seed=42), cls(seed=43)
                expected = first.inject(self.csi, dose=1.0, eps=0.185)
                first.inject(1.0 - self.csi, dose=0.5, eps=0.185)
                np.testing.assert_array_equal(first.inject(self.csi, 1.0, 0.185), expected)
                np.testing.assert_array_equal(same.inject(self.csi, 1.0, 0.185), expected)
                if cls is BadNetsTrigger:
                    # A white patch has no random pattern; seed controls the
                    # dataset's poison selection, not this fixed operator.
                    np.testing.assert_array_equal(other.inject(self.csi, 1.0, 0.185), expected)
                else:
                    self.assertFalse(np.array_equal(other.inject(self.csi, 1.0, 0.185), expected))

    def test_no_event_metadata_or_deferred_injection(self):
        for cls in CLASSES:
            with self.subTest(trigger=cls.__name__):
                trigger = cls(seed=42)
                self.assertIs(trigger.requires_deferred_injection, False)
                self.assertIs(trigger.requires_event_metadata, False)
                self.assertIn('CSI-adapted', trigger.baseline_name)
                trigger.inject(self.csi, dose=1.0, eps=0.185)

    def test_patch_stays_inside_configured_antenna_frequency_packet_support(self):
        trigger = BadNetsTrigger(seed=42, patch_subcarriers=5, patch_packets=2,
                                 patch_start=(10, 4), antennas=(1,))
        result = trigger.inject(self.csi, dose=1.0, eps=0.185)
        self.assertEqual(int(trigger.mask.sum()), 10)
        np.testing.assert_array_equal(result[~trigger.mask], self.csi[~trigger.mask])
        full = trigger.inject(self.csi, dose=1.0, eps=1.0)
        np.testing.assert_array_equal(full[trigger.mask], trigger.pattern[trigger.mask])
        np.testing.assert_array_equal(full[~trigger.mask], self.csi[~trigger.mask])

    def test_patch_and_blend_dose_response_and_distinct_strengths(self):
        for cls in (BadNetsTrigger, BlendedTrigger):
            with self.subTest(trigger=cls.__name__):
                trigger = cls(seed=42)
                low = trigger.inject(self.csi, dose=0.25, eps=0.185)
                high = trigger.inject(self.csi, dose=1.0, eps=0.185)
                low_delta, high_delta = low - self.csi, high - self.csi
                np.testing.assert_allclose(low_delta, 0.25 * high_delta, rtol=2e-4, atol=1e-7)
                self.assertGreater(float(np.linalg.norm(high_delta)), float(np.linalg.norm(low_delta)))
                limit = 1.0 if cls is BadNetsTrigger else 0.185
                self.assertLessEqual(float(np.max(np.abs(high_delta))), limit + 1e-7)

    def test_blended_uses_published_convex_combination_in_unit_range(self):
        trigger = BlendedTrigger(seed=42)
        dose, eps = 0.75, 0.185
        alpha = dose * eps
        expected = (1.0 - alpha) * self.csi + alpha * trigger.pattern
        np.testing.assert_array_equal(trigger.inject(self.csi, dose, eps), expected)
        np.testing.assert_array_equal(trigger.inject(self.csi, 1.0, 1.0), trigger.pattern)

    def test_wanet_grid_dose_response_and_geometric_bound(self):
        trigger = WaNetTrigger(seed=42, strength=0.5)
        low = trigger.sampling_grid(dose=0.25, eps=0.185)
        high = trigger.sampling_grid(dose=1.0, eps=0.185)
        identity = trigger.identity_grid
        np.testing.assert_array_equal(trigger.sampling_grid(0.0, 0.185), identity)
        self.assertEqual(high.shape, (114, 10, 2))
        self.assertLessEqual(float(np.max(np.abs(high - identity))), 0.185 * 0.5 + 1e-12)
        self.assertGreater(float(np.linalg.norm(high - identity)), float(np.linalg.norm(low - identity)))
        # Six frequency rows and one packet put the grid at least 0.0925
        # normalized units from either edge, so clipping cannot affect it.
        interior = np.s_[6:-6, 1:-1, :]
        np.testing.assert_allclose((low - identity)[interior],
                                   0.25 * (high - identity)[interior], rtol=1e-12, atol=1e-12)

    def test_wanet_preserves_antenna_axis_and_channel_relationships(self):
        trigger = WaNetTrigger(seed=42)
        amplitude = np.broadcast_to(np.array([0.1, 0.4, 0.9], dtype=np.float32)[:, None, None], SHAPE)
        np.testing.assert_array_equal(trigger.inject(amplitude, dose=1.0, eps=0.185), amplitude)
        signal = np.linspace(0.1, 0.8, 1140).reshape(114, 10).astype(np.float32)
        related = np.stack((signal, 0.5 * signal, 0.25 * signal))
        result = trigger.inject(related, dose=1.0, eps=0.185)
        np.testing.assert_allclose(result[1], 0.5 * result[0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(result[2], 0.25 * result[0], rtol=1e-6, atol=1e-7)

    def test_wanet_dose_changes_smooth_csi_and_zero_strength_is_identity(self):
        trigger = WaNetTrigger(seed=42)
        signal = np.linspace(0.1, 0.8, 1140).reshape(114, 10).astype(np.float32)
        smooth_csi = np.broadcast_to(signal, SHAPE)
        low = trigger.inject(smooth_csi, dose=0.25, eps=0.185)
        high = trigger.inject(smooth_csi, dose=1.0, eps=0.185)
        self.assertGreater(float(np.linalg.norm(high - smooth_csi)),
                           float(np.linalg.norm(low - smooth_csi)))
        np.testing.assert_array_equal(WaNetTrigger(strength=0).inject(self.csi, 1.0, 0.185), self.csi)

    def test_wanet_clean_label_cover_is_sample_seeded_and_zero_dose_identity(self):
        trigger = WaNetTrigger(seed=42)
        np.testing.assert_array_equal(trigger.noise_inject(self.csi, seed=17, dose=0.0, eps=0.185),
                                      self.csi)
        np.testing.assert_array_equal(trigger.noise_inject(self.csi, seed=17, dose=1.0, eps=0.0),
                                      self.csi)
        expected = trigger.noise_inject(self.csi, seed=17, eps=0.185)
        trigger.noise_inject(1.0 - self.csi, seed=31, eps=0.185)
        np.testing.assert_array_equal(trigger.noise_inject(self.csi, seed=17, eps=0.185), expected)
        np.testing.assert_array_equal(WaNetTrigger(seed=42).noise_inject(self.csi, seed=17, eps=0.185),
                                      expected)
        self.assertFalse(np.array_equal(trigger.noise_inject(self.csi, seed=18, eps=0.185), expected))
        self.assertFalse(np.array_equal(trigger.inject(self.csi, 1.0, 0.185), expected))
        self.assertEqual(expected.shape, SHAPE)
        self.assertEqual(expected.dtype, np.float32)
        self.assertTrue(np.isfinite(expected).all())
        self.assertGreaterEqual(float(expected.min()), 0.0)
        self.assertLessEqual(float(expected.max()), 1.0)

    def test_wanet_cover_adds_uniform_jitter_to_fixed_warp_without_antenna_mixing(self):
        trigger = WaNetTrigger(seed=42, noise_strength=1.0)
        base = trigger.sampling_grid(dose=1.0, eps=0.185)
        uniform = np.random.default_rng(17).uniform(-1.0, 1.0, size=base.shape)
        expected = np.clip(base + 0.185 * uniform, -1.0, 1.0)
        np.testing.assert_array_equal(trigger.noise_sampling_grid(seed=17, eps=0.185), expected)
        amplitude = np.broadcast_to(np.array([0.1, 0.4, 0.9], dtype=np.float32)[:, None, None], SHAPE)
        np.testing.assert_array_equal(trigger.noise_inject(amplitude, seed=17, eps=0.185), amplitude)
        without_jitter = WaNetTrigger(seed=42, noise_strength=0.0)
        np.testing.assert_array_equal(without_jitter.noise_inject(self.csi, seed=17, eps=0.185),
                                      without_jitter.inject(self.csi, dose=1.0, eps=0.185))

    def test_invalid_input_is_rejected(self):
        for cls in CLASSES:
            with self.subTest(trigger=cls.__name__):
                trigger = cls(seed=42)
                for value in (np.zeros((3, 180, 20)), self.csi.astype(complex),
                              np.full(SHAPE, np.nan), np.full(SHAPE, 1.01)):
                    with self.assertRaises(ValueError):
                        trigger.inject(value, 1.0, 0.185)
                for dose, eps in ((-1.0, 0.185), (np.nan, 0.185), (1.0, -0.1), (1.0, np.inf)):
                    with self.assertRaises(ValueError):
                        trigger.inject(self.csi, dose, eps)

    def test_constructor_and_config_validation(self):
        with self.assertRaises(ValueError):
            BadNetsTrigger(patch_subcarriers=115)
        with self.assertRaises(ValueError):
            BadNetsTrigger(patch_start=(113, 9))
        with self.assertRaises(ValueError):
            BadNetsTrigger(antennas=(3,))
        with self.assertRaises(ValueError):
            WaNetTrigger(grid_size=1)
        for name, cls in zip(('badnets', 'blended', 'wanet'), CLASSES):
            trigger = build_traditional_trigger(name, {'experiment_name': 'mmfi', 'seed': 42})
            self.assertIsInstance(trigger, cls)
            np.testing.assert_array_equal(trigger.inject(self.csi, 1.0, 0.185),
                                          cls(seed=42).inject(self.csi, 1.0, 0.185))
        with self.assertRaises(ValueError):
            build_traditional_trigger('wanet', {'experiment_name': 'piw3d'})
        with self.assertRaises(ValueError):
            build_traditional_trigger('unknown', {'experiment_name': 'mmfi'})


if __name__ == '__main__':
    unittest.main()
