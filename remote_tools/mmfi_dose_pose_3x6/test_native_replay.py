"""Synthetic replay-plumbing tests; not CUDA/numerical-reproduction evidence."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dose_native_replay", HERE / "Export-MMFiDosePose.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ArrayTensor:
    """A deliberately tiny NumPy-backed tensor mock; no real model or Torch."""
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)
    def to(self, device):
        if device != "cpu-test-only":
            raise ValueError("Tests must not use CUDA")
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.array


class NativeReplayTests(unittest.TestCase):
    def test_original_batches_and_last_partial_batch(self):
        plan = EXPORT.native_batch_plan(33264, 32)
        self.assertEqual([(p["start"], p["stop"]) for p in plan],
                         [(0, 32), (16608, 16640), (33248, 33264)])
        self.assertEqual([p["selected"][0]["offset"] for p in plan], [0, 24, 15])
        self.assertEqual(sum(p["stop"] - p["start"] for p in plan), 80)

    def test_nearby_selected_frames_share_their_original_batch(self):
        plan = EXPORT.native_batch_plan(33264, 32, indices=[0, 1])
        self.assertEqual(len(plan), 1)
        self.assertEqual([x["offset"] for x in plan[0]["selected"]], [0, 1])

    def test_changed_batch_or_test_count_refused(self):
        for n, b in [(33264, 3), (33264, 16), (33264, True), (33265, 32)]:
            with self.subTest(n=n, batch=b), self.assertRaises(ValueError):
                EXPORT.native_batch_plan(n, b)

    def test_invalid_or_duplicate_ids_refused(self):
        for ids in [[0, 0], [-1], [33264], [True]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                EXPORT.native_batch_plan(33264, 32, indices=ids)

    def test_runtime_matches_deterministic_mmfi_without_tf32_override(self):
        for original_cudnn_tf32 in (True, False):
            fake = SimpleNamespace(backends=SimpleNamespace(
                cudnn=SimpleNamespace(deterministic=False, benchmark=True,
                                      allow_tf32=original_cudnn_tf32, version=lambda: 123),
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
                get_float32_matmul_precision=lambda: "highest")
            record = EXPORT.configure_native_runtime(fake)
            self.assertTrue(fake.backends.cudnn.deterministic)
            self.assertFalse(fake.backends.cudnn.benchmark)
            self.assertEqual(fake.backends.cudnn.allow_tf32, original_cudnn_tf32)
            self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
            self.assertEqual(record["cudnn_allow_tf32"], original_cudnn_tf32)

    def test_reported_0147545mm_mismatch_still_fails(self):
        errors = copy.deepcopy(EXPORT.EXPECTED_T)
        errors[1][0] += .147545
        report = EXPORT.replay_comparison(errors, EXPORT.EXPECTED_C)
        self.assertFalse(report["passed"])
        self.assertEqual(report["tolerance_mm"], .1)
        self.assertAlmostEqual(report["max_abs_delta_mm"], .147545)

    def test_replay_within_unchanged_threshold_passes(self):
        errors = copy.deepcopy(EXPORT.EXPECTED_T)
        errors[2][1] += .05
        report = EXPORT.replay_comparison(errors, EXPORT.EXPECTED_C)
        self.assertTrue(report["passed"])
        self.assertEqual(report["target"]["expected_mm"], EXPORT.EXPECTED_T)

    def test_nan_and_wrong_shape_refused(self):
        for errors in [[[float("nan"), 0]] * 3, [[0, 0]], [[0, 0, 0]] * 3]:
            with self.assertRaises(ValueError):
                EXPORT.replay_comparison(errors, EXPORT.EXPECTED_C)

    def test_failure_saves_evidence_not_a_pose_npz_and_preserves_existing_report(self):
        with tempfile.TemporaryDirectory(prefix="dose_replay_failure_test_") as tmp:
            output = Path(tmp) / "poses.npz"
            report = {"status": "REPLAY_MISMATCH_NO_POSE_EXPORT", "synthetic_test_only": True}
            with contextlib.redirect_stdout(io.StringIO()):
                EXPORT.write_failure_report(output, report)
                before = output.with_suffix(".replay_failed.json").read_bytes()
                EXPORT.write_failure_report(output, dict(report, new_attempt=True))
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(before), report)
            self.assertEqual(output.with_suffix(".replay_failed.json").read_bytes(), before)

    def run_mock_native(self, drift_truth=False):
        visits, modes, batches = [], [], []
        class Base:
            def __len__(self):
                return 33264
        class Dataset:
            def __init__(self, base, trigger, **kwargs):
                self.dose = kwargs["fixed_dose"]
                modes.append(kwargs["mode"])
            def __getitem__(self, index):
                visits.append(index)
                return dict(index=index, dose=self.dose)
        def collate(items):
            ids = np.array([v["index"] for v in items], dtype=np.float32)
            dose = items[0]["dose"]
            batches.append((len(items), int(ids[0]), int(ids[-1])))
            poses = np.broadcast_to(ids[:, None, None, None] * .001, (len(items), 1, 17, 3)).copy()
            targets = poses + dose * .02
            if drift_truth and dose == .4:
                poses = poses + .1
            return dict(csi=ArrayTensor(np.stack((ids, np.full_like(ids, dose)), axis=-1)),
                        pose=ArrayTensor(poses), target=ArrayTensor(targets))
        def model(tensor):
            ids, dose = tensor.array[:, 0], tensor.array[:, 1]
            poses = np.broadcast_to((ids * .001 + dose * .01)[:, None, None, None],
                                    (len(ids), 1, 17, 3)).copy()
            return ArrayTensor(poses), None
        cfg = dict(batch_size=32, pivot=1, payload_axis=[0., 0., 1.], eps=.185,
                   theta_max_deg=40., dose_mode="linear")
        with contextlib.redirect_stdout(io.StringIO()):
            result = EXPORT.predict_native_batches(model, Base(), None, cfg, "cpu-test-only",
                SimpleNamespace(no_grad=contextlib.nullcontext), Dataset, collate)
        return result, visits, modes, batches

    def test_native_batch_loop_retains_fixed_ids_all_six_doses(self):
        (doses, truth, pred, target, plan), visits, modes, batches = self.run_mock_native()
        self.assertEqual(pred.shape, (6, 3, 17, 3))
        self.assertEqual(len(visits), 6 * 80)
        self.assertEqual([b[0] for b in batches], [32, 32, 16] * 6)
        self.assertEqual(modes, ["clean"] + ["trigger@dose"] * 5)
        np.testing.assert_array_equal(target[0], truth)
        for col, dose in enumerate(doses):
            np.testing.assert_allclose(pred[col, :, 0, 0], np.asarray(EXPORT.FRAME_IDS) * .001 + dose * .01,
                                       atol=5e-6, rtol=0)
        self.assertEqual([p["selected"][0]["test_index"] for p in plan], EXPORT.FRAME_IDS)

    def test_truth_drift_across_doses_refused(self):
        with self.assertRaisesRegex(ValueError, "true poses changed"):
            self.run_mock_native(drift_truth=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
