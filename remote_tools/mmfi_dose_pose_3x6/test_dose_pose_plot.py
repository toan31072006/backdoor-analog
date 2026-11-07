"""Offline tests using invented poses ONLY, not experimental evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dose_pose_plot", HERE / "Plot-MMFiDosePose.py")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def fixture():
    """Hand-invented poses/response, no CSI or model involved."""
    pose = np.array([
        [0, 0, 0], [.12, .05, 0], [.15, .45, .02], [.17, .85, .04],
        [-.12, .05, 0], [-.16, .45, -.02], [-.18, .85, .03],
        [0, -.22, 0], [0, -.46, 0], [0, -.59, 0], [0, -.78, 0],
        [-.20, -.46, 0], [-.37, -.24, .02], [-.44, -.02, .08],
        [.20, -.46, 0], [.37, -.24, -.03], [.44, -.02, .03],
    ], dtype=float)
    truth = np.stack([pose, pose.copy(), pose.copy()])
    truth[1, 12:14] += [-.10, -.35, .10]
    truth[2, 15:17] += [.05, -.50, -.10]
    truth += np.array([[.2, .3, .4], [2., 1., .6], [-.5, .7, 2.]])[:, None, :]
    target = []
    for dose in PLOT.DOSES:
        a = np.deg2rad(40 * dose)
        c, s = np.cos(a), np.sin(a)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        t = truth.copy()
        t[:, [2, 3]] = (truth[:, [2, 3]] - truth[:, 1:2]) @ r.T + truth[:, 1:2]
        target.append(t)
    target = np.stack(target)
    # Arbitrary under-response and fixed error solely for layout/testing.
    bias = np.random.default_rng(707).normal(0, .015, truth.shape)
    pred = truth[None] + bias[None] + .8 * (target - truth[None])
    return dict(ground_truth=truth, predictions=pred, targets=target,
                doses=PLOT.DOSES.copy(), edges=PLOT.EDGES.copy(),
                test_indices=np.array(PLOT.FRAME_IDS))


class DosePosePlotTests(unittest.TestCase):
    def setUp(self):
        self.bundle = fixture()

    def test_expected_array_contract(self):
        PLOT.validate_arrays(self.bundle)

    def test_transposed_predictions_rejected(self):
        self.bundle["predictions"] = self.bundle["predictions"].swapaxes(0, 1)
        with self.assertRaisesRegex(ValueError, "order"):
            PLOT.validate_arrays(self.bundle)

    def test_nan_rejected(self):
        self.bundle["predictions"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            PLOT.validate_arrays(self.bundle)

    def test_dose_grid_rejected(self):
        self.bundle["doses"][1] = .3
        with self.assertRaisesRegex(ValueError, "Dose grid"):
            PLOT.validate_arrays(self.bundle)

    def test_target_changed_outside_limb_rejected(self):
        self.bundle["targets"][3, 1, 15, 1] += .1
        with self.assertRaisesRegex(ValueError, "payload"):
            PLOT.validate_arrays(self.bundle)

    def test_wrong_test_identity_rejected(self):
        self.bundle["test_indices"][2] -= 1
        with self.assertRaisesRegex(ValueError, "identities"):
            PLOT.validate_arrays(self.bundle)

    def test_shared_gt_origin_preserves_absolute_errors(self):
        before = copy.deepcopy(self.bundle)
        gt, pred, target, lo, hi = PLOT.display_arrays(self.bundle)
        self.assertTrue(np.array_equal(gt[:, 0], np.zeros((3, 3))))
        # Error is preserved, not aligned away by each prediction's pelvis.
        np.testing.assert_allclose(np.linalg.norm(pred - target, axis=-1),
            np.linalg.norm(before["predictions"] - before["targets"], axis=-1), atol=1e-14)
        self.assertGreater(np.linalg.norm(pred[0, 0, 0]), 0)
        self.assertTrue(np.all(hi > lo))
        for key in before:
            np.testing.assert_array_equal(self.bundle[key], before[key])

    def test_metrics_are_vector_residuals(self):
        rows = PLOT.metric_rows(self.bundle)
        self.assertEqual(len(rows), 18)
        self.assertEqual(sorted({row["test_index"] for row in rows}), PLOT.FRAME_IDS)
        for row in rows:
            if row["dose"] == 0:
                self.assertEqual(row["DCE_mm"], 0)
                self.assertEqual(row["predicted_mm"], 0)
            else:
                self.assertAlmostEqual(row["predicted_mm"], .8 * row["requested_mm"])
                self.assertAlmostEqual(row["DCE_mm"], .2 * row["requested_mm"])

    def test_nonresponse_floor(self):
        self.bundle["predictions"][:] = self.bundle["predictions"][0]
        for row in PLOT.metric_rows(self.bundle):
            self.assertAlmostEqual(row["DCE_mm"], row["requested_mm"])

    def test_reversed_response_is_not_control(self):
        gt = self.bundle["ground_truth"]
        self.bundle["predictions"] = gt - (self.bundle["targets"] - gt)
        for row in PLOT.metric_rows(self.bundle):
            self.assertAlmostEqual(row["predicted_mm"], row["requested_mm"])
            self.assertAlmostEqual(row["DCE_mm"], 2 * row["requested_mm"])

    def test_loader_rejects_explicit_synthetic_export(self):
        meta = dict(seed=42, rho=.1, split="P1S1", epoch=49,
                    cfg_fingerprint=PLOT.FINGERPRINT, checkpoint_sha256=PLOT.CHECKPOINT_SHA,
                    new_training=False, test_indices=PLOT.FRAME_IDS, units="metres",
                    max_reference_error_delta_mm=0., synthetic=True)
        self.bundle.update(metadata_json=np.array(json.dumps(meta)),
                           reference_target_error_mm=PLOT.EXPECTED_T,
                           reference_clean_to_target_error_mm=PLOT.EXPECTED_C)
        with tempfile.TemporaryDirectory(prefix="dose_pose_unit_") as tmp:
            path = Path(tmp) / "SYNTHETIC_TEST_ONLY.npz"
            np.savez_compressed(path, **self.bundle)
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                PLOT.load_bundle(path)

    def test_loader_happy_path_and_replay_tampering_with_mocked_references(self):
        # Numeric references are mocked within this test, never in production code.
        pred, target = self.bundle["predictions"], self.bundle["targets"]
        et = np.linalg.norm(pred[-1][:, [2, 3]] - target[-1][:, [2, 3]], axis=-1) * 1000
        ec = np.linalg.norm(pred[0][:, [2, 3]] - target[-1][:, [2, 3]], axis=-1) * 1000
        meta = dict(seed=42, rho=.1, split="P1S1", epoch=49,
                    cfg_fingerprint=PLOT.FINGERPRINT, checkpoint_sha256=PLOT.CHECKPOINT_SHA,
                    new_training=False, test_indices=PLOT.FRAME_IDS, units="metres",
                    max_reference_error_delta_mm=0.)
        self.bundle.update(metadata_json=np.array(json.dumps(meta)),
                           reference_target_error_mm=et, reference_clean_to_target_error_mm=ec)
        with tempfile.TemporaryDirectory(prefix="dose_pose_unit_") as tmp:
            path = Path(tmp) / "MOCKED_REFERENCE_TEST_ONLY.npz"
            np.savez_compressed(path, **self.bundle)
            with mock.patch.object(PLOT, "EXPECTED_T", et), mock.patch.object(PLOT, "EXPECTED_C", ec):
                loaded = PLOT.load_bundle(path)
                np.testing.assert_array_equal(loaded["predictions"], pred)
                self.bundle["predictions"][-1, 1, 2, 0] += 1.
                np.savez_compressed(path, **self.bundle)
                with self.assertRaisesRegex(ValueError, "replay"):
                    PLOT.load_bundle(path)

    def test_render_18_panels_shared_camera_and_limits(self):
        fig = PLOT.create_figure(self.bundle,
            fixture_label="SYNTHETIC LAYOUT TEST - NOT MODEL RESULTS")
        self.assertEqual(len(fig.axes), 18)
        labels = [text.get_text() for text in fig.texts]
        self.assertEqual([label for label in labels if label.startswith("(")],
                         ["(a) Input 1", "(b) Input 2", "(c) Input 3"])
        self.assertFalse(any("TEST index" in label for label in labels))
        first = fig.axes[0]
        for ax in fig.axes:
            np.testing.assert_allclose(ax.get_xlim(), first.get_xlim())
            np.testing.assert_allclose(ax.get_ylim(), first.get_ylim())
            np.testing.assert_allclose(ax.get_zlim(), first.get_zlim())
            self.assertEqual((ax.elev, ax.azim), (12., -72.))
        self.assertTrue(any("SYNTHETIC" in text.get_text() for text in fig.texts))
        # Actual renderer smoke check, with an unmistakably named/watermarked preview.
        preview = HERE / "previews" / "synthetic_3x6_layout_check.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        if not preview.exists():
            with preview.open("xb") as handle:
                fig.savefig(handle, format="png", dpi=180)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_output_writer_and_overwrite_refusal_with_mock_loader(self):
        self.bundle["metadata"] = {"synthetic": True, "scope": "unit test only"}
        original_create = PLOT.create_figure

        def watermarked(*args, **kwargs):
            return original_create(*args, **kwargs, fixture_label="SYNTHETIC TEST ONLY")

        with tempfile.TemporaryDirectory(prefix="dose_pose_writer_unit_") as tmp:
            inp, out = Path(tmp) / "SYNTHETIC.npz", Path(tmp) / "SYNTHETIC_OUTPUT"
            np.savez_compressed(inp, **fixture())
            argv = ["plot", "--input", str(inp), "--out-dir", str(out)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(PLOT, "load_bundle", return_value=self.bundle), \
                    mock.patch.object(PLOT, "create_figure", side_effect=watermarked):
                PLOT.main()
                for suffix in ("pdf", "png", "csv", "json"):
                    self.assertGreater((out / f"fig_same_input_dose_s42.{suffix}").stat().st_size, 100)
                manifest = json.loads((out / "fig_same_input_dose_s42.json").read_text())
                self.assertEqual(manifest["source_metadata"]["synthetic"], True)
                self.assertEqual((manifest["rows"], manifest["columns"]), (3, 6))
                with self.assertRaises(FileExistsError):
                    PLOT.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
