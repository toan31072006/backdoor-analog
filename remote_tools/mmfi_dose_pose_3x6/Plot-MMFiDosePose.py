"""Plot the audited real-prediction export as three inputs x six doses.

Requires NumPy and Matplotlib only. Never loads a checkpoint, trains a model,
fits a response, aligns predictions, or modifies the manuscript. The input is
the NPZ produced by the companion Export-MMFiDosePose.py (metres).
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DOSES = np.array([0, .2, .4, .6, .8, 1.], dtype=float)
FRAME_IDS = [0, 16632, 33263]
CHECKPOINT_SHA = "a10aa8ded68b5f9ac447a3bfe16634665cb30b5337027eeea49732a4caa99a66"
FINGERPRINT = "902d56cca15b9c8e3ae28f13e4978fb59be2b10174216d33d4d617e6027ecbee"
EDGES = np.array([
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
])
TARGET_JOINTS = [2, 3]
TARGET_EDGES = np.array([(1, 2), (2, 3)])
EXPECTED_T = np.array([[31.80086326599121, 80.52739715576172],
                       [143.39553833007812, 167.7744140625],
                       [135.762451171875, 217.89186096191406]])
EXPECTED_C = np.array([[177.80282592773438, 427.4768371582031],
                       [289.2816467285156, 571.5725708007812],
                       [138.57679748535156, 484.2738037109375]])
GRAY, ORANGE, BLUE = "#929292", "#d9720a", "#1f5fa8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_arrays(bundle):
    """Structural/geometric checks, also usable with explicit test fixtures."""
    gt, pred, target = (bundle[k] for k in ("ground_truth", "predictions", "targets"))
    require(gt.shape == (3, 17, 3), "Expected three 17-joint ground-truth poses")
    require(pred.shape == target.shape == (6, 3, 17, 3),
            "Expected predictions and targets in [dose, sample, joint, xyz] order")
    for name in ("ground_truth", "predictions", "targets", "doses"):
        require(np.isfinite(bundle[name]).all(), f"Non-finite values: {name}")
    require(bundle["doses"].shape == (6,) and np.allclose(bundle["doses"], DOSES,
            atol=1e-10, rtol=0), "Dose grid must be 0, .2, .4, .6, .8, 1")
    require(np.array_equal(bundle["test_indices"], FRAME_IDS), "Unexpected fixed TEST identities")
    require(np.array_equal(bundle["edges"], EDGES), "Unexpected MM-Fi skeleton topology")
    require(np.array_equal(target[0], gt), "The zero-dose target must equal GT")
    # Verify the stored labels, but never construct or alter model predictions.
    for i, dose in enumerate(DOSES):
        a = np.deg2rad(40 * dose)
        c, s = np.cos(a), np.sin(a)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        expected = gt.astype(float).copy()
        expected[:, TARGET_JOINTS] = (
            (gt[:, TARGET_JOINTS] - gt[:, 1:2]) @ rotation.T + gt[:, 1:2])
        require(np.allclose(target[i], expected, rtol=0, atol=2e-6),
                f"Stored target does not match the recorded payload at d={dose}")


def load_bundle(path):
    """Check this specific export's provenance and saved numerical replay."""
    with np.load(path, allow_pickle=False) as z:
        names = ("ground_truth", "predictions", "targets", "doses", "edges",
                 "test_indices", "reference_target_error_mm",
                 "reference_clean_to_target_error_mm", "metadata_json")
        require(all(k in z.files for k in names), "Incomplete dose-pose export")
        bundle = {k: z[k].copy() for k in names}
    validate_arrays(bundle)
    metadata = json.loads(str(bundle.pop("metadata_json").item()))
    require(isinstance(metadata, dict), "Expected JSON metadata object")
    for key, value in (("seed", 42), ("rho", .1), ("split", "P1S1"), ("epoch", 49),
                       ("cfg_fingerprint", FINGERPRINT), ("checkpoint_sha256", CHECKPOINT_SHA),
                       ("new_training", False), ("test_indices", FRAME_IDS)):
        require(key in metadata and metadata[key] == value, f"Unexpected metadata: {key}")
    require("metres" in metadata.get("units", ""), "Expected coordinates in metres")
    require(metadata.get("max_reference_error_delta_mm", float("inf")) <= .1,
            "Exporter did not pass the 0.1 mm replay check")
    require(metadata.get("synthetic", False) is False, "Synthetic input is not an empirical export")
    pred, target = bundle["predictions"], bundle["targets"]
    endpoint = target[-1][:, TARGET_JOINTS]
    actual_t = np.linalg.norm(pred[-1][:, TARGET_JOINTS] - endpoint, axis=-1) * 1000
    actual_c = np.linalg.norm(pred[0][:, TARGET_JOINTS] - endpoint, axis=-1) * 1000
    for actual, expected, key in (
        (actual_t, EXPECTED_T, "reference_target_error_mm"),
        (actual_c, EXPECTED_C, "reference_clean_to_target_error_mm"),
    ):
        require(bundle[key].shape == (3, 2) and np.isfinite(bundle[key]).all(),
                f"Invalid stored reference array: {key}")
        require(np.max(np.abs(actual - expected)) <= .1, "Prediction replay differs from recorded errors")
        require(np.allclose(actual, bundle[key], atol=.001, rtol=0),
                "Predictions disagree with the export's own reference errors")
    bundle["metadata"] = metadata
    return bundle


def display_arrays(bundle):
    """One translation per row, from GT only; identical for all three series.

    Fixed display rotation (x,y,z) -> (x,z,-y) follows the supplied style script.
    No per-dose/per-prediction alignment, scaling, or orientation normalization.
    """
    origin = bundle["ground_truth"][:, 0:1, :].astype(float)

    def transform(array):
        shifted = array.astype(float) - origin
        return np.stack((shifted[..., 0], shifted[..., 2], -shifted[..., 1]), axis=-1)

    gt = transform(bundle["ground_truth"])
    pred = transform(bundle["predictions"])
    target = transform(bundle["targets"])
    all_points = np.concatenate((gt.reshape(-1, 3), pred.reshape(-1, 3), target.reshape(-1, 3)))
    lo, hi = all_points.min(0), all_points.max(0)
    pad = max(float(np.max(hi - lo)) * .045, .02)
    return gt, pred, target, lo - pad, hi + pad


def metric_rows(bundle):
    """Per-frame target-joint norms, in mm; NOT full-test dataset means."""
    gt = bundle["ground_truth"][:, TARGET_JOINTS].astype(float)
    pred = bundle["predictions"][:, :, TARGET_JOINTS].astype(float)
    target = bundle["targets"][:, :, TARGET_JOINTS].astype(float)
    intended, induced = target - gt, pred - pred[0]
    quantities = {
        "T_MPJPE_mm": np.linalg.norm(pred - target, axis=-1).mean(-1) * 1000,
        "requested_mm": np.linalg.norm(intended, axis=-1).mean(-1) * 1000,
        "predicted_mm": np.linalg.norm(induced, axis=-1).mean(-1) * 1000,
        "DCE_mm": np.linalg.norm(induced - intended, axis=-1).mean(-1) * 1000,
    }
    return [dict(test_index=int(index), dose=float(dose),
                 **{k: float(v[col, row]) for k, v in quantities.items()})
            for row, index in enumerate(bundle["test_indices"])
            for col, dose in enumerate(bundle["doses"])]


def create_figure(bundle, elev=12., azim=-72., fixture_label=None):
    """Rendering only. Tests MUST watermark invented arrays via fixture_label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 8, "pdf.fonttype": 42,
                         "savefig.dpi": 400})
    gt, pred, target, lo, hi = display_arrays(bundle)
    fig = plt.figure(figsize=(7.2, 5.1), facecolor="white")
    left, width = .025, .96 / 6
    row_tops = [.89, .615, .34]

    def bones(ax, pose, edges, color, lw, ls="-", alpha=1.):
        for a, b in edges:
            ax.plot(*pose[[a, b]].T, color=color, lw=lw, ls=ls, alpha=alpha,
                    solid_capstyle="round", dash_capstyle="round")

    for row in range(3):
        fig.text(.035, row_tops[row] + .023,
                 f"({chr(97 + row)}) Input {row + 1}  |  TEST index {int(bundle['test_indices'][row])}",
                 fontsize=8, va="bottom")
        for col, dose in enumerate(bundle["doses"]):
            ax = fig.add_axes([left + col * width, row_tops[row] - .248, width, .25],
                              projection="3d")
            ax.set_proj_type("ortho")
            bones(ax, gt[row], EDGES, GRAY, 1.15)
            ax.scatter(*gt[row].T, s=5, color=GRAY, depthshade=False,
                       edgecolors="white", linewidths=.15)
            bones(ax, pred[col, row], EDGES, BLUE, .9, alpha=.8)
            bones(ax, pred[col, row], TARGET_EDGES, BLUE, 1.55)
            ax.scatter(*pred[col, row, TARGET_JOINTS].T, s=10, color=BLUE,
                       depthshade=False, edgecolors="black", linewidths=.2)
            # At d=0, target=GT; drawing orange over it would hide the clean reference.
            if dose > 0:
                bones(ax, target[col, row], TARGET_EDGES, ORANGE, 1.5, (0, (2.2, 1.3)))
                ax.scatter(*target[col, row, TARGET_JOINTS].T, s=13, marker="s",
                           facecolors="none", edgecolors=ORANGE, linewidths=.7, depthshade=False)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(hi - lo)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            if row == 0:
                fig.text(left + (col + .5) * width, .969, rf"$d={dose:.1f}$",
                         fontsize=9, ha="center")
    handles = [Line2D([], [], color=GRAY, lw=1.5, label="Ground truth"),
               Line2D([], [], color=ORANGE, lw=1.5, ls=(0, (2.2, 1.3)), marker="s",
                      markerfacecolor="none", markersize=3, label="Attacker target (limb)"),
               Line2D([], [], color=BLUE, lw=1.5, marker="o", markersize=3, label="Model prediction")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .018),
               ncol=3, frameon=False, fontsize=8, handlelength=1.7, columnspacing=1.25)
    fig.text(.5, .073, "Same input within each row; fixed view and scale; target = GT at d = 0.",
             ha="center", fontsize=8)
    if fixture_label:
        fig.text(.5, .50, fixture_label, ha="center", va="center", color="#a02c2c",
                 rotation=18, fontsize=17, weight="bold", alpha=.85,
                 bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--elev", type=float, default=12.)
    parser.add_argument("--azim", type=float, default=-72.)
    args = parser.parse_args()
    require(np.isfinite([args.elev, args.azim]).all(), "Camera angles must be finite")
    bundle = load_bundle(args.input)
    stem = "fig_same_input_dose_s42"
    paths = {ext: args.out_dir / f"{stem}.{ext}" for ext in ("pdf", "png", "csv", "json")}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Preserve existing figure; use a new --out-dir: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = create_figure(bundle, args.elev, args.azim)
    for ext in ("pdf", "png"):
        with paths[ext].open("xb") as handle:
            fig.savefig(handle, format=ext, dpi=400, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    rows = metric_rows(bundle)
    with paths["csv"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"input": str(args.input.resolve()),
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_metadata": bundle["metadata"], "rows": 3, "columns": 6,
                "test_indices": FRAME_IDS, "doses": DOSES.tolist(),
                "display": {"transform": "(x,y,z) -> (x,z,-y)",
                            "origin": "each row's GT pelvis; shared across all doses and series",
                            "projection": "orthographic", "elev": args.elev, "azim": args.azim,
                            "scale": "one shared metric box over all 18 panels", "alignment": "none"},
                "metric_scope": "individual frames, two target joints; not full-test means",
                "outputs_sha256": {ext: hashlib.sha256(paths[ext].read_bytes()).hexdigest()
                                   for ext in ("pdf", "png", "csv")}}
    with paths["json"].open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"DOSE_POSE_FIGURE_COMPLETE {paths['pdf']}")
    print(f"Preview: {paths['png']}")


if __name__ == "__main__":
    main()
