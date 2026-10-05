"""Fail closed if a dry-run matrix deviates from the paper main contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SEEDS = {42, 0, 1}


def _same_path(actual: object, expected: Path) -> bool:
    return Path(str(actual)).expanduser().resolve() == expected.expanduser().resolve()


def _load(path: Path) -> list[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read matrix {path}: {exc}") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"matrix must be a JSON list: {path}")
    return value


def _assert_common(
    rows: list[dict],
    *,
    dataset: str,
    dataset_root: Path,
    action_npy: Path,
    pivot: int,
) -> None:
    epochs = 50 if dataset == "mmfi" else 200
    theta = 40.0 if dataset == "mmfi" else 90.0
    eps = 0.185 if dataset == "mmfi" else 0.3
    optimizer = "sgd" if dataset == "mmfi" else "adamw"
    for index, row in enumerate(rows):
        cfg = row.get("config")
        if not isinstance(cfg, dict):
            raise RuntimeError(f"row {index} has no resolved config")
        expected = {
            "model": "hpeli",
            "scenario": "bend",
            "pivot": pivot,
            "theta_max_deg": theta,
            "eps": eps,
            "dose_mode": "linear",
            "poison_select": "uniform",
            "victim_loss": "mpjpe",
            "epochs": epochs,
            "batch_size": 32,
            "optimizer": optimizer,
            "lr": 1e-3 if dataset == "mmfi" else 1e-2,
            "payload_axis": [0.0, 0.0, 1.0],
        }
        for key, wanted in expected.items():
            actual = row.get(key) if key in ("model", "scenario") else cfg.get(key)
            if actual != wanted:
                raise RuntimeError(
                    f"{dataset} row {index}: {key}={actual!r}, expected {wanted!r}"
                )
        if not _same_path(cfg.get("dataset_root"), dataset_root):
            raise RuntimeError(
                f"{dataset} row {index}: wrong dataset_root {cfg.get('dataset_root')!r}"
            )
        if not _same_path(cfg.get("action_npy"), action_npy):
            raise RuntimeError(
                f"{dataset} row {index}: wrong action_npy {cfg.get('action_npy')!r}"
            )
        if cfg.get("dose_grid") != [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            raise RuntimeError(f"{dataset} row {index}: wrong dose_grid")


def verify(
    *,
    dataset: str,
    attacks_path: Path,
    clean_path: Path,
    dataset_root: Path,
    action_npy: Path,
    pivot: int,
) -> None:
    attacks = _load(attacks_path)
    clean = _load(clean_path)
    _assert_common(
        attacks + clean,
        dataset=dataset,
        dataset_root=dataset_root,
        action_npy=action_npy,
        pivot=pivot,
    )

    observed_attacks = Counter((r.get("trigger"), r.get("seed")) for r in attacks)
    expected_attacks = Counter(
        ("micro_dropper", seed) for seed in SEEDS
    )
    if observed_attacks != expected_attacks:
        raise RuntimeError(
            f"{dataset}: attack matrix mismatch; observed={observed_attacks}, "
            f"expected={expected_attacks}"
        )
    if any(float(row["config"].get("rho", -1)) != 0.4 for row in attacks):
        raise RuntimeError(f"{dataset}: attacked rows must use rho=0.4")

    observed_clean = Counter((r.get("trigger"), r.get("seed")) for r in clean)
    expected_clean = Counter(("micro_dropper", seed) for seed in SEEDS)
    if observed_clean != expected_clean:
        raise RuntimeError(
            f"{dataset}: clean matrix mismatch; observed={observed_clean}, "
            f"expected={expected_clean}"
        )
    if any(float(row["config"].get("rho", -1)) != 0.0 for row in clean):
        raise RuntimeError(f"{dataset}: clean control rows must use rho=0")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("mmfi", "piw3d"))
    parser.add_argument("--attacks", required=True, type=Path)
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--action-npy", required=True, type=Path)
    parser.add_argument("--pivot", required=True, type=int)
    args = parser.parse_args()

    verify(
        dataset=args.dataset,
        attacks_path=args.attacks,
        clean_path=args.clean,
        dataset_root=args.dataset_root,
        action_npy=args.action_npy,
        pivot=args.pivot,
    )
    print(f"[contract] {args.dataset}: 3 attack + 3 clean rows verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
