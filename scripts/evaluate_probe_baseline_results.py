#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# ============================================================
# IO
# ============================================================

def load_npz_dict(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}

    meta = {}
    if "meta_json" in out:
        try:
            raw = out["meta_json"]
            if isinstance(raw, np.ndarray):
                raw = raw.item()
            meta = json.loads(raw)
        except Exception:
            meta = {}

    out["_meta"] = meta
    out["_path"] = str(path)
    return out


def parse_result_arg(item: str) -> Tuple[str, str]:
    """
    Expected:
      --result teacher_name=path_to_probe_outputs_domain.npz
    """
    if "=" not in item:
        raise ValueError(f"--result must be teacher=path, got: {item}")
    teacher, path = item.split("=", 1)
    teacher = teacher.strip()
    path = path.strip()
    if not teacher or not path:
        raise ValueError(f"Invalid --result argument: {item}")
    return teacher, path


# ============================================================
# Metrics / helpers
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(exp_x.sum(axis=axis, keepdims=True), 1e-12, None)


def ensure_alignment(teacher_to_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure all teacher outputs correspond to the same target-domain sample set.
    Requires:
      - same y_true
      - same paths
      - same sample count
      - same class count for proba
    """
    teacher_names = sorted(teacher_to_data.keys())
    if not teacher_names:
        raise ValueError("No teacher outputs loaded.")

    ref_teacher = teacher_names[0]
    ref = teacher_to_data[ref_teacher]

    ref_y = np.asarray(ref["y_true"], dtype=np.int64)
    ref_paths = np.asarray(ref["paths"], dtype=object)

    if "proba" not in ref:
        raise ValueError(f"Reference teacher {ref_teacher} missing 'proba'.")

    ref_proba = np.asarray(ref["proba"], dtype=np.float32)
    ref_num_classes = ref_proba.shape[1]

    for teacher in teacher_names[1:]:
        cur = teacher_to_data[teacher]

        if "proba" not in cur:
            raise ValueError(f"Teacher {teacher} missing 'proba'.")

        cur_y = np.asarray(cur["y_true"], dtype=np.int64)
        cur_paths = np.asarray(cur["paths"], dtype=object)
        cur_proba = np.asarray(cur["proba"], dtype=np.float32)

        if len(cur_y) != len(ref_y):
            raise ValueError(
                f"Sample count mismatch: {teacher} has {len(cur_y)} vs {ref_teacher} has {len(ref_y)}"
            )

        if not np.array_equal(cur_y, ref_y):
            raise ValueError(f"y_true mismatch between {ref_teacher} and {teacher}")

        if len(cur_paths) != len(ref_paths):
            raise ValueError(f"path count mismatch between {ref_teacher} and {teacher}")

        if not np.array_equal(cur_paths, ref_paths):
            raise ValueError(f"path ordering mismatch between {ref_teacher} and {teacher}")

        if cur_proba.shape != ref_proba.shape:
            raise ValueError(
                f"Probability shape mismatch: {teacher} has {cur_proba.shape}, "
                f"{ref_teacher} has {ref_proba.shape}"
            )

        if cur_proba.shape[1] != ref_num_classes:
            raise ValueError("Class-count mismatch across teachers.")

    return {
        "y_true": ref_y,
        "paths": ref_paths,
        "num_classes": ref_num_classes,
    }


def average_rank_correlation(rank_a: List[str], rank_b: List[str]) -> float:
    if len(rank_a) != len(rank_b):
        raise ValueError("Rank lists must have same length")

    pos_a = {name: i for i, name in enumerate(rank_a)}
    pos_b = {name: i for i, name in enumerate(rank_b)}

    names = rank_a
    ra = np.array([pos_a[n] for n in names], dtype=np.float64)
    rb = np.array([pos_b[n] for n in names], dtype=np.float64)

    ra = ra - ra.mean()
    rb = rb - rb.mean()

    denom = np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())
    if denom == 0:
        return 1.0
    return float((ra * rb).sum() / denom)


def pairwise_order_flip_fraction(accs_a: Dict[str, float], accs_b: Dict[str, float]) -> float:
    names = sorted(accs_a.keys())
    total = 0
    flips = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t1, t2 = names[i], names[j]
            da = accs_a[t1] - accs_a[t2]
            db = accs_b[t1] - accs_b[t2]

            if da == 0 or db == 0:
                continue

            total += 1
            if np.sign(da) != np.sign(db):
                flips += 1

    if total == 0:
        return 0.0
    return float(flips / total)


# ============================================================
# Baselines
# ============================================================

def evaluate_single_teacher(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = proba.argmax(axis=1)
    return compute_metrics(y_true, y_pred)


def evaluate_uniform_ensemble(y_true: np.ndarray, teacher_to_proba: Dict[str, np.ndarray]) -> Dict[str, float]:
    proba_stack = np.stack([teacher_to_proba[t] for t in sorted(teacher_to_proba.keys())], axis=0)
    mean_proba = proba_stack.mean(axis=0)
    y_pred = mean_proba.argmax(axis=1)
    return compute_metrics(y_true, y_pred)


def evaluate_confidence_hard_selection(
    y_true: np.ndarray,
    teacher_to_proba: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], np.ndarray]:
    teacher_names = sorted(teacher_to_proba.keys())
    proba_stack = np.stack([teacher_to_proba[t] for t in teacher_names], axis=0)  # [T, N, C]
    conf = proba_stack.max(axis=2)                                                # [T, N]
    best_teacher_idx = conf.argmax(axis=0)                                        # [N]
    chosen_proba = proba_stack[best_teacher_idx, np.arange(proba_stack.shape[1])]
    y_pred = chosen_proba.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred)
    return metrics, best_teacher_idx


def evaluate_confidence_soft_ensemble(
    y_true: np.ndarray,
    teacher_to_proba: Dict[str, np.ndarray],
    temperature: float = 1.0,
) -> Dict[str, float]:
    teacher_names = sorted(teacher_to_proba.keys())
    proba_stack = np.stack([teacher_to_proba[t] for t in teacher_names], axis=0)  # [T, N, C]
    conf = proba_stack.max(axis=2)                                                 # [T, N]
    conf = conf / max(temperature, 1e-8)
    weights = softmax_np(conf, axis=0)                                             # [T, N]
    weighted_proba = (weights[:, :, None] * proba_stack).sum(axis=0)
    y_pred = weighted_proba.argmax(axis=1)
    return compute_metrics(y_true, y_pred)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate probe-level baselines from patched probe_outputs_<domain>.npz files.\n\n"
            "Each run should correspond to a single target domain and a single train domain.\n"
            "Provide one NPZ per teacher for that target domain."
        )
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Teacher result in the form teacher_name=path_to_probe_outputs_target_domain.npz",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="probe_level")
    parser.add_argument(
        "--confidence_soft_temperature",
        type=float,
        default=1.0,
        help="Temperature for confidence-weighted soft ensemble.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------
    teacher_to_data: Dict[str, Dict[str, Any]] = {}
    for item in args.result:
        teacher, path = parse_result_arg(item)
        if teacher in teacher_to_data:
            raise ValueError(f"Duplicate teacher: {teacher}")
        teacher_to_data[teacher] = load_npz_dict(path)

    teacher_names = sorted(teacher_to_data.keys())
    alignment = ensure_alignment(teacher_to_data)

    y_true = alignment["y_true"]
    paths = alignment["paths"]

    # Pull shared metadata from first teacher
    ref_meta = teacher_to_data[teacher_names[0]].get("_meta", {})
    dataset_name = ref_meta.get("dataset_name", "unknown")
    train_domain = ref_meta.get("train_domain", "unknown")
    target_domain = ref_meta.get("target_domain", "unknown")
    split = ref_meta.get("split", "unknown")
    class_names = ref_meta.get("class_names", [])

    # --------------------------------------------------------
    # Per-teacher metrics
    # --------------------------------------------------------
    teacher_rows = []
    teacher_accs: Dict[str, float] = {}

    teacher_to_proba = {}
    for teacher in teacher_names:
        data = teacher_to_data[teacher]
        proba = np.asarray(data["proba"], dtype=np.float32)
        teacher_to_proba[teacher] = proba

        metrics = evaluate_single_teacher(y_true, proba)
        teacher_accs[teacher] = metrics["accuracy"]

        teacher_rows.append({
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain": target_domain,
            "split": split,
            "teacher": teacher,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "n_samples": metrics["n_samples"],
            "source_npz": data.get("_path", ""),
        })

    teacher_df = pd.DataFrame(teacher_rows).sort_values(
        ["accuracy", "teacher"], ascending=[False, True]
    ).reset_index(drop=True)
    teacher_df.to_csv(outdir / f"teacher_metrics_{args.tag}.csv", index=False)

    oracle_teacher = str(teacher_df.iloc[0]["teacher"])
    oracle_acc = float(teacher_df.iloc[0]["accuracy"])
    oracle_bal_acc = float(teacher_df.iloc[0]["balanced_accuracy"])

    # --------------------------------------------------------
    # Baselines for this target domain
    # --------------------------------------------------------
    mean_single_teacher_acc = (
        teacher_df[["accuracy"]].mean().iloc[0]
    )

    global_best_teacher = oracle_teacher  # placeholder for single-target evaluation
    # Note:
    # true global-best and LODO require aggregation across target domains.
    # We still output the per-domain single-teacher table here.

    baseline_rows = []

    # Oracle
    baseline_rows.append({
        "dataset_name": dataset_name,
        "train_domain": train_domain,
        "target_domain": target_domain,
        "split": split,
        "method": "oracle_target_best_teacher",
        "selected_teacher": oracle_teacher,
        "accuracy": oracle_acc,
        "balanced_accuracy": oracle_bal_acc,
        "n_samples": int(len(y_true)),
    })

    # Uniform ensemble
    m_uniform = evaluate_uniform_ensemble(y_true, teacher_to_proba)
    baseline_rows.append({
        "dataset_name": dataset_name,
        "train_domain": train_domain,
        "target_domain": target_domain,
        "split": split,
        "method": "uniform_probability_ensemble",
        "selected_teacher": None,
        **m_uniform,
    })

    # Confidence hard
    m_conf_hard, chosen_teacher_idx = evaluate_confidence_hard_selection(y_true, teacher_to_proba)
    chosen_teachers = np.array(teacher_names, dtype=object)[chosen_teacher_idx]
    teacher_counts = pd.Series(chosen_teachers).value_counts().sort_index()

    baseline_rows.append({
        "dataset_name": dataset_name,
        "train_domain": train_domain,
        "target_domain": target_domain,
        "split": split,
        "method": "confidence_hard_selection",
        "selected_teacher": None,
        "selection_histogram_json": teacher_counts.to_json(),
        **m_conf_hard,
    })

    # Confidence soft
    m_conf_soft = evaluate_confidence_soft_ensemble(
        y_true=y_true,
        teacher_to_proba=teacher_to_proba,
        temperature=args.confidence_soft_temperature,
    )
    baseline_rows.append({
        "dataset_name": dataset_name,
        "train_domain": train_domain,
        "target_domain": target_domain,
        "split": split,
        "method": "confidence_soft_ensemble",
        "selected_teacher": None,
        "confidence_soft_temperature": args.confidence_soft_temperature,
        **m_conf_soft,
    })

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(outdir / f"baseline_metrics_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Optional saved merged predictions for debugging / later analysis
    # --------------------------------------------------------
    mean_proba = np.stack([teacher_to_proba[t] for t in teacher_names], axis=0).mean(axis=0)
    np.savez_compressed(
        outdir / f"uniform_ensemble_outputs_{args.tag}.npz",
        y_true=y_true.astype(np.int64),
        y_pred=mean_proba.argmax(axis=1).astype(np.int64),
        proba=mean_proba.astype(np.float32),
        paths=np.array([str(p) for p in paths], dtype=object),
        meta_json=json.dumps({
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain": target_domain,
            "split": split,
            "teacher_names": teacher_names,
            "class_names": class_names,
            "method": "uniform_probability_ensemble",
        }),
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"Train domain:  {train_domain}")
    print(f"Target domain: {target_domain}")
    print(f"Split:         {split}")

    print("\n=== Single-teacher probe metrics ===")
    print(teacher_df.to_string(index=False))

    print("\n=== Probe-level baselines ===")
    print(baseline_df.to_string(index=False))

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()


# $trainDomain = "location_38"

# $targetDomains = @(
#     "location_38",
#     "location_43",
#     "location_46",
#     "location_100"
# )

# $teachers = @(
#     "openclip_l14_openai_qgelu",
#     "openclip_b16_datacomp",
#     "openclip_so400m_siglip",
#     "openclip_l14_dfn2b",
#     "openclip_h14_laion2b",
#     "openclip_h14_378_dfn5b",
#     "openclip_convnext_xxlarge"
# )

# foreach ($targetDomain in $targetDomains) {

#     Write-Host "==============================="
#     Write-Host "Evaluating target: $targetDomain"
#     Write-Host "==============================="

#     $argsList = @()

#     foreach ($t in $teachers) {
#         $path = "terra_probe_results\terra_incognita\$trainDomain\$t\probe_outputs_$targetDomain.npz"

#         if (-not (Test-Path $path)) {
#             throw "Missing file: $path"
#         }

#         $argsList += "--result"
#         $argsList += "${t}=${path}"
#     }

#     python scripts\evaluate_probe_baseline_results.py `
#         @argsList `
#         --outdir "terra_probe_results\probe_baselines\$trainDomain\$targetDomain" `
#         --tag "terra_incognita_${trainDomain}_${targetDomain}" `
#         --confidence_soft_temperature 1.0
# }