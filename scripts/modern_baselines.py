#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors


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
    if "=" not in item:
        raise ValueError(f"--result must be teacher=path, got: {item}")
    teacher, path = item.split("=", 1)
    teacher = teacher.strip()
    path = path.strip()
    if not teacher or not path:
        raise ValueError(f"Invalid --result argument: {item}")
    return teacher, path


# ============================================================
# Basic helpers
# ============================================================

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(exp_x.sum(axis=axis, keepdims=True), 1e-12, None)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(denom, eps, None)


def entropy_from_proba(proba: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(proba, eps, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def get_first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


# ============================================================
# Schema-flexible extraction
# ============================================================

def extract_standardized_arrays(data: Dict[str, Any], teacher_name: str) -> Dict[str, Any]:
    """
    Standardize a teacher NPZ to:
      y_true: [N]
      paths:  [N]
      proba:  [N, C]
      feats:  [N, D] or None
    """

    y_true = get_first_present(data, ["labels", "y_true", "targets"])
    if y_true is None:
        raise ValueError(
            f"Teacher {teacher_name} is missing labels. "
            f"Tried keys: labels, y_true, targets. "
            f"Available keys: {sorted(k for k in data.keys() if not k.startswith('_'))}"
        )
    y_true = np.asarray(y_true, dtype=np.int64)

    paths = get_first_present(data, ["paths", "image_paths", "filenames"])
    if paths is None:
        raise ValueError(
            f"Teacher {teacher_name} is missing paths. "
            f"Tried keys: paths, image_paths, filenames. "
            f"Available keys: {sorted(k for k in data.keys() if not k.startswith('_'))}"
        )
    paths = np.asarray(paths, dtype=object)

    proba = get_first_present(data, ["proba", "probs", "probabilities"])
    logits = get_first_present(data, ["logits"])

    if proba is not None:
        proba = np.asarray(proba, dtype=np.float32)
    elif logits is not None:
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim != 2:
            raise ValueError(f"Teacher {teacher_name} logits must be [N, C], got {logits.shape}")
        proba = softmax_np(logits, axis=1).astype(np.float32)
    else:
        raise ValueError(
            f"Teacher {teacher_name} is missing both proba and logits. "
            f"Tried keys: proba/probs/probabilities and logits. "
            f"Available keys: {sorted(k for k in data.keys() if not k.startswith('_'))}"
        )

    if proba.ndim != 2:
        raise ValueError(f"Teacher {teacher_name} proba must be [N, C], got {proba.shape}")

    feats = get_first_present(data, ["feats", "features", "x_feats", "embeddings"])
    if feats is not None:
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(f"Teacher {teacher_name} feats must be [N, D], got {feats.shape}")

    saved_preds = get_first_present(data, ["preds", "y_pred"])
    if saved_preds is not None:
        saved_preds = np.asarray(saved_preds, dtype=np.int64)
        recomputed_preds = proba.argmax(axis=1)
        if len(saved_preds) == len(recomputed_preds) and not np.array_equal(saved_preds, recomputed_preds):
            print(f"[warn] {teacher_name}: saved preds do not match recomputed argmax(proba). Continuing.")

    return {
        "y_true": y_true,
        "paths": paths,
        "proba": proba,
        "feats": feats,
    }


def ensure_alignment_and_standardize(
    teacher_to_data: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    teacher_names = sorted(teacher_to_data.keys())
    if not teacher_names:
        raise ValueError("No teacher outputs loaded.")

    standardized = {
        t: extract_standardized_arrays(teacher_to_data[t], t)
        for t in teacher_names
    }

    ref_teacher = teacher_names[0]
    ref = standardized[ref_teacher]

    ref_y = ref["y_true"]
    ref_paths = ref["paths"]
    ref_proba = ref["proba"]

    for teacher in teacher_names[1:]:
        cur = standardized[teacher]

        if len(cur["y_true"]) != len(ref_y):
            raise ValueError(
                f"Sample count mismatch: {teacher} has {len(cur['y_true'])} "
                f"vs {ref_teacher} has {len(ref_y)}"
            )

        if not np.array_equal(cur["y_true"], ref_y):
            raise ValueError(f"y_true mismatch between {ref_teacher} and {teacher}")

        if len(cur["paths"]) != len(ref_paths):
            raise ValueError(f"path count mismatch between {ref_teacher} and {teacher}")

        if not np.array_equal(cur["paths"], ref_paths):
            raise ValueError(f"path ordering mismatch between {ref_teacher} and {teacher}")

        if cur["proba"].shape != ref_proba.shape:
            raise ValueError(
                f"Probability shape mismatch: {teacher} has {cur['proba'].shape}, "
                f"{ref_teacher} has {ref_proba.shape}"
            )

    has_features = all(standardized[t]["feats"] is not None for t in teacher_names)

    return standardized, {
        "y_true": ref_y,
        "paths": ref_paths,
        "num_classes": ref_proba.shape[1],
        "num_samples": len(ref_y),
        "has_features": has_features,
    }


def build_teacher_arrays_from_standardized(
    standardized: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], np.ndarray, np.ndarray | None]:
    teacher_names = sorted(standardized.keys())

    proba_list = []
    feat_list = []

    for teacher in teacher_names:
        proba = np.asarray(standardized[teacher]["proba"], dtype=np.float32)
        proba_list.append(proba)

        feats = standardized[teacher]["feats"]
        if feats is not None:
            feat_list.append(l2_normalize(np.asarray(feats, dtype=np.float32), axis=1))

    proba_stack = np.stack(proba_list, axis=0)  # [M, N, C]

    feat_stack = None
    if len(feat_list) == len(teacher_names):
        feat_stack = np.stack(feat_list, axis=0)  # [M, N, D]

    return teacher_names, proba_stack, feat_stack


def build_anchor_features(feat_stack: np.ndarray) -> np.ndarray:
    parts = [feat_stack[m] for m in range(feat_stack.shape[0])]
    anchor = np.concatenate(parts, axis=1)
    return l2_normalize(anchor, axis=1)


# ============================================================
# Single-teacher eval
# ============================================================

def evaluate_single_teacher(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = proba.argmax(axis=1)
    return compute_metrics(y_true, y_pred)


# ============================================================
# Ensemble helpers
# ============================================================

def weighted_ensemble_predictions(proba_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = (weights[:, :, None] * proba_stack).sum(axis=0)
    return weighted.argmax(axis=1)


def metrics_from_mean_proba(y_true: np.ndarray, mean_proba: np.ndarray) -> Dict[str, float]:
    y_pred = mean_proba.argmax(axis=1)
    return compute_metrics(y_true, y_pred)


# ============================================================
# Modern baselines
# ============================================================

def baseline_uniform(
    y_true: np.ndarray,
    proba_stack: np.ndarray,
) -> Dict[str, Any]:
    mean_proba = proba_stack.mean(axis=0)
    out = metrics_from_mean_proba(y_true, mean_proba)
    out["method"] = "uniform_probability_ensemble"
    out["selected_teacher"] = None
    out["uses_target_labels_for_selection"] = False
    return out


def baseline_entropy_weighted(
    y_true: np.ndarray,
    proba_stack: np.ndarray,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    ent = entropy_from_proba(proba_stack, axis=2)  # [M, N]
    scores = -ent / max(temperature, 1e-8)
    weights = softmax_np(scores, axis=0)
    y_pred = weighted_ensemble_predictions(proba_stack, weights)

    out = compute_metrics(y_true, y_pred)
    out["method"] = "entropy_weighted_ensemble"
    out["selected_teacher"] = None
    out["entropy_temperature"] = temperature
    out["uses_target_labels_for_selection"] = False
    return out


def baseline_agreement_weighted(
    y_true: np.ndarray,
    proba_stack: np.ndarray,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    mean_proba = proba_stack.mean(axis=0, keepdims=True)

    kl = (
        proba_stack
        * (
            np.log(np.clip(proba_stack, 1e-12, 1.0))
            - np.log(np.clip(mean_proba, 1e-12, 1.0))
        )
    ).sum(axis=2)

    scores = -kl / max(temperature, 1e-8)
    weights = softmax_np(scores, axis=0)
    y_pred = weighted_ensemble_predictions(proba_stack, weights)

    out = compute_metrics(y_true, y_pred)
    out["method"] = "agreement_weighted_ensemble"
    out["selected_teacher"] = None
    out["agreement_temperature"] = temperature
    out["uses_target_labels_for_selection"] = False
    return out


def baseline_knn_agreement(
    y_true: np.ndarray,
    proba_stack: np.ndarray,
    anchor_features: np.ndarray,
    k: int = 15,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    m, n, _ = proba_stack.shape
    k_eff = min(max(2, k), n)

    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nbrs.fit(anchor_features)
    _, indices = nbrs.kneighbors(anchor_features)

    local_scores = np.zeros((m, n), dtype=np.float32)

    for i in range(n):
        neigh_idx = indices[i]
        neigh_mean = proba_stack[:, neigh_idx, :].mean(axis=1)

        kl = (
            proba_stack[:, i, :]
            * (
                np.log(np.clip(proba_stack[:, i, :], 1e-12, 1.0))
                - np.log(np.clip(neigh_mean, 1e-12, 1.0))
            )
        ).sum(axis=1)

        local_scores[:, i] = -kl / max(temperature, 1e-8)

    weights = softmax_np(local_scores, axis=0)
    y_pred = weighted_ensemble_predictions(proba_stack, weights)

    out = compute_metrics(y_true, y_pred)
    out["method"] = "knn_agreement_weighted_ensemble"
    out["selected_teacher"] = None
    out["knn_k"] = k_eff
    out["knn_temperature"] = temperature
    out["uses_target_labels_for_selection"] = False
    return out


def baseline_cluster_routing(
    y_true: np.ndarray,
    proba_stack: np.ndarray,
    anchor_features: np.ndarray,
    n_clusters: int = 10,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    m, n, _ = proba_stack.shape
    n_clusters_eff = min(max(2, n_clusters), n)

    kmeans = KMeans(n_clusters=n_clusters_eff, random_state=0, n_init=10)
    cluster_ids = kmeans.fit_predict(anchor_features)

    conf = proba_stack.max(axis=2)
    cluster_weights = np.zeros((m, n_clusters_eff), dtype=np.float32)

    for cid in range(n_clusters_eff):
        mask = cluster_ids == cid
        if not np.any(mask):
            cluster_weights[:, cid] = 1.0 / m
            continue
        cluster_conf = conf[:, mask].mean(axis=1)
        cluster_weights[:, cid] = softmax_np(cluster_conf / max(temperature, 1e-8), axis=0)

    weights = cluster_weights[:, cluster_ids]
    y_pred = weighted_ensemble_predictions(proba_stack, weights)

    out = compute_metrics(y_true, y_pred)
    out["method"] = "cluster_routing_ensemble"
    out["selected_teacher"] = None
    out["n_clusters"] = n_clusters_eff
    out["cluster_temperature"] = temperature
    out["uses_target_labels_for_selection"] = False
    return out


def baseline_greedy_ensemble_selection(
    y_true: np.ndarray,
    proba_stack: np.ndarray,
    teacher_names: List[str],
    max_models: int = 5,
) -> Dict[str, Any]:
    m, _, _ = proba_stack.shape
    max_models_eff = min(max(1, max_models), m)

    selected: List[int] = []
    remaining = list(range(m))
    current_mean = None

    for _ in range(max_models_eff):
        best_score = None
        best_idx = None
        best_candidate = None

        for cand in remaining:
            if current_mean is None:
                candidate_mean = proba_stack[cand]
            else:
                candidate_mean = (len(selected) * current_mean + proba_stack[cand]) / (len(selected) + 1)

            score = -float(entropy_from_proba(candidate_mean, axis=1).mean())

            if best_score is None or score > best_score:
                best_score = score
                best_idx = cand
                best_candidate = candidate_mean

        selected.append(best_idx)
        remaining.remove(best_idx)
        current_mean = best_candidate

    y_pred = current_mean.argmax(axis=1)
    out = compute_metrics(y_true, y_pred)
    out["method"] = "greedy_subset_ensemble"
    out["selected_teacher"] = None
    out["selected_teachers_json"] = json.dumps([teacher_names[i] for i in selected])
    out["greedy_subset_max_size"] = max_models_eff
    out["uses_target_labels_for_selection"] = False
    return out


def baseline_diversity_subset(
    y_true: np.ndarray,
    proba_stack: np.ndarray,
    teacher_names: List[str],
    max_models: int = 5,
) -> Dict[str, Any]:
    m, _, _ = proba_stack.shape
    max_models_eff = min(max(1, max_models), m)

    disagreement = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            d = float(np.mean(np.abs(proba_stack[i] - proba_stack[j])))
            disagreement[i, j] = d
            disagreement[j, i] = d

    selected = [int(np.argmax(disagreement.sum(axis=1)))]

    while len(selected) < max_models_eff:
        best_score = None
        best_idx = None
        for cand in range(m):
            if cand in selected:
                continue
            score = float(disagreement[cand, selected].sum())
            if best_score is None or score > best_score:
                best_score = score
                best_idx = cand
        selected.append(best_idx)

    mean_proba = proba_stack[selected].mean(axis=0)
    out = metrics_from_mean_proba(y_true, mean_proba)
    out["method"] = "diversity_subset_ensemble"
    out["selected_teacher"] = None
    out["selected_teachers_json"] = json.dumps([teacher_names[i] for i in selected])
    out["diversity_subset_size"] = max_models_eff
    out["uses_target_labels_for_selection"] = False
    return out


def baseline_entropy_sharpening(
    y_true: np.ndarray,
    proba_stack: np.ndarray,
    sharpening_temp: float = 0.7,
) -> Dict[str, Any]:
    temp = max(sharpening_temp, 1e-8)
    sharpened = softmax_np(np.log(np.clip(proba_stack, 1e-12, 1.0)) / temp, axis=2)
    mean_proba = sharpened.mean(axis=0)

    out = metrics_from_mean_proba(y_true, mean_proba)
    out["method"] = "entropy_sharpening_ensemble"
    out["selected_teacher"] = None
    out["sharpening_temp"] = sharpening_temp
    out["uses_target_labels_for_selection"] = False
    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate modern ensemble baselines and emit the same CSV outputs "
            "as the classical probe baseline script."
        )
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Teacher result in the form teacher_name=path_to_npz",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="modern_baselines")
    parser.add_argument("--debug_keys", action="store_true")

    parser.add_argument("--entropy_temperature", type=float, default=1.0)
    parser.add_argument("--agreement_temperature", type=float, default=1.0)
    parser.add_argument("--knn_k", type=int, default=15)
    parser.add_argument("--knn_temperature", type=float, default=1.0)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--cluster_temperature", type=float, default=1.0)
    parser.add_argument("--greedy_max_models", type=int, default=5)
    parser.add_argument("--diversity_max_models", type=int, default=5)
    parser.add_argument("--sharpening_temp", type=float, default=0.7)

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

    if args.debug_keys:
        print("\n=== NPZ keys by teacher ===")
        for teacher in sorted(teacher_to_data.keys()):
            visible_keys = sorted(k for k in teacher_to_data[teacher].keys() if not k.startswith("_"))
            print(f"{teacher}: {visible_keys}")

    standardized, info = ensure_alignment_and_standardize(teacher_to_data)
    y_true = info["y_true"]
    paths = info["paths"]

    teacher_names, proba_stack, feat_stack = build_teacher_arrays_from_standardized(standardized)
    anchor_features = build_anchor_features(feat_stack) if feat_stack is not None else None

    # --------------------------------------------------------
    # Shared metadata
    # --------------------------------------------------------
    ref_meta = teacher_to_data[teacher_names[0]].get("_meta", {})
    dataset_name = ref_meta.get("dataset_name", "unknown")
    train_domain = ref_meta.get("train_domain", "unknown")
    target_domain = ref_meta.get("target_domain", "unknown")
    split = ref_meta.get("split", "unknown")

    # --------------------------------------------------------
    # Teacher metrics CSV (same format as old script)
    # --------------------------------------------------------
    teacher_rows = []
    teacher_accs: Dict[str, float] = {}

    for teacher in teacher_names:
        proba = standardized[teacher]["proba"]
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
            "source_npz": teacher_to_data[teacher].get("_path", ""),
        })

    teacher_df = pd.DataFrame(teacher_rows).sort_values(
        ["accuracy", "teacher"], ascending=[False, True]
    ).reset_index(drop=True)
    teacher_df.to_csv(outdir / f"teacher_metrics_{args.tag}.csv", index=False)

    oracle_teacher = str(teacher_df.iloc[0]["teacher"])
    oracle_acc = float(teacher_df.iloc[0]["accuracy"])
    oracle_bal_acc = float(teacher_df.iloc[0]["balanced_accuracy"])

    # --------------------------------------------------------
    # Baseline metrics CSV (same format as old script)
    # --------------------------------------------------------
    baseline_rows = []

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
        "uses_target_labels_for_selection": True,
    })

    def attach_common(row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        out["dataset_name"] = dataset_name
        out["train_domain"] = train_domain
        out["target_domain"] = target_domain
        out["split"] = split
        return out

    baseline_rows.append(attach_common(
        baseline_uniform(y_true=y_true, proba_stack=proba_stack)
    ))

    baseline_rows.append(attach_common(
        baseline_entropy_weighted(
            y_true=y_true,
            proba_stack=proba_stack,
            temperature=args.entropy_temperature,
        )
    ))

    baseline_rows.append(attach_common(
        baseline_agreement_weighted(
            y_true=y_true,
            proba_stack=proba_stack,
            temperature=args.agreement_temperature,
        )
    ))

    if anchor_features is not None:
        baseline_rows.append(attach_common(
            baseline_knn_agreement(
                y_true=y_true,
                proba_stack=proba_stack,
                anchor_features=anchor_features,
                k=args.knn_k,
                temperature=args.knn_temperature,
            )
        ))

        baseline_rows.append(attach_common(
            baseline_cluster_routing(
                y_true=y_true,
                proba_stack=proba_stack,
                anchor_features=anchor_features,
                n_clusters=args.n_clusters,
                temperature=args.cluster_temperature,
            )
        ))
    else:
        print("No feature tensors found across all teachers; skipping kNN and cluster-routing baselines.")

    baseline_rows.append(attach_common(
        baseline_greedy_ensemble_selection(
            y_true=y_true,
            proba_stack=proba_stack,
            teacher_names=teacher_names,
            max_models=args.greedy_max_models,
        )
    ))

    baseline_rows.append(attach_common(
        baseline_diversity_subset(
            y_true=y_true,
            proba_stack=proba_stack,
            teacher_names=teacher_names,
            max_models=args.diversity_max_models,
        )
    ))

    baseline_rows.append(attach_common(
        baseline_entropy_sharpening(
            y_true=y_true,
            proba_stack=proba_stack,
            sharpening_temp=args.sharpening_temp,
        )
    ))

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(outdir / f"baseline_metrics_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Optional uniform outputs for debugging
    # --------------------------------------------------------
    mean_proba = proba_stack.mean(axis=0)
    np.savez_compressed(
        outdir / f"uniform_ensemble_outputs_{args.tag}.npz",
        y_true=y_true.astype(np.int64),
        y_pred=mean_proba.argmax(axis=1).astype(np.int64),
        proba=mean_proba.astype(np.float32),
        paths=np.asarray(paths, dtype=object),
        meta_json=json.dumps({
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain": target_domain,
            "split": split,
            "teacher_names": teacher_names,
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

    print("\n=== Single-teacher metrics ===")
    print(teacher_df.to_string(index=False))

    print("\n=== Baseline metrics ===")
    display_cols = [c for c in ["method", "accuracy", "balanced_accuracy", "n_samples"] if c in baseline_df.columns]
    print(baseline_df[display_cols].to_string(index=False))

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

#     python scripts\modern_baselines.py `
#         @argsList `
#         --outdir "terra_probe_results\probe_baselines_modern\$trainDomain\$targetDomain" `
#         --tag "terra_incognita_${trainDomain}_${targetDomain}" `
#         --entropy_temperature 1.0 `
#         --agreement_temperature 1.0 `
#         --knn_k 15 `
#         --knn_temperature 1.0 `
#         --n_clusters 10 `
#         --cluster_temperature 1.0 `
#         --greedy_max_models 5 `
#         --diversity_max_models 5 `
#         --sharpening_temp 0.7
# }