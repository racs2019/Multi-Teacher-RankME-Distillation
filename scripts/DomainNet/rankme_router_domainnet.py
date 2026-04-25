#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
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


def parse_keyval_arg(item: str) -> Tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"Expected teacher=path, got: {item}")
    teacher, path = item.split("=", 1)
    teacher = teacher.strip()
    path = path.strip()
    if not teacher or not path:
        raise ValueError(f"Invalid argument: {item}")
    return teacher, path


# ============================================================
# Math helpers
# ============================================================

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(exp_x.sum(axis=axis, keepdims=True), 1e-12, None)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def rankme_effective_rank(
    z: np.ndarray,
    eps: float = 1e-8,
    center_columns: bool = False,
) -> float:
    """
    RankMe effective rank:
      exp( - sum p_k log p_k )
    where p_k are normalized singular values.
    """
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {z.shape}")

    if center_columns:
        z = z - z.mean(axis=0, keepdims=True)

    try:
        s = np.linalg.svd(z, compute_uv=False, full_matrices=False)
    except np.linalg.LinAlgError:
        s = np.linalg.svd(
            z + eps * np.random.randn(*z.shape).astype(np.float32),
            compute_uv=False,
            full_matrices=False,
        )

    s = np.clip(s, eps, None)
    p = s / np.clip(np.sum(s), eps, None)
    h = -np.sum(p * np.log(p + eps))
    return float(np.exp(h))


def robust_zscore_across_teachers(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    x shape: [T, N]
    Robustly normalize teacher scores per sample.
    """
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True)
    return (x - med) / np.clip(1.4826 * mad, eps, None)


# ============================================================
# Alignment / loading
# ============================================================

def ensure_feature_alignment(teacher_to_feat: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    teachers = sorted(teacher_to_feat.keys())
    if not teachers:
        raise ValueError("No teacher features provided.")

    ref_teacher = teachers[0]
    ref = teacher_to_feat[ref_teacher]

    ref_feats = np.asarray(ref["feats"], dtype=np.float32)
    ref_labels = np.asarray(ref["labels"], dtype=np.int64)
    ref_paths = np.asarray(ref["paths"], dtype=object)

    for teacher in teachers[1:]:
        cur = teacher_to_feat[teacher]
        cur_feats = np.asarray(cur["feats"], dtype=np.float32)
        cur_labels = np.asarray(cur["labels"], dtype=np.int64)
        cur_paths = np.asarray(cur["paths"], dtype=object)

        if cur_feats.shape[0] != ref_feats.shape[0]:
            raise ValueError(f"Feature sample-count mismatch for {teacher}")

        if not np.array_equal(cur_labels, ref_labels):
            raise ValueError(f"Feature labels mismatch for {teacher}")

        if not np.array_equal(cur_paths, ref_paths):
            raise ValueError(f"Feature path ordering mismatch for {teacher}")

    return {
        "labels": ref_labels,
        "paths": ref_paths,
        "n_samples": ref_feats.shape[0],
    }


def ensure_probe_alignment(teacher_to_probe: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    teachers = sorted(teacher_to_probe.keys())
    if not teachers:
        raise ValueError("No probe outputs provided.")

    ref_teacher = teachers[0]
    ref = teacher_to_probe[ref_teacher]

    ref_y = np.asarray(ref["y_true"], dtype=np.int64)
    ref_paths = np.asarray(ref["paths"], dtype=object)
    ref_proba = np.asarray(ref["proba"], dtype=np.float32)

    for teacher in teachers[1:]:
        cur = teacher_to_probe[teacher]
        cur_y = np.asarray(cur["y_true"], dtype=np.int64)
        cur_paths = np.asarray(cur["paths"], dtype=object)
        cur_proba = np.asarray(cur["proba"], dtype=np.float32)

        if not np.array_equal(cur_y, ref_y):
            raise ValueError(f"Probe y_true mismatch for {teacher}")
        if not np.array_equal(cur_paths, ref_paths):
            raise ValueError(f"Probe path ordering mismatch for {teacher}")
        if cur_proba.shape != ref_proba.shape:
            raise ValueError(f"Probe proba shape mismatch for {teacher}")

    return {
        "y_true": ref_y,
        "paths": ref_paths,
        "n_samples": ref_proba.shape[0],
        "num_classes": ref_proba.shape[1],
    }


def ensure_cross_alignment(
    feat_info: Dict[str, Any],
    probe_info: Dict[str, Any],
) -> None:
    if not np.array_equal(feat_info["labels"], probe_info["y_true"]):
        raise ValueError("Feature labels and probe y_true do not match.")
    if not np.array_equal(feat_info["paths"], probe_info["paths"]):
        raise ValueError("Feature paths and probe paths do not match.")


def subset_feature_dict_to_probe_paths(
    feat_data: Dict[str, Any],
    probe_paths: np.ndarray,
) -> Dict[str, Any]:
    """
    Subset a feature NPZ dict to exactly the sample order in probe_paths.

    Needed because:
      - features_<train_domain>.npz may contain the full domain
      - probe_outputs_<train_domain>.npz contains only the held-out validation subset
    """
    feat_paths = np.asarray(feat_data["paths"], dtype=object)
    feat_labels = np.asarray(feat_data["labels"], dtype=np.int64)
    feat_feats = np.asarray(feat_data["feats"], dtype=np.float32)

    path_to_idx = {}
    for i, p in enumerate(feat_paths):
        p_str = str(p)
        if p_str in path_to_idx:
            raise ValueError(f"Duplicate feature path detected: {p_str}")
        path_to_idx[p_str] = i

    selected_idx = []
    for p in probe_paths:
        p_str = str(p)
        if p_str not in path_to_idx:
            raise ValueError(f"Probe path not found in feature file: {p_str}")
        selected_idx.append(path_to_idx[p_str])

    selected_idx = np.asarray(selected_idx, dtype=np.int64)

    out = dict(feat_data)
    out["feats"] = feat_feats[selected_idx]
    out["labels"] = feat_labels[selected_idx]
    out["paths"] = feat_paths[selected_idx]
    return out


# ============================================================
# Anchor space / neighborhoods
# ============================================================

def build_anchor_features(
    teacher_to_feat: Dict[str, Dict[str, Any]],
    anchor_mode: str,
    anchor_teacher: str | None,
) -> np.ndarray:
    teachers = sorted(teacher_to_feat.keys())

    if anchor_mode == "single_teacher":
        if anchor_teacher is None:
            raise ValueError("--anchor_teacher is required when anchor_mode=single_teacher")
        if anchor_teacher not in teacher_to_feat:
            raise ValueError(f"Unknown anchor teacher: {anchor_teacher}")
        feats = np.asarray(teacher_to_feat[anchor_teacher]["feats"], dtype=np.float32)
        return l2_normalize(feats)

    if anchor_mode == "concat_all":
        pieces = []
        for teacher in teachers:
            feats = np.asarray(teacher_to_feat[teacher]["feats"], dtype=np.float32)
            pieces.append(l2_normalize(feats))
        return np.concatenate(pieces, axis=1).astype(np.float32)

    if anchor_mode == "mean_all":
        pieces = []
        for teacher in teachers:
            feats = np.asarray(teacher_to_feat[teacher]["feats"], dtype=np.float32)
            pieces.append(l2_normalize(feats))
        return np.mean(np.stack(pieces, axis=0), axis=0).astype(np.float32)

    raise ValueError(f"Unknown anchor_mode: {anchor_mode}")


def compute_knn_indices(anchor_feats: np.ndarray, k: int) -> np.ndarray:
    n = anchor_feats.shape[0]
    k = max(2, min(k, n))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(anchor_feats)
    indices = nn.kneighbors(anchor_feats, return_distance=False)
    return indices.astype(np.int64)


# ============================================================
# RankMe scores
# ============================================================

def compute_local_rankme_scores(
    teacher_to_feat: Dict[str, Dict[str, Any]],
    knn_idx: np.ndarray,
    center_columns: bool = False,
    subsample_dim_cap: int | None = None,
) -> Dict[str, np.ndarray]:
    """
    Returns teacher -> [N] RankMe scores.
    """
    out: Dict[str, np.ndarray] = {}

    for teacher, data in teacher_to_feat.items():
        feats = np.asarray(data["feats"], dtype=np.float32)
        feats = l2_normalize(feats)

        if subsample_dim_cap is not None and feats.shape[1] > subsample_dim_cap:
            feats = feats[:, :subsample_dim_cap]

        n = feats.shape[0]
        scores = np.zeros(n, dtype=np.float32)

        for i in range(n):
            z = feats[knn_idx[i]]
            scores[i] = rankme_effective_rank(
                z,
                center_columns=center_columns,
            )

        out[teacher] = scores

    return out


# ============================================================
# Weight construction
# ============================================================

def get_global_prior_vector(
    teachers: List[str],
    global_prior_csv: str | None,
    target_column: str = "accuracy",
) -> np.ndarray:
    """
    Optional CSV with columns:
      teacher, accuracy
    e.g. teacher_mean_metrics_*.csv
    """
    if global_prior_csv is None:
        return np.zeros(len(teachers), dtype=np.float32)

    df = pd.read_csv(global_prior_csv)
    if "teacher" not in df.columns or target_column not in df.columns:
        raise ValueError(
            f"Global prior CSV must contain columns ['teacher', '{target_column}']"
        )

    score_map = {str(r["teacher"]): float(r[target_column]) for _, r in df.iterrows()}
    vals = np.array([score_map.get(t, 0.0) for t in teachers], dtype=np.float32)

    if np.std(vals) > 1e-8:
        vals = (vals - vals.mean()) / vals.std()
    else:
        vals = np.zeros_like(vals)

    return vals


def build_teacher_weight_matrix(
    teacher_names: List[str],
    teacher_to_probe: Dict[str, Dict[str, Any]],
    teacher_to_rankme: Dict[str, np.ndarray],
    alpha_rankme: float,
    beta_conf: float,
    gamma_prior: float,
    global_prior_vec: np.ndarray,
    weight_temperature: float,
    normalize_rankme_per_sample: bool = True,
    min_teacher_weight: float = 0.0,
    score_mode: str = "hybrid",
) -> Dict[str, np.ndarray]:
    """
    Returns a dict with:
      rankme_scores_z: [T, N]
      conf_scores: [T, N]
      fused_scores: [T, N]
      weights: [T, N]
    """
    first_teacher = teacher_names[0]

    rankme_mat = np.stack(
        [np.asarray(teacher_to_rankme[t], dtype=np.float32) for t in teacher_names],
        axis=0
    )  # [T, N]

    if normalize_rankme_per_sample:
        rankme_z = robust_zscore_across_teachers(rankme_mat)
    else:
        rankme_z = rankme_mat

    conf_mat = np.stack(
        [np.asarray(teacher_to_probe[t]["proba"], dtype=np.float32).max(axis=1) for t in teacher_names],
        axis=0
    )  # [T, N]

    conf_score = np.log(np.clip(conf_mat, 1e-8, 1.0))
    prior = np.asarray(global_prior_vec, dtype=np.float32)[:, None]  # [T, 1]

    if score_mode == "rankme_only":
        fused = alpha_rankme * rankme_z
    elif score_mode == "confidence_only":
        fused = beta_conf * conf_score
    elif score_mode == "hybrid":
        fused = alpha_rankme * rankme_z + beta_conf * conf_score + gamma_prior * prior
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    fused = fused / max(weight_temperature, 1e-8)
    weights = softmax_np(fused, axis=0)  # over teachers

    if min_teacher_weight > 0.0:
        weights = np.maximum(weights, float(min_teacher_weight))
        weights = weights / np.clip(weights.sum(axis=0, keepdims=True), 1e-12, None)

    return {
        "rankme_scores_raw": rankme_mat,
        "rankme_scores_z": rankme_z,
        "conf_scores": conf_score,
        "fused_scores": fused,
        "weights": weights,
    }


# ============================================================
# Prediction
# ============================================================

def weighted_probability_ensemble(
    teacher_names: List[str],
    teacher_to_probe: Dict[str, Dict[str, Any]],
    weights: np.ndarray,
) -> np.ndarray:
    """
    weights: [T, N]
    returns weighted proba [N, C]
    """
    proba_stack = np.stack(
        [np.asarray(teacher_to_probe[t]["proba"], dtype=np.float32) for t in teacher_names],
        axis=0
    )  # [T, N, C]

    weighted = (weights[:, :, None] * proba_stack).sum(axis=0)
    weighted = weighted / np.clip(weighted.sum(axis=1, keepdims=True), 1e-12, None)
    return weighted.astype(np.float32)


def uniform_probability_ensemble(
    teacher_names: List[str],
    teacher_to_probe: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """
    returns uniform ensemble proba [N, C]
    """
    proba_stack = np.stack(
        [np.asarray(teacher_to_probe[t]["proba"], dtype=np.float32) for t in teacher_names],
        axis=0
    )  # [T, N, C]

    mean_proba = proba_stack.mean(axis=0)
    mean_proba = mean_proba / np.clip(mean_proba.sum(axis=1, keepdims=True), 1e-12, None)
    return mean_proba.astype(np.float32)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "RankMe-guided probe router / weighted ensemble.\n"
            "Provide one features NPZ and one probe_outputs NPZ per teacher "
            "for a fixed train_domain + target_domain."
        )
    )
    parser.add_argument("--feature", action="append", required=True, help="teacher=path_to_features_target_domain.npz")
    parser.add_argument("--probe", action="append", required=True, help="teacher=path_to_probe_outputs_target_domain.npz")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="rankme_router")

    # Neighborhood / anchor
    parser.add_argument("--anchor_mode", choices=["single_teacher", "concat_all", "mean_all"], default="concat_all")
    parser.add_argument("--anchor_teacher", type=str, default=None)
    parser.add_argument("--k_neighbors", type=int, default=25)
    parser.add_argument("--center_rankme_columns", action="store_true")
    parser.add_argument("--feature_dim_cap", type=int, default=None)

    # Score fusion
    parser.add_argument("--score_mode", choices=["hybrid", "rankme_only", "confidence_only"], default="hybrid")
    parser.add_argument("--alpha_rankme", type=float, default=2.0)
    parser.add_argument("--beta_conf", type=float, default=1.0)
    parser.add_argument("--gamma_prior", type=float, default=0.5)
    parser.add_argument("--weight_temperature", type=float, default=1.0)
    parser.add_argument("--min_teacher_weight", type=float, default=0.02)
    parser.add_argument("--no_rankme_zscore", action="store_true")

    # Deviation from ensemble
    parser.add_argument(
        "--ensemble_deviation_lambda",
        type=float,
        default=0.15,
        help="How much to deviate from the uniform ensemble toward the RankMe-weighted ensemble.",
    )

    # Optional prior
    parser.add_argument(
        "--global_prior_csv",
        type=str,
        default=None,
        help="Optional teacher_mean_metrics CSV with columns teacher, accuracy",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load features and probes
    # -------------------------
    teacher_to_feat = {}
    for item in args.feature:
        teacher, path = parse_keyval_arg(item)
        teacher_to_feat[teacher] = load_npz_dict(path)

    teacher_to_probe = {}
    for item in args.probe:
        teacher, path = parse_keyval_arg(item)
        teacher_to_probe[teacher] = load_npz_dict(path)

    feat_teachers = set(teacher_to_feat.keys())
    probe_teachers = set(teacher_to_probe.keys())
    if feat_teachers != probe_teachers:
        raise ValueError(
            f"Feature/probe teacher mismatch.\n"
            f"feature={sorted(feat_teachers)}\n"
            f"probe={sorted(probe_teachers)}"
        )

    teacher_names = sorted(feat_teachers)

    # First align probes across teachers
    probe_info = ensure_probe_alignment(teacher_to_probe)

    # Then subset every feature file to the exact probe sample set
    probe_paths = probe_info["paths"]
    for teacher in sorted(teacher_to_feat.keys()):
        teacher_to_feat[teacher] = subset_feature_dict_to_probe_paths(
            teacher_to_feat[teacher],
            probe_paths=probe_paths,
        )

    # Now feature alignment should be exact
    feat_info = ensure_feature_alignment(teacher_to_feat)
    ensure_cross_alignment(feat_info, probe_info)

    y_true = probe_info["y_true"]
    paths = probe_info["paths"]

    ref_meta = teacher_to_probe[teacher_names[0]].get("_meta", {})
    dataset_name = ref_meta.get("dataset_name", "unknown")
    train_domain = ref_meta.get("train_domain", "unknown")
    target_domain = ref_meta.get("target_domain", "unknown")
    split = ref_meta.get("split", "unknown")
    class_names = ref_meta.get("class_names", [])

    # -------------------------
    # Anchor + neighborhoods
    # -------------------------
    anchor_feats = build_anchor_features(
        teacher_to_feat=teacher_to_feat,
        anchor_mode=args.anchor_mode,
        anchor_teacher=args.anchor_teacher,
    )
    knn_idx = compute_knn_indices(anchor_feats, k=args.k_neighbors)

    # -------------------------
    # Local RankMe
    # -------------------------
    teacher_to_rankme = compute_local_rankme_scores(
        teacher_to_feat=teacher_to_feat,
        knn_idx=knn_idx,
        center_columns=args.center_rankme_columns,
        subsample_dim_cap=args.feature_dim_cap,
    )

    # -------------------------
    # Global prior
    # -------------------------
    global_prior_vec = get_global_prior_vector(
        teachers=teacher_names,
        global_prior_csv=args.global_prior_csv,
        target_column="accuracy",
    )

    # -------------------------
    # Build weights
    # -------------------------
    score_pack = build_teacher_weight_matrix(
        teacher_names=teacher_names,
        teacher_to_probe=teacher_to_probe,
        teacher_to_rankme=teacher_to_rankme,
        alpha_rankme=args.alpha_rankme,
        beta_conf=args.beta_conf,
        gamma_prior=args.gamma_prior,
        global_prior_vec=global_prior_vec,
        weight_temperature=args.weight_temperature,
        normalize_rankme_per_sample=not args.no_rankme_zscore,
        min_teacher_weight=args.min_teacher_weight,
        score_mode=args.score_mode,
    )

    weights = score_pack["weights"]

    rankme_proba = weighted_probability_ensemble(
        teacher_names=teacher_names,
        teacher_to_probe=teacher_to_probe,
        weights=weights,
    )

    uniform_proba = uniform_probability_ensemble(
        teacher_names=teacher_names,
        teacher_to_probe=teacher_to_probe,
    )

    lam = float(args.ensemble_deviation_lambda)
    lam = max(0.0, min(1.0, lam))

    final_proba = (1.0 - lam) * uniform_proba + lam * rankme_proba
    final_proba = final_proba / np.clip(final_proba.sum(axis=1, keepdims=True), 1e-12, None)

    y_pred = final_proba.argmax(axis=1)
    metrics = compute_metrics(y_true, y_pred)

    # -------------------------
    # Teacher-only reference metrics
    # -------------------------
    teacher_rows = []
    for idx, t in enumerate(teacher_names):
        proba = np.asarray(teacher_to_probe[t]["proba"], dtype=np.float32)
        yp = proba.argmax(axis=1)
        m = compute_metrics(y_true, yp)
        teacher_rows.append({
            "teacher": t,
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mean_rankme": float(np.mean(score_pack["rankme_scores_raw"][idx])),
            "mean_conf_log": float(np.mean(score_pack["conf_scores"][idx])),
            "global_prior": float(global_prior_vec[idx]),
            "mean_weight": float(np.mean(weights[idx])),
        })

    teacher_df = pd.DataFrame(teacher_rows).sort_values(
        ["accuracy", "teacher"], ascending=[False, True]
    ).reset_index(drop=True)
    teacher_df.to_csv(outdir / f"teacher_reference_metrics_{args.tag}.csv", index=False)

    # -------------------------
    # Save result row
    # -------------------------
    result_row = {
        "dataset_name": dataset_name,
        "train_domain": train_domain,
        "target_domain": target_domain,
        "split": split,
        "method": f"rankme_delta_{args.score_mode}",
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "n_samples": metrics["n_samples"],
        "anchor_mode": args.anchor_mode,
        "anchor_teacher": args.anchor_teacher,
        "k_neighbors": args.k_neighbors,
        "alpha_rankme": args.alpha_rankme,
        "beta_conf": args.beta_conf,
        "gamma_prior": args.gamma_prior,
        "weight_temperature": args.weight_temperature,
        "min_teacher_weight": args.min_teacher_weight,
        "ensemble_deviation_lambda": lam,
        "center_rankme_columns": bool(args.center_rankme_columns),
        "normalize_rankme_per_sample": bool(not args.no_rankme_zscore),
        "feature_dim_cap": args.feature_dim_cap,
        "score_mode": args.score_mode,
    }
    result_df = pd.DataFrame([result_row])
    result_df.to_csv(outdir / f"rankme_result_{args.tag}.csv", index=False)

    # -------------------------
    # Save detailed outputs
    # -------------------------
    np.savez_compressed(
        outdir / f"rankme_outputs_{args.tag}.npz",
        y_true=y_true.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        proba=final_proba.astype(np.float32),
        rankme_proba=rankme_proba.astype(np.float32),
        uniform_proba=uniform_proba.astype(np.float32),
        weights=weights.astype(np.float32),
        rankme_scores_raw=score_pack["rankme_scores_raw"].astype(np.float32),
        rankme_scores_z=score_pack["rankme_scores_z"].astype(np.float32),
        conf_scores=score_pack["conf_scores"].astype(np.float32),
        fused_scores=score_pack["fused_scores"].astype(np.float32),
        knn_idx=knn_idx.astype(np.int64),
        paths=np.array([str(p) for p in paths], dtype=object),
        teacher_names=np.array(teacher_names, dtype=object),
        meta_json=json.dumps({
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain": target_domain,
            "split": split,
            "class_names": class_names,
            "method": f"rankme_delta_{args.score_mode}",
            "anchor_mode": args.anchor_mode,
            "anchor_teacher": args.anchor_teacher,
            "k_neighbors": args.k_neighbors,
            "alpha_rankme": args.alpha_rankme,
            "beta_conf": args.beta_conf,
            "gamma_prior": args.gamma_prior,
            "weight_temperature": args.weight_temperature,
            "min_teacher_weight": args.min_teacher_weight,
            "ensemble_deviation_lambda": lam,
            "center_rankme_columns": bool(args.center_rankme_columns),
            "normalize_rankme_per_sample": bool(not args.no_rankme_zscore),
            "feature_dim_cap": args.feature_dim_cap,
        }),
    )

    with open(outdir / f"rankme_result_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump({
            "result": result_row,
            "teachers": teacher_rows,
        }, f, indent=2)

    # -------------------------
    # Print summary
    # -------------------------
    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"Train domain:  {train_domain}")
    print(f"Target domain: {target_domain}")
    print(f"Split:         {split}")

    print("\n=== Teacher references ===")
    print(teacher_df.to_string(index=False))

    print("\n=== RankMe delta result ===")
    print(result_df.to_string(index=False))

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

# $trainDomain = "quickdraw"

# $targetDomains = @("real", "sketch", "infograph", "quickdraw")

# $teachers = @(
#     "openclip_l14_openai_qgelu",
#     "openclip_b16_datacomp",
#     "openclip_so400m_siglip",
#     "openclip_l14_dfn2b",
#     "openclip_h14_laion2b",
#     "openclip_h14_378_dfn5b",
#     "openclip_convnext_xxlarge"
# )

# $priorCsv = "domainnet_probe_results\probe_baselines_combined\$trainDomain\cross_target_summary\teacher_mean_metrics_domainnet_${trainDomain}_combined.csv"

# foreach ($targetDomain in $targetDomains) {

#     Write-Host "==============================="
#     Write-Host "RankMe target: $targetDomain"
#     Write-Host "==============================="

#     $argsList = @()

#     foreach ($t in $teachers) {

#         $feat  = "teacher_npzs_domainnet\domainnet__${t}__${targetDomain}.npz"
#         $probe = "quickdraw\domainnet\$trainDomain\$t\probe_outputs_$targetDomain.npz"

#         if (-not (Test-Path $feat)) {
#             throw "Missing feature NPZ: $feat"
#         }
#         if (-not (Test-Path $probe)) {
#             throw "Missing probe NPZ: $probe"
#         }

#         $argsList += "--feature"; $argsList += "${t}=${feat}"
#         $argsList += "--probe";   $argsList += "${t}=${probe}"
#     }

#     if (-not (Test-Path $priorCsv)) {
#         throw "Missing prior CSV: $priorCsv"
#     }

#     python scripts\DomainNet\rankme_router_domainnet.py `
#         @argsList `
#         --outdir "domainnet_probe_results\rankme_delta\$trainDomain\$targetDomain" `
#         --tag "domainnet_${trainDomain}_${targetDomain}" `
#         --anchor_mode concat_all `
#         --k_neighbors 25 `
#         --score_mode hybrid `
#         --alpha_rankme 2.0 `
#         --beta_conf 1.0 `
#         --gamma_prior 0.5 `
#         --weight_temperature 1.0 `
#         --min_teacher_weight 0.02 `
#         --ensemble_deviation_lambda 0.5 `
#         --global_prior_csv $priorCsv
# }