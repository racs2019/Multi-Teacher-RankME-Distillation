#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def parse_named_path(item: str) -> Tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"Expected teacher=path, got: {item}")
    teacher, path = item.split("=", 1)
    teacher = teacher.strip()
    path = path.strip()
    if not teacher or not path:
        raise ValueError(f"Invalid named path: {item}")
    return teacher, path


def get_first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


# ============================================================
# Math helpers
# ============================================================


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=axis, keepdims=True), 1e-12, None)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(denom, eps, None)


def entropy_from_proba(proba: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(proba, eps, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def jsd_to_mean(proba_stack: np.ndarray) -> np.ndarray:
    mean_p = np.clip(proba_stack.mean(axis=0), 1e-12, 1.0)
    h_mean = entropy_from_proba(mean_p, axis=1)
    h_each = entropy_from_proba(np.clip(proba_stack, 1e-12, 1.0), axis=2).mean(axis=0)
    return np.clip(h_mean - h_each, 0.0, None)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def parse_float_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float")
    return vals


def parse_int_list(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one int")
    return vals


def normalize_scores(scores: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """
    scores: [M, N], higher = better.
    Normalized across teachers per sample.
    """
    scores = np.asarray(scores, dtype=np.float32)

    if mode == "none":
        return scores

    if mode == "zscore":
        mu = scores.mean(axis=0, keepdims=True)
        sd = scores.std(axis=0, keepdims=True)
        return (scores - mu) / np.clip(sd, 1e-6, None)

    if mode == "minmax":
        lo = scores.min(axis=0, keepdims=True)
        hi = scores.max(axis=0, keepdims=True)
        return (scores - lo) / np.clip(hi - lo, 1e-6, None)

    if mode == "rank":
        m, n = scores.shape
        out = np.zeros_like(scores, dtype=np.float32)
        order = np.argsort(scores, axis=0)
        for j in range(n):
            out[order[:, j], j] = np.arange(m, dtype=np.float32)
        return out / max(float(m - 1), 1.0)

    raise ValueError(f"Unknown score normalization: {mode}")


def weighted_prediction_from_scores(
    proba_stack: np.ndarray,
    scores: np.ndarray,
    temperature: float,
    score_normalize: str,
) -> Tuple[np.ndarray, np.ndarray]:
    s = normalize_scores(scores, score_normalize)
    weights = softmax_np(s / max(float(temperature), 1e-8), axis=0).astype(np.float32)
    out = (weights[:, :, None] * proba_stack).sum(axis=0)
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out.astype(np.float32), weights


def delta_refine(base_proba: np.ndarray, routed_proba: np.ndarray, lam: float) -> np.ndarray:
    out = (1.0 - lam) * base_proba + lam * routed_proba
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out.astype(np.float32)


# ============================================================
# Alignment
# ============================================================


def extract_standardized_arrays(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    y_true = get_first_present(data, ["labels", "y_true", "targets"])
    if y_true is None:
        raise ValueError(f"{name}: missing labels")
    y_true = np.asarray(y_true, dtype=np.int64)

    paths = get_first_present(data, ["paths", "image_paths", "filenames"])
    if paths is None:
        raise ValueError(f"{name}: missing paths")
    paths = np.asarray(paths, dtype=object)

    proba = get_first_present(data, ["proba", "probs", "probabilities"])
    logits = get_first_present(data, ["logits"])

    if proba is not None:
        proba = np.asarray(proba, dtype=np.float32)
    elif logits is not None:
        logits = np.asarray(logits, dtype=np.float32)
        proba = softmax_np(logits, axis=1).astype(np.float32)
    else:
        raise ValueError(f"{name}: missing proba/logits")

    if proba.ndim != 2:
        raise ValueError(f"{name}: expected proba/logits to become [N,C], got {proba.shape}")

    feats = get_first_present(data, ["feats", "features", "x_feats", "embeddings"])
    if feats is not None:
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(f"{name}: expected feats [N,D], got {feats.shape}")

    return {"y_true": y_true, "paths": paths, "proba": proba, "feats": feats}


def attach_features_by_path(
    standardized: Dict[str, Dict[str, Any]],
    teacher_to_feature_data: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    if not teacher_to_feature_data:
        return standardized

    out: Dict[str, Dict[str, Any]] = {}
    for teacher, arr in standardized.items():
        cur = dict(arr)
        if teacher not in teacher_to_feature_data:
            out[teacher] = cur
            continue

        feat_data = extract_standardized_arrays(teacher_to_feature_data[teacher], teacher)
        feat_paths = np.asarray(feat_data["paths"], dtype=object)
        feat_feats = feat_data["feats"]
        if feat_feats is None:
            out[teacher] = cur
            continue

        path_to_idx = {str(p): i for i, p in enumerate(feat_paths)}
        wanted = [str(p) for p in cur["paths"]]

        missing = [p for p in wanted if p not in path_to_idx]
        if missing:
            raise ValueError(f"{teacher}: missing {len(missing)} paths in feature NPZ; examples={missing[:5]}")

        indices = np.asarray([path_to_idx[p] for p in wanted], dtype=np.int64)
        cur["feats"] = np.asarray(feat_feats, dtype=np.float32)[indices]
        out[teacher] = cur

    return out


def ensure_alignment_and_standardize(
    teacher_to_data: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    teacher_names = sorted(teacher_to_data.keys())
    if not teacher_names:
        raise ValueError("No teacher outputs loaded.")

    standardized = {t: extract_standardized_arrays(teacher_to_data[t], t) for t in teacher_names}

    ref_t = teacher_names[0]
    ref = standardized[ref_t]
    ref_y = ref["y_true"]
    ref_paths = ref["paths"]
    ref_shape = ref["proba"].shape

    for t in teacher_names[1:]:
        cur = standardized[t]
        if not np.array_equal(cur["y_true"], ref_y):
            raise ValueError(f"y_true mismatch: {t}")
        if not np.array_equal(cur["paths"], ref_paths):
            raise ValueError(f"path ordering mismatch: {t}")
        if cur["proba"].shape != ref_shape:
            raise ValueError(f"proba shape mismatch: {t}: {cur['proba'].shape} vs {ref_shape}")

    return standardized, {
        "y_true": ref_y,
        "paths": ref_paths,
        "num_classes": ref_shape[1],
        "num_samples": len(ref_y),
    }


def build_teacher_arrays(
    standardized: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, Optional[List[np.ndarray]]]:
    teacher_names = sorted(standardized.keys())
    proba_list = []
    feat_list = []

    for t in teacher_names:
        proba_list.append(np.asarray(standardized[t]["proba"], dtype=np.float32))
        feats = standardized[t]["feats"]
        if feats is not None:
            feat_list.append(l2_normalize(np.asarray(feats, dtype=np.float32), axis=1))

    proba_stack = np.stack(proba_list, axis=0)  # [M,N,C]

    if len(feat_list) != len(teacher_names):
        return teacher_names, proba_stack, None
    return teacher_names, proba_stack, feat_list


def build_anchor_features(feat_list: List[np.ndarray]) -> np.ndarray:
    return l2_normalize(np.concatenate(feat_list, axis=1), axis=1)


# ============================================================
# Neighborhoods
# ============================================================


def compute_neighbor_indices(
    anchor_features: np.ndarray,
    k_neighbors: int,
    metric: str = "euclidean",
    exclude_self: bool = True,
) -> np.ndarray:
    n = anchor_features.shape[0]
    k_neighbors = int(k_neighbors)

    if exclude_self:
        k_eff = min(max(2, k_neighbors + 1), n)
    else:
        k_eff = min(max(1, k_neighbors), n)

    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nbrs.fit(anchor_features)
    raw_idx = nbrs.kneighbors(anchor_features, return_distance=False).astype(np.int32)

    if not exclude_self:
        return raw_idx

    idx = np.zeros((n, min(k_neighbors, max(n - 1, 1))), dtype=np.int32)
    for i in range(n):
        row = [j for j in raw_idx[i].tolist() if j != i]
        if not row:
            row = [i]
        row = row[: idx.shape[1]]
        if len(row) < idx.shape[1]:
            row += [row[-1]] * (idx.shape[1] - len(row))
        idx[i] = np.asarray(row, dtype=np.int32)

    return idx


# ============================================================
# Baseline score functions
# ============================================================


def uniform_ensemble(proba_stack: np.ndarray) -> np.ndarray:
    p = proba_stack.mean(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def confidence_scores(proba_stack: np.ndarray) -> np.ndarray:
    return proba_stack.max(axis=2)


def entropy_scores(proba_stack: np.ndarray) -> np.ndarray:
    return -entropy_from_proba(proba_stack, axis=2)


def agreement_scores(proba_stack: np.ndarray) -> np.ndarray:
    mean_proba = np.clip(proba_stack.mean(axis=0, keepdims=True), 1e-12, 1.0)
    p = np.clip(proba_stack, 1e-12, 1.0)
    kl = (p * (np.log(p) - np.log(mean_proba))).sum(axis=2)
    return -kl


# ============================================================
# Local reliability score functions
# ============================================================


def local_predictive_stability_kl_scores(
    proba_stack: np.ndarray,
    neighbor_idx: np.ndarray,
) -> np.ndarray:
    """
    LPS-KL.
    score_m(i) = -KL(p_m(i) || mean_{j in N(i)} p_m(j)).
    """
    m, n, _ = proba_stack.shape
    scores = np.zeros((m, n), dtype=np.float32)
    p = np.clip(proba_stack, 1e-12, 1.0)

    for i in range(n):
        neigh = neighbor_idx[i]
        neigh_mean = np.clip(p[:, neigh, :].mean(axis=1), 1e-12, 1.0)
        kl = (p[:, i, :] * (np.log(p[:, i, :]) - np.log(neigh_mean))).sum(axis=1)
        scores[:, i] = -kl.astype(np.float32)

    return scores


def local_predictive_stability_l2_scores(
    proba_stack: np.ndarray,
    neighbor_idx: np.ndarray,
) -> np.ndarray:
    """
    LPS-L2.
    score_m(i) = -mean_{j in N(i)} ||p_m(i) - p_m(j)||_2^2.
    """
    m, n, _ = proba_stack.shape
    scores = np.zeros((m, n), dtype=np.float32)

    for i in range(n):
        neigh = neighbor_idx[i]
        diff = proba_stack[:, neigh, :] - proba_stack[:, i:i + 1, :]
        l2 = np.mean(np.sum(diff * diff, axis=2), axis=1)
        scores[:, i] = -l2.astype(np.float32)

    return scores


def local_consensus_scores(
    proba_stack: np.ndarray,
    neighbor_idx: np.ndarray,
) -> np.ndarray:
    """
    Local consensus with uniform ensemble over target neighborhood.
    score_m(i) = mean_{j in N(i)} <p_m(j), p_ens(j)>.
    """
    m, n, _ = proba_stack.shape
    scores = np.zeros((m, n), dtype=np.float32)
    ens = proba_stack.mean(axis=0)

    for i in range(n):
        neigh = neighbor_idx[i]
        sims = np.sum(proba_stack[:, neigh, :] * ens[None, neigh, :], axis=2)
        scores[:, i] = sims.mean(axis=1).astype(np.float32)

    return scores


def local_margin_scores(
    proba_stack: np.ndarray,
    neighbor_idx: np.ndarray,
) -> np.ndarray:
    """Mean top-1 minus top-2 prediction margin over target neighborhood."""
    m, n, _ = proba_stack.shape
    scores = np.zeros((m, n), dtype=np.float32)

    for i in range(n):
        neigh = neighbor_idx[i]
        p = proba_stack[:, neigh, :]
        top2 = np.partition(p, -2, axis=2)[:, :, -2:]
        margin = top2[:, :, 1] - top2[:, :, 0]
        scores[:, i] = margin.mean(axis=1).astype(np.float32)

    return scores


def target_typicality_scores(
    feat_list: List[np.ndarray],
    neighbor_idx: np.ndarray,
) -> np.ndarray:
    """Teacher-specific target typicality: mean cosine similarity to anchor-neighborhood samples."""
    m = len(feat_list)
    n = feat_list[0].shape[0]
    scores = np.zeros((m, n), dtype=np.float32)

    for teacher_idx, feats in enumerate(feat_list):
        feats = l2_normalize(feats, axis=1)
        for i in range(n):
            neigh = neighbor_idx[i]
            sims = feats[neigh] @ feats[i]
            scores[teacher_idx, i] = float(sims.mean())

    return scores


# ============================================================
# Saving
# ============================================================


def safe_method_name(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace(",", "_")
    )


def save_proba_npz(
    out_path: Path,
    y_true: np.ndarray,
    proba: np.ndarray,
    paths: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        y_true=y_true.astype(np.int64),
        y_pred=proba.argmax(axis=1).astype(np.int64),
        proba=proba.astype(np.float32),
        paths=np.asarray(paths, dtype=object),
        meta_json=json.dumps(meta),
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LPS/local reliability sweep for teacher ensembles under domain shift."
    )

    parser.add_argument("--feature", action="append", required=True, help="teacher=feature_npz")
    parser.add_argument("--probe", action="append", required=True, help="teacher=probe_npz")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="lps_reliability_sweep")
    parser.add_argument("--save_method_outputs", action="store_true")

    parser.add_argument("--k_values", type=str, default="10,20,50")
    parser.add_argument("--neighbor_metric", type=str, default="euclidean", choices=["cosine", "euclidean"])
    parser.add_argument("--exclude_self", action="store_true", default=True)

    parser.add_argument("--sweep_temperatures", type=str, default="0.5,1.0,2.0,4.0")
    parser.add_argument("--sweep_lambdas", type=str, default="0.10,0.20,0.30,0.50")
    parser.add_argument(
        "--methods",
        type=str,
        default="lps_l2,lps_kl,local_consensus",
        help="Comma-separated subset of: lps_kl,lps_l2,local_consensus,local_margin,target_typicality",
    )
    parser.add_argument("--score_normalize", type=str, default="zscore", choices=["none", "zscore", "minmax", "rank"])
    parser.add_argument("--delta_base", type=str, default="uniform", choices=["uniform", "agreement"])
    args = parser.parse_args()

    k_values = parse_int_list(args.k_values)
    temperatures = parse_float_list(args.sweep_temperatures)
    lambdas = parse_float_list(args.sweep_lambdas)
    selected_methods = {x.strip() for x in args.methods.split(",") if x.strip()}

    valid_methods = {"lps_kl", "lps_l2", "local_consensus", "local_margin", "target_typicality"}
    unknown = selected_methods - valid_methods
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    teacher_to_probe: Dict[str, Dict[str, Any]] = {}
    for item in args.probe:
        t, path = parse_named_path(item)
        if t in teacher_to_probe:
            raise ValueError(f"Duplicate probe teacher: {t}")
        teacher_to_probe[t] = load_npz_dict(path)

    teacher_to_feat: Dict[str, Dict[str, Any]] = {}
    for item in args.feature:
        t, path = parse_named_path(item)
        if t in teacher_to_feat:
            raise ValueError(f"Duplicate feature teacher: {t}")
        teacher_to_feat[t] = load_npz_dict(path)

    standardized, info = ensure_alignment_and_standardize(teacher_to_probe)
    standardized = attach_features_by_path(standardized, teacher_to_feat)
    teacher_names, proba_stack, feat_list = build_teacher_arrays(standardized)
    if feat_list is None:
        raise RuntimeError("All teachers need feature NPZs for this sweep.")

    y_true = info["y_true"]
    paths = info["paths"]
    anchor_features = build_anchor_features(feat_list)

    ref_meta = teacher_to_probe[teacher_names[0]].get("_meta", {})
    dataset_name = ref_meta.get("dataset_name", "unknown")
    train_domain = ref_meta.get("train_domain", "unknown")
    target_domain = ref_meta.get("target_domain", "unknown")
    split = ref_meta.get("split", "unknown")

    results: List[Dict[str, Any]] = []
    best_by_family: Dict[str, Tuple[float, str, np.ndarray]] = {}

    def add_result(method_name: str, proba: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> None:
        row = {
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain": target_domain,
            "split": split,
            "method": method_name,
            **compute_metrics(y_true, proba),
        }
        if extra:
            row.update(extra)
        results.append(row)

    def consider_best(key: str, method_name: str, proba: np.ndarray) -> None:
        acc = float(accuracy_score(y_true, proba.argmax(axis=1)))
        if key not in best_by_family or acc > best_by_family[key][0]:
            best_by_family[key] = (acc, method_name, proba.astype(np.float32))

    # -------------------------
    # Baselines
    # -------------------------
    uniform_proba = uniform_ensemble(proba_stack)
    add_result("uniform_probability_ensemble", uniform_proba, {"family": "baseline", "mode": "uniform"})
    consider_best("baseline_uniform", "uniform_probability_ensemble", uniform_proba)

    for temp in temperatures:
        for family, score_fn in [
            ("confidence", confidence_scores),
            ("entropy", entropy_scores),
            ("agreement", agreement_scores),
        ]:
            scores = score_fn(proba_stack)
            proba, weights = weighted_prediction_from_scores(proba_stack, scores, temp, args.score_normalize)
            name = f"{family}_weighted_T{temp}"
            add_result(
                name,
                proba,
                {
                    "family": "baseline",
                    "mode": family,
                    "k": np.nan,
                    "temperature": temp,
                    "lambda": np.nan,
                    "score_normalize": args.score_normalize,
                    "mean_max_weight": float(weights.max(axis=0).mean()),
                },
            )
            consider_best(f"baseline_{family}", name, proba)

    agreement_base_scores = agreement_scores(proba_stack)
    agreement_base, _ = weighted_prediction_from_scores(proba_stack, agreement_base_scores, 1.0, args.score_normalize)
    delta_base = uniform_proba if args.delta_base == "uniform" else agreement_base

    # -------------------------
    # Local reliability sweep
    # -------------------------
    for k in k_values:
        print(f"Computing neighbors and local reliability scores for k={k} ...")
        neighbor_idx = compute_neighbor_indices(
            anchor_features,
            k,
            metric=args.neighbor_metric,
            exclude_self=args.exclude_self,
        )

        score_bank: Dict[str, np.ndarray] = {}
        if "lps_l2" in selected_methods:
            score_bank["lps_l2"] = local_predictive_stability_l2_scores(proba_stack, neighbor_idx)
        if "lps_kl" in selected_methods:
            score_bank["lps_kl"] = local_predictive_stability_kl_scores(proba_stack, neighbor_idx)
        if "local_consensus" in selected_methods:
            score_bank["local_consensus"] = local_consensus_scores(proba_stack, neighbor_idx)
        if "local_margin" in selected_methods:
            score_bank["local_margin"] = local_margin_scores(proba_stack, neighbor_idx)
        if "target_typicality" in selected_methods:
            score_bank["target_typicality"] = target_typicality_scores(feat_list, neighbor_idx)

        for family, scores in score_bank.items():
            for temp in temperatures:
                routed_proba, weights = weighted_prediction_from_scores(
                    proba_stack=proba_stack,
                    scores=scores,
                    temperature=temp,
                    score_normalize=args.score_normalize,
                )
                method_name = f"{family}_k{k}_T{temp}"
                add_result(
                    method_name,
                    routed_proba,
                    {
                        "family": family,
                        "mode": "direct_weighted",
                        "k": k,
                        "temperature": temp,
                        "lambda": np.nan,
                        "score_normalize": args.score_normalize,
                        "mean_max_weight": float(weights.max(axis=0).mean()),
                    },
                )
                consider_best(f"{family}_direct", method_name, routed_proba)

                for lam in lambdas:
                    delta_proba = delta_refine(delta_base, routed_proba, lam)
                    delta_name = f"delta_{args.delta_base}_{family}_k{k}_T{temp}_lam{lam}"
                    add_result(
                        delta_name,
                        delta_proba,
                        {
                            "family": family,
                            "mode": f"delta_{args.delta_base}",
                            "k": k,
                            "temperature": temp,
                            "lambda": lam,
                            "score_normalize": args.score_normalize,
                            "mean_max_weight": float(weights.max(axis=0).mean()),
                        },
                    )
                    consider_best(f"{family}_delta_{args.delta_base}", delta_name, delta_proba)

    # -------------------------
    # Save
    # -------------------------
    result_df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    csv_path = outdir / f"lps_reliability_sweep_{args.tag}.csv"
    result_df.to_csv(csv_path, index=False)

    common_meta = {
        "dataset_name": dataset_name,
        "train_domain": train_domain,
        "target_domain": target_domain,
        "split": split,
        "teacher_names": teacher_names,
        "k_values": k_values,
        "neighbor_metric": args.neighbor_metric,
        "exclude_self": args.exclude_self,
        "temperatures": temperatures,
        "lambdas": lambdas,
        "score_normalize": args.score_normalize,
        "delta_base": args.delta_base,
        "methods": sorted(selected_methods),
    }

    if args.save_method_outputs:
        for family_key, (_, method_name, proba) in best_by_family.items():
            save_proba_npz(
                outdir / f"BEST_{safe_method_name(method_name)}_{args.tag}.npz",
                y_true=y_true,
                proba=proba,
                paths=paths,
                meta={**common_meta, "method": method_name, "family_key": family_key},
            )

    diag_path = outdir / f"lps_reliability_sweep_diag_{args.tag}.npz"
    np.savez_compressed(
        diag_path,
        teacher_names=np.asarray(teacher_names, dtype=object),
        paths=np.asarray(paths, dtype=object),
        y_true=y_true.astype(np.int64),
        disagreement=jsd_to_mean(proba_stack).astype(np.float32),
    )

    print("\n=== LPS/local reliability sweep summary: top 30 ===")
    cols = [
        "method",
        "accuracy",
        "balanced_accuracy",
        "family",
        "mode",
        "k",
        "temperature",
        "lambda",
        "mean_max_weight",
    ]
    existing = [c for c in cols if c in result_df.columns]
    print(result_df[existing].head(30).to_string(index=False))

    print(f"\nSaved summary CSV to: {csv_path}")
    print(f"Saved diagnostics to:  {diag_path}")

    if best_by_family:
        print("\n=== Best by family ===")
        for family_key, (acc, method_name, _) in sorted(
            best_by_family.items(),
            key=lambda x: x[1][0],
            reverse=True,
        ):
            print(f"{family_key:36s} acc={acc:.6f}  {method_name}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Suggested PowerShell call
# ----------------------------------------------------------------------
# $env:OPENBLAS_NUM_THREADS = "8"
# $env:OMP_NUM_THREADS = "8"
# $env:MKL_NUM_THREADS = "8"

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

# foreach ($targetDomain in $targetDomains) {
#     Write-Host "==============================="
#     Write-Host "START target: $targetDomain"
#     Write-Host "==============================="

#     $argsList = @()
#     foreach ($t in $teachers) {
#         $feat  = "teacher_npzs_domainnet\domainnet__${t}__${targetDomain}.npz"
#         $probe = "quickdraw\domainnet\$trainDomain\$t\probe_outputs_$targetDomain.npz"
#         $argsList += "--feature"
#         $argsList += "${t}=${feat}"
#         $argsList += "--probe"
#         $argsList += "${t}=${probe}"
#     }

#     python scripts\DomainNet\lps.py `
#         @argsList `
#         --outdir "domainnet_probe_results\lps_reliability_sweep\$trainDomain\$targetDomain" `
#         --tag "domainnet_${trainDomain}_${targetDomain}" `
#         --k_values "10,20,50" `
#         --neighbor_metric euclidean `
#         --sweep_temperatures "0.5,1.0,2.0,4.0" `
#         --sweep_lambdas "0.10,0.20,0.30,0.50" `
#         --methods "lps_l2,lps_kl,local_consensus" `
#         --score_normalize zscore `
#         --delta_base uniform `
#         --save_method_outputs
# }
