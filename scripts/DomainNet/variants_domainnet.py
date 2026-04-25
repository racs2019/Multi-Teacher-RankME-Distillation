#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
# Basic math
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
    """
    proba_stack: [M, N, C]
    returns per-sample disagreement: [N]
    """
    mean_p = np.clip(proba_stack.mean(axis=0), 1e-12, 1.0)
    h_mean = entropy_from_proba(mean_p, axis=1)
    h_each = entropy_from_proba(np.clip(proba_stack, 1e-12, 1.0), axis=2).mean(axis=0)
    return np.clip(h_mean - h_each, 0.0, None)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
    }


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-12)


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

    feats = get_first_present(data, ["feats", "features", "x_feats", "embeddings"])
    if feats is not None:
        feats = np.asarray(feats, dtype=np.float32)

    return {"y_true": y_true, "paths": paths, "proba": proba, "feats": feats}


def attach_features_by_path(
    standardized: Dict[str, Dict[str, Any]],
    teacher_to_feature_data: Dict[str, Dict[str, Any]] | None,
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
        idx = []
        for p in wanted:
            if p not in path_to_idx:
                raise ValueError(f"{teacher}: missing path in feature NPZ: {p}")
            idx.append(path_to_idx[p])

        cur["feats"] = np.asarray(feat_feats, dtype=np.float32)[np.asarray(idx, dtype=np.int64)]
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
            raise ValueError(f"proba shape mismatch: {t}")

    return standardized, {
        "y_true": ref_y,
        "paths": ref_paths,
        "num_classes": ref_shape[1],
        "num_samples": len(ref_y),
    }


def build_teacher_arrays(
    standardized: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], np.ndarray, List[np.ndarray] | None]:
    teacher_names = sorted(standardized.keys())
    proba_list = []
    feat_list = []

    for t in teacher_names:
        proba_list.append(np.asarray(standardized[t]["proba"], dtype=np.float32))
        feats = standardized[t]["feats"]
        if feats is not None:
            feat_list.append(l2_normalize(np.asarray(feats, dtype=np.float32), axis=1))

    proba_stack = np.stack(proba_list, axis=0)  # [M, N, C]
    if len(feat_list) != len(teacher_names):
        return teacher_names, proba_stack, None
    return teacher_names, proba_stack, feat_list


def build_anchor_features(feat_list: List[np.ndarray]) -> np.ndarray:
    return l2_normalize(np.concatenate(feat_list, axis=1), axis=1)


# ============================================================
# Neighborhood tools
# ============================================================


def compute_neighbor_indices(anchor_features: np.ndarray, k_neighbors: int, metric: str = "euclidean") -> np.ndarray:
    n = anchor_features.shape[0]
    k_eff = min(max(2, k_neighbors), n)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nbrs.fit(anchor_features)
    return nbrs.kneighbors(anchor_features, return_distance=False).astype(np.int32)


def anchor_density(anchor_features: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    n = anchor_features.shape[0]
    dens = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neigh = anchor_features[neighbor_idx[i]]
        dens[i] = float(np.mean(neigh @ anchor_features[i]))
    return dens


def local_scale_consistency(local_stack: np.ndarray) -> np.ndarray:
    """
    local_stack: [S, N, C]
    Returns per-sample consistency across neighborhood scales.
    Higher = more trustworthy local signal.
    """
    scale_disagreement = jsd_to_mean(local_stack)  # [N]
    return 1.0 - normalize_01(scale_disagreement)


# ============================================================
# Baselines
# ============================================================


def uniform_ensemble(proba_stack: np.ndarray) -> np.ndarray:
    p = proba_stack.mean(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def agreement_weighted_ensemble(proba_stack: np.ndarray, temperature: float = 1.0) -> np.ndarray:
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
    p = (weights[:, :, None] * proba_stack).sum(axis=0)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


# ============================================================
# Strong local router
# ============================================================


def knn_teacher_agreement_router(
    proba_stack: np.ndarray,
    neighbor_idx: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Teacher-specific local agreement:
    each teacher is scored by how well its current prediction matches
    its own neighborhood-average prediction.
    """
    m, n, _ = proba_stack.shape
    local_scores = np.zeros((m, n), dtype=np.float32)

    for i in range(n):
        neigh = neighbor_idx[i]
        neigh_mean = proba_stack[:, neigh, :].mean(axis=1)  # [M, C]
        kl = (
            proba_stack[:, i, :]
            * (
                np.log(np.clip(proba_stack[:, i, :], 1e-12, 1.0))
                - np.log(np.clip(neigh_mean, 1e-12, 1.0))
            )
        ).sum(axis=1)
        local_scores[:, i] = -kl

    weights = softmax_np(local_scores / max(temperature, 1e-8), axis=0)
    out = (weights[:, :, None] * proba_stack).sum(axis=0)
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out, weights


def multiscale_local_router(
    proba_stack: np.ndarray,
    anchor_features: np.ndarray,
    k_values: List[int],
    metric: str = "euclidean",
    temperature: float = 1.0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Returns:
      per_scale_outputs: method_name -> [N,C]
      local_stack: [S,N,C]
      neighbor_indices_by_k: k -> [N,k]
      density_by_k: k -> [N]
    """
    per_scale_outputs: Dict[str, np.ndarray] = {}
    local_list = []
    neighbor_indices_by_k: Dict[int, np.ndarray] = {}
    density_by_k: Dict[int, np.ndarray] = {}

    for k in k_values:
        idx = compute_neighbor_indices(anchor_features, k, metric=metric)
        density_by_k[k] = anchor_density(anchor_features, idx)
        neighbor_indices_by_k[k] = idx
        out, _ = knn_teacher_agreement_router(proba_stack, idx, temperature=temperature)
        per_scale_outputs[f"knn_teacher_agreement_k{k}"] = out
        local_list.append(out)

    local_stack = np.stack(local_list, axis=0)  # [S,N,C]
    return per_scale_outputs, local_stack, neighbor_indices_by_k, density_by_k


# ============================================================
# Paper methods
# ============================================================


def adaptive_global_local_blend(
    global_proba: np.ndarray,
    local_proba: np.ndarray,
    disagreement: np.ndarray,
    density: np.ndarray,
    scale_consistency: np.ndarray,
    gate_scale: float = 0.50,
    density_weight: float = 0.20,
    consistency_weight: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reliability-conditioned interpolation between global and local routing.
    """
    d_norm = normalize_01(disagreement)
    dens_norm = normalize_01(density)
    cons_norm = normalize_01(scale_consistency)

    local_trust = (
        (1.0 - density_weight - consistency_weight) * d_norm
        + density_weight * dens_norm
        + consistency_weight * cons_norm
    )

    gate = np.clip(gate_scale * local_trust, 0.0, 1.0).astype(np.float32)

    out = (1.0 - gate[:, None]) * global_proba + gate[:, None] * local_proba
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out, gate


def adaptive_tri_blend(
    uniform_proba: np.ndarray,
    global_proba: np.ndarray,
    local_proba: np.ndarray,
    disagreement: np.ndarray,
    density: np.ndarray,
    scale_consistency: np.ndarray,
    tri_temperature: float = 1.0,
    easy_weight: float = 1.0,
    global_weight: float = 1.0,
    local_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3-way adaptive blend:
      - uniform for easy / in-domain
      - agreement for severe / noisy shift
      - local for moderate / trustworthy local structure
    Returns:
      out: [N,C]
      weights: [N,3] corresponding to [uniform, global, local]
    """
    d_norm = normalize_01(disagreement)
    dens_norm = normalize_01(density)
    cons_norm = normalize_01(scale_consistency)

    easy_score = easy_weight * (1.0 - d_norm)
    global_score = global_weight * ((1.0 - dens_norm) + (1.0 - cons_norm)) / 2.0
    local_score = local_weight * (d_norm + dens_norm + cons_norm) / 3.0

    logits = np.stack([easy_score, global_score, local_score], axis=1)  # [N,3]
    weights = softmax_np(logits / max(tri_temperature, 1e-8), axis=1)

    out = (
        weights[:, [0]] * uniform_proba
        + weights[:, [1]] * global_proba
        + weights[:, [2]] * local_proba
    )
    out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out, weights


# ============================================================
# Save helper
# ============================================================


def save_proba_npz(out_path: Path, y_true: np.ndarray, proba: np.ndarray, paths: np.ndarray, meta: Dict[str, Any]) -> None:
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


def parse_k_values(s: str) -> List[int]:
    ks = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not ks:
        raise ValueError("k_values must contain at least one integer")
    return ks


def main() -> None:
    parser = argparse.ArgumentParser(description="Final multi-scale adaptive global-local routing harness for DomainNet.")
    parser.add_argument("--feature", action="append", required=True, help="teacher=feature_npz")
    parser.add_argument("--probe", action="append", required=True, help="teacher=probe_npz")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="domainnet_multiscale_adaptive")
    parser.add_argument("--save_method_outputs", action="store_true")

    parser.add_argument("--k_values", type=str, default="10,25,50")
    parser.add_argument("--neighbor_metric", type=str, default="euclidean", choices=["cosine", "euclidean"])
    parser.add_argument("--temperature", type=float, default=1.0)

    # 2-way blend settings
    parser.add_argument("--gate_scale", type=float, default=0.50)
    parser.add_argument("--density_weight", type=float, default=0.20)
    parser.add_argument("--consistency_weight", type=float, default=0.30)

    # 3-way blend settings
    parser.add_argument("--tri_temperature", type=float, default=1.0)
    parser.add_argument("--easy_weight", type=float, default=1.0)
    parser.add_argument("--global_weight", type=float, default=1.0)
    parser.add_argument("--local_weight", type=float, default=1.0)

    args = parser.parse_args()
    k_values = parse_k_values(args.k_values)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    teacher_to_probe: Dict[str, Dict[str, Any]] = {}
    for item in args.probe:
        t, path = parse_named_path(item)
        teacher_to_probe[t] = load_npz_dict(path)

    teacher_to_feat: Dict[str, Dict[str, Any]] = {}
    for item in args.feature:
        t, path = parse_named_path(item)
        teacher_to_feat[t] = load_npz_dict(path)

    standardized, info = ensure_alignment_and_standardize(teacher_to_probe)
    standardized = attach_features_by_path(standardized, teacher_to_feat)
    teacher_names, proba_stack, feat_list = build_teacher_arrays(standardized)
    if feat_list is None:
        raise RuntimeError("All teachers need feature NPZs for this harness.")

    y_true = info["y_true"]
    paths = info["paths"]

    anchor_features = build_anchor_features(feat_list)
    disagreement = jsd_to_mean(proba_stack)

    # baselines
    uniform_proba = uniform_ensemble(proba_stack)
    agreement_proba = agreement_weighted_ensemble(proba_stack, temperature=args.temperature)

    # multi-scale local routers
    per_scale_outputs, local_stack, neighbor_indices_by_k, density_by_k = multiscale_local_router(
        proba_stack=proba_stack,
        anchor_features=anchor_features,
        k_values=k_values,
        metric=args.neighbor_metric,
        temperature=args.temperature,
    )

    # multiscale local mean
    local_multiscale_mean = local_stack.mean(axis=0)
    local_multiscale_mean = local_multiscale_mean / np.clip(local_multiscale_mean.sum(axis=1, keepdims=True), 1e-12, None)

    # use middle scale density if available, otherwise average
    mid_k = k_values[len(k_values) // 2]
    density = density_by_k[mid_k] if mid_k in density_by_k else np.mean(np.stack(list(density_by_k.values()), axis=0), axis=0)
    scale_consistency = local_scale_consistency(local_stack)

    # paper methods
    adaptive_2way, gate_2way = adaptive_global_local_blend(
        global_proba=agreement_proba,
        local_proba=local_multiscale_mean,
        disagreement=disagreement,
        density=density,
        scale_consistency=scale_consistency,
        gate_scale=args.gate_scale,
        density_weight=args.density_weight,
        consistency_weight=args.consistency_weight,
    )

    adaptive_3way, tri_weights = adaptive_tri_blend(
        uniform_proba=uniform_proba,
        global_proba=agreement_proba,
        local_proba=local_multiscale_mean,
        disagreement=disagreement,
        density=density,
        scale_consistency=scale_consistency,
        tri_temperature=args.tri_temperature,
        easy_weight=args.easy_weight,
        global_weight=args.global_weight,
        local_weight=args.local_weight,
    )

    results: List[Dict[str, Any]] = []
    outputs: Dict[str, np.ndarray] = {}

    def add_result(method_name: str, proba: np.ndarray, extra: Dict[str, Any] | None = None):
        y_pred = proba.argmax(axis=1)
        row = {"method": method_name, **compute_metrics(y_true, y_pred)}
        if extra:
            row.update(extra)
        results.append(row)
        outputs[method_name] = proba

    # references
    add_result("uniform_probability_ensemble", uniform_proba)
    add_result("agreement_weighted_ensemble", agreement_proba)

    # single-scale local routers
    for method_name, proba in per_scale_outputs.items():
        add_result(method_name, proba)

    # multiscale local
    add_result("knn_teacher_agreement_multiscale_mean", local_multiscale_mean)

    # final methods
    add_result(
        "adaptive_global_local_blend",
        adaptive_2way,
        {
            "gate_scale": args.gate_scale,
            "density_weight": args.density_weight,
            "consistency_weight": args.consistency_weight,
            "mean_gate": float(gate_2way.mean()),
        },
    )

    add_result(
        "adaptive_tri_blend",
        adaptive_3way,
        {
            "tri_temperature": args.tri_temperature,
            "easy_weight": args.easy_weight,
            "global_weight": args.global_weight,
            "local_weight": args.local_weight,
            "mean_uniform_weight": float(tri_weights[:, 0].mean()),
            "mean_global_weight": float(tri_weights[:, 1].mean()),
            "mean_local_weight": float(tri_weights[:, 2].mean()),
        },
    )

    ref_meta = teacher_to_probe[teacher_names[0]].get("_meta", {})
    dataset_name = ref_meta.get("dataset_name", "domainnet")
    train_domain = ref_meta.get("train_domain", "unknown")
    target_domain = ref_meta.get("target_domain", "unknown")
    split = ref_meta.get("split", "unknown")

    result_df = pd.DataFrame(results)
    result_df.insert(0, "dataset_name", dataset_name)
    result_df.insert(1, "train_domain", train_domain)
    result_df.insert(2, "target_domain", target_domain)
    result_df.insert(3, "split", split)
    result_df = result_df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    csv_path = outdir / f"multiscale_adaptive_{args.tag}.csv"
    result_df.to_csv(csv_path, index=False)

    if args.save_method_outputs:
        common_meta = {
            "dataset_name": dataset_name,
            "train_domain": train_domain,
            "target_domain": target_domain,
            "split": split,
            "teacher_names": teacher_names,
            "k_values": k_values,
            "neighbor_metric": args.neighbor_metric,
            "temperature": args.temperature,
        }

        for method_name, proba in outputs.items():
            save_proba_npz(
                outdir / f"{method_name}_{args.tag}.npz",
                y_true=y_true,
                proba=proba,
                paths=paths,
                meta={**common_meta, "method": method_name},
            )

    diag_path = outdir / f"multiscale_adaptive_diag_{args.tag}.npz"
    np.savez_compressed(
        diag_path,
        teacher_names=np.asarray(teacher_names, dtype=object),
        disagreement=disagreement.astype(np.float32),
        density=density.astype(np.float32),
        scale_consistency=scale_consistency.astype(np.float32),
        gate_2way=gate_2way.astype(np.float32),
        tri_weights=tri_weights.astype(np.float32),
        paths=np.asarray(paths, dtype=object),
        y_true=y_true.astype(np.int64),
    )

    print("\n=== Multi-scale adaptive summary ===")
    print(result_df[["method", "accuracy", "balanced_accuracy"]].to_string(index=False))
    print(f"\nSaved summary CSV to: {csv_path}")
    print(f"Saved diagnostics to:  {diag_path}")


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

#     python scripts\DomainNet\variants_domainnet.py `
#         @argsList `
#         --outdir "domainnet_probe_results\multiscale_adaptive\$trainDomain\$targetDomain" `
#         --tag "domainnet_${trainDomain}_${targetDomain}" `
#         --k_values "10,25,50" `
#         --neighbor_metric euclidean `
#         --temperature 1.0 `
#         --gate_scale 0.50 `
#         --density_weight 0.20 `
#         --consistency_weight 0.30 `
#         --tri_temperature 1.0 `
#         --easy_weight 1.0 `
#         --global_weight 1.0 `
#         --local_weight 1.0 `
#         --save_method_outputs
# }