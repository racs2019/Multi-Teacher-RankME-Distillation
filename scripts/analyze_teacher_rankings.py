#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def load_npz_dict(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}

    if "meta_json" in out:
        try:
            meta_json = out["meta_json"]
            if isinstance(meta_json, np.ndarray):
                meta_json = meta_json.item()
            out["_meta"] = json.loads(meta_json)
        except Exception:
            out["_meta"] = {}
    else:
        out["_meta"] = {}

    return out


def compute_cluster_entropy(labels: np.ndarray, num_classes: int, eps: float = 1e-12) -> float:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs + eps)).sum())


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

    for t1, t2 in combinations(names, 2):
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


def parse_teacher_arg(item: str) -> Tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"--teacher must be alias=path, got: {item}")
    name, path = item.split("=", 1)
    return name, path


def infer_domain_and_teacher(alias: str, data: Dict[str, np.ndarray], path: str) -> Tuple[str, str]:
    meta = data.get("_meta", {})
    domain = meta.get("domain", None)
    teacher = meta.get("teacher_name", None)

    if teacher is None:
        teacher = alias

    if domain is None:
        stem = Path(path).stem
        parts = stem.split("__")
        domain = parts[-1] if len(parts) >= 2 else "unknown_domain"

    return domain, teacher


def get_preds(data: Dict[str, np.ndarray]) -> np.ndarray:
    if "preds" in data:
        return data["preds"].astype(np.int64)
    if "logits" in data:
        return data["logits"].argmax(axis=1).astype(np.int64)
    raise ValueError("NPZ has neither 'preds' nor 'logits'")


def ensure_domain_alignment(domain_to_teacher_data: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> None:
    for domain, teacher_map in domain_to_teacher_data.items():
        teacher_names = sorted(teacher_map.keys())
        if not teacher_names:
            continue

        ref = teacher_map[teacher_names[0]]
        ref_labels = ref["labels"].astype(np.int64)
        ref_paths = ref["paths"]

        for teacher in teacher_names[1:]:
            cur = teacher_map[teacher]
            cur_labels = cur["labels"].astype(np.int64)
            cur_paths = cur["paths"]

            if len(cur_labels) != len(ref_labels):
                raise ValueError(
                    f"Sample count mismatch in domain={domain}: "
                    f"{teacher} has {len(cur_labels)} vs {teacher_names[0]} has {len(ref_labels)}"
                )

            if not np.array_equal(cur_labels, ref_labels):
                raise ValueError(f"Label mismatch in domain={domain} between teachers")

            if len(cur_paths) != len(ref_paths):
                raise ValueError(f"Path count mismatch in domain={domain} between teachers")

            if not np.array_equal(cur_paths, ref_paths):
                raise ValueError(f"Path ordering mismatch in domain={domain} between teachers")


def build_neutral_anchor(teachers: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    pieces = []
    for teacher_name in sorted(teachers.keys()):
        feats = teachers[teacher_name]["feats"].astype(np.float32)
        feats = l2_normalize(feats)
        pieces.append(feats)
    return np.concatenate(pieces, axis=1).astype(np.float32)


def run_cluster_analysis_for_domain(
    domain: str,
    teacher_map: Dict[str, Dict[str, np.ndarray]],
    k: int,
    seed: int,
):
    teacher_names = sorted(teacher_map.keys())
    ref_teacher = teacher_names[0]

    labels = teacher_map[ref_teacher]["labels"].astype(np.int64)
    paths = teacher_map[ref_teacher]["paths"]
    num_classes = int(labels.max()) + 1

    anchor_feats = build_neutral_anchor(teacher_map)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    clusters = kmeans.fit_predict(anchor_feats)

    membership_df = pd.DataFrame({
        "domain": domain,
        "path": paths,
        "label": labels,
        "cluster": clusters,
    })

    cluster_meta_rows = []
    for c in range(k):
        mask = clusters == c
        y_c = labels[mask]
        if len(y_c) == 0:
            continue

        cluster_meta_rows.append({
            "domain": domain,
            "cluster": c,
            "size": int(mask.sum()),
            "label_entropy": compute_cluster_entropy(y_c, num_classes),
            "n_unique_labels": int(np.unique(y_c).size),
        })
    cluster_meta_df = pd.DataFrame(cluster_meta_rows).sort_values("cluster").reset_index(drop=True)

    results = []
    for teacher_name in teacher_names:
        data = teacher_map[teacher_name]
        y = data["labels"].astype(np.int64)
        preds = get_preds(data)
        overall_acc = accuracy_score(y, preds)

        for c in range(k):
            mask = clusters == c
            if mask.sum() == 0:
                continue

            acc = accuracy_score(y[mask], preds[mask])
            results.append({
                "domain": domain,
                "cluster": c,
                "teacher": teacher_name,
                "size": int(mask.sum()),
                "overall_accuracy": float(overall_acc),
                "cluster_accuracy": float(acc),
            })

    cluster_acc_df = pd.DataFrame(results).sort_values(
        ["domain", "cluster", "teacher"]
    ).reset_index(drop=True)

    winners_df = (
        cluster_acc_df.loc[cluster_acc_df.groupby(["domain", "cluster"])["cluster_accuracy"].idxmax()]
        .sort_values(["domain", "cluster"])
        .reset_index(drop=True)
    )

    spread_rows = []
    for (dom, c), g in cluster_acc_df.groupby(["domain", "cluster"], sort=True):
        g_sorted = g.sort_values(["cluster_accuracy", "teacher"], ascending=[False, True]).reset_index(drop=True)

        best = float(g_sorted.iloc[0]["cluster_accuracy"])
        second = float(g_sorted.iloc[1]["cluster_accuracy"]) if len(g_sorted) > 1 else best
        worst = float(g_sorted.iloc[-1]["cluster_accuracy"])

        spread_rows.append({
            "domain": dom,
            "cluster": int(c),
            "size": int(g_sorted.iloc[0]["size"]),
            "best_acc": best,
            "second_acc": second,
            "worst_acc": worst,
            "top_gap": best - second,
            "acc_spread": best - worst,
            "winner": str(g_sorted.iloc[0]["teacher"]),
        })

    spread_df = pd.DataFrame(spread_rows).sort_values(["domain", "cluster"]).reset_index(drop=True)
    spread_df = spread_df.merge(cluster_meta_df, on=["domain", "cluster", "size"], how="left")

    return membership_df, cluster_meta_df, cluster_acc_df, winners_df.merge(
        cluster_meta_df, on=["domain", "cluster", "size"], how="left"
    ), spread_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pretrained teacher ranking flips across domains and optionally within-domain clusters."
    )
    parser.add_argument(
        "--teacher",
        action="append",
        required=True,
        help="Teacher spec in the form alias=path_to_npz. Repeat this flag for all teacher/domain npz files.",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="analysis")
    parser.add_argument("--run_cluster_analysis", action="store_true")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    domain_to_teacher_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for item in args.teacher:
        alias, path = parse_teacher_arg(item)
        data = load_npz_dict(path)
        domain, teacher = infer_domain_and_teacher(alias, data, path)

        if domain not in domain_to_teacher_data:
            domain_to_teacher_data[domain] = {}
        if teacher in domain_to_teacher_data[domain]:
            raise ValueError(f"Duplicate teacher={teacher} for domain={domain}")

        domain_to_teacher_data[domain][teacher] = data

    domains = sorted(domain_to_teacher_data.keys())
    if not domains:
        raise ValueError("No domains found")

    ensure_domain_alignment(domain_to_teacher_data)

    teacher_sets = {domain: set(domain_to_teacher_data[domain].keys()) for domain in domains}
    ref_teacher_set = teacher_sets[domains[0]]
    for domain in domains[1:]:
        if teacher_sets[domain] != ref_teacher_set:
            raise ValueError(
                f"Teacher set mismatch.\n"
                f"{domains[0]}: {sorted(ref_teacher_set)}\n"
                f"{domain}: {sorted(teacher_sets[domain])}"
            )

    teacher_names = sorted(ref_teacher_set)

    domain_rows = []
    ranking_map: Dict[str, List[str]] = {}
    acc_map: Dict[str, Dict[str, float]] = {}

    for domain in domains:
        accs_this_domain = {}

        for teacher in teacher_names:
            data = domain_to_teacher_data[domain][teacher]
            y = data["labels"].astype(np.int64)
            preds = get_preds(data)
            acc = float(accuracy_score(y, preds))

            meta = data.get("_meta", {})
            domain_rows.append({
                "domain": domain,
                "teacher": teacher,
                "accuracy": acc,
                "n_samples": int(len(y)),
                "feature_dim": int(data["feats"].shape[1]),
                "logit_dim": int(data["logits"].shape[1]) if "logits" in data else -1,
                "model_tag": meta.get("model_tag", "unknown"),
                "dataset_name": meta.get("dataset_name", "unknown"),
            })
            accs_this_domain[teacher] = acc

        acc_map[domain] = accs_this_domain
        ranking_map[domain] = [
            t for t, _ in sorted(accs_this_domain.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    domain_df = pd.DataFrame(domain_rows).sort_values(
        ["domain", "accuracy", "teacher"], ascending=[True, False, True]
    )
    domain_df.to_csv(outdir / f"domain_teacher_accuracy_{args.tag}.csv", index=False)

    domain_winners = (
        domain_df.loc[domain_df.groupby("domain")["accuracy"].idxmax()]
        .sort_values("domain")
        .reset_index(drop=True)
    )
    domain_winners.to_csv(outdir / f"domain_teacher_winners_{args.tag}.csv", index=False)

    pivot_df = domain_df.pivot(index="domain", columns="teacher", values="accuracy").reset_index()
    pivot_df.to_csv(outdir / f"domain_teacher_accuracy_pivot_{args.tag}.csv", index=False)

    ranking_rows = []
    for domain in domains:
        row = {"domain": domain}
        for rank_idx, teacher in enumerate(ranking_map[domain], start=1):
            row[f"rank_{rank_idx}"] = teacher
        ranking_rows.append(row)
    ranking_df = pd.DataFrame(ranking_rows).sort_values("domain")
    ranking_df.to_csv(outdir / f"domain_teacher_rankings_{args.tag}.csv", index=False)

    pair_rows = []
    for d1, d2 in combinations(domains, 2):
        best1 = ranking_map[d1][0]
        best2 = ranking_map[d2][0]
        same_winner = int(best1 == best2)

        pair_rows.append({
            "domain_a": d1,
            "domain_b": d2,
            "winner_a": best1,
            "winner_b": best2,
            "same_winner": same_winner,
            "winner_changed": int(not same_winner),
            "rank_corr": average_rank_correlation(ranking_map[d1], ranking_map[d2]),
            "pairwise_flip_fraction": pairwise_order_flip_fraction(acc_map[d1], acc_map[d2]),
        })

    pair_df = pd.DataFrame(pair_rows).sort_values(["domain_a", "domain_b"])
    pair_df.to_csv(outdir / f"domain_pair_flips_{args.tag}.csv", index=False)

    # leave-one-domain-out source selection
    source_select_rows = []
    for target_domain in domains:
        source_domains = [d for d in domains if d != target_domain]

        pooled_source_acc = {}
        for teacher in teacher_names:
            vals = [acc_map[d][teacher] for d in source_domains]
            pooled_source_acc[teacher] = float(np.mean(vals))

        selected_teacher = sorted(
            pooled_source_acc.items(), key=lambda kv: (-kv[1], kv[0])
        )[0][0]

        target_acc = acc_map[target_domain][selected_teacher]
        oracle_teacher = ranking_map[target_domain][0]
        oracle_acc = acc_map[target_domain][oracle_teacher]

        source_select_rows.append({
            "target_domain": target_domain,
            "selected_from_sources": selected_teacher,
            "selected_teacher_source_mean_acc": pooled_source_acc[selected_teacher],
            "selected_teacher_target_acc": target_acc,
            "oracle_teacher": oracle_teacher,
            "oracle_target_acc": oracle_acc,
            "gap_to_oracle": oracle_acc - target_acc,
        })

    source_select_df = pd.DataFrame(source_select_rows).sort_values("target_domain")
    source_select_df.to_csv(outdir / f"source_selected_vs_oracle_{args.tag}.csv", index=False)

    if args.run_cluster_analysis:
        membership_dfs = []
        cluster_meta_dfs = []
        cluster_acc_dfs = []
        cluster_winner_dfs = []
        cluster_spread_dfs = []

        for domain in domains:
            membership_df, cluster_meta_df, cluster_acc_df, cluster_winner_df, cluster_spread_df = \
                run_cluster_analysis_for_domain(
                    domain=domain,
                    teacher_map=domain_to_teacher_data[domain],
                    k=args.k,
                    seed=args.seed,
                )

            membership_dfs.append(membership_df)
            cluster_meta_dfs.append(cluster_meta_df)
            cluster_acc_dfs.append(cluster_acc_df)
            cluster_winner_dfs.append(cluster_winner_df)
            cluster_spread_dfs.append(cluster_spread_df)

        pd.concat(membership_dfs, ignore_index=True).to_csv(
            outdir / f"cluster_membership_{args.tag}.csv", index=False
        )
        pd.concat(cluster_meta_dfs, ignore_index=True).to_csv(
            outdir / f"cluster_meta_{args.tag}.csv", index=False
        )
        pd.concat(cluster_acc_dfs, ignore_index=True).to_csv(
            outdir / f"cluster_teacher_accuracy_{args.tag}.csv", index=False
        )
        pd.concat(cluster_winner_dfs, ignore_index=True).to_csv(
            outdir / f"cluster_teacher_winners_{args.tag}.csv", index=False
        )
        pd.concat(cluster_spread_dfs, ignore_index=True).to_csv(
            outdir / f"cluster_teacher_spread_{args.tag}.csv", index=False
        )

    print("\n=== Domain-level teacher accuracies ===")
    print(domain_df.to_string(index=False))

    print("\n=== Domain winners ===")
    print(domain_winners.to_string(index=False))

    print("\n=== Domain pair ranking flips ===")
    if len(pair_df) > 0:
        print(pair_df.to_string(index=False))
    else:
        print("Only one domain provided; no pairwise flip analysis.")

    print("\n=== Source-selected vs oracle on each target domain ===")
    print(source_select_df.to_string(index=False))

    print("\nSaved files:")
    print(outdir / f"domain_teacher_accuracy_{args.tag}.csv")
    print(outdir / f"domain_teacher_winners_{args.tag}.csv")
    print(outdir / f"domain_teacher_accuracy_pivot_{args.tag}.csv")
    print(outdir / f"domain_teacher_rankings_{args.tag}.csv")
    print(outdir / f"domain_pair_flips_{args.tag}.csv")
    print(outdir / f"source_selected_vs_oracle_{args.tag}.csv")

    if args.run_cluster_analysis:
        print(outdir / f"cluster_membership_{args.tag}.csv")
        print(outdir / f"cluster_meta_{args.tag}.csv")
        print(outdir / f"cluster_teacher_accuracy_{args.tag}.csv")
        print(outdir / f"cluster_teacher_winners_{args.tag}.csv")
        print(outdir / f"cluster_teacher_spread_{args.tag}.csv")


if __name__ == "__main__":
    main()

# python analyze_teacher_rankings.py `
#   --teacher openclip_l14_openai_qgelu=teacher_npzs\terra_incognita__openclip_l14_openai_qgelu__location_38.npz `
#   --teacher openclip_b16_datacomp=teacher_npzs\terra_incognita__openclip_b16_datacomp__location_38.npz `
#   --teacher openclip_so400m_siglip=teacher_npzs\terra_incognita__openclip_so400m_siglip__location_38.npz `
#   --teacher openclip_l14_dfn2b=teacher_npzs\terra_incognita__openclip_l14_dfn2b__location_38.npz `
#   --teacher openclip_h14_laion2b=teacher_npzs\terra_incognita__openclip_h14_laion2b__location_38.npz `
#   --teacher openclip_h14_378_dfn5b=teacher_npzs\terra_incognita__openclip_h14_378_dfn5b__location_38.npz `
#   --teacher openclip_convnext_xxlarge=teacher_npzs\terra_incognita__openclip_convnext_xxlarge__location_38.npz `
#   --teacher openclip_l14_openai_qgelu=teacher_npzs\terra_incognita__openclip_l14_openai_qgelu__location_43.npz `
#   --teacher openclip_b16_datacomp=teacher_npzs\terra_incognita__openclip_b16_datacomp__location_43.npz `
#   --teacher openclip_so400m_siglip=teacher_npzs\terra_incognita__openclip_so400m_siglip__location_43.npz `
#   --teacher openclip_l14_dfn2b=teacher_npzs\terra_incognita__openclip_l14_dfn2b__location_43.npz `
#   --teacher openclip_h14_laion2b=teacher_npzs\terra_incognita__openclip_h14_laion2b__location_43.npz `
#   --teacher openclip_h14_378_dfn5b=teacher_npzs\terra_incognita__openclip_h14_378_dfn5b__location_43.npz `
#   --teacher openclip_convnext_xxlarge=teacher_npzs\terra_incognita__openclip_convnext_xxlarge__location_43.npz `
#   --teacher openclip_l14_openai_qgelu=teacher_npzs\terra_incognita__openclip_l14_openai_qgelu__location_46.npz `
#   --teacher openclip_b16_datacomp=teacher_npzs\terra_incognita__openclip_b16_datacomp__location_46.npz `
#   --teacher openclip_so400m_siglip=teacher_npzs\terra_incognita__openclip_so400m_siglip__location_46.npz `
#   --teacher openclip_l14_dfn2b=teacher_npzs\terra_incognita__openclip_l14_dfn2b__location_46.npz `
#   --teacher openclip_h14_laion2b=teacher_npzs\terra_incognita__openclip_h14_laion2b__location_46.npz `
#   --teacher openclip_h14_378_dfn5b=teacher_npzs\terra_incognita__openclip_h14_378_dfn5b__location_46.npz `
#   --teacher openclip_convnext_xxlarge=teacher_npzs\terra_incognita__openclip_convnext_xxlarge__location_46.npz `
#   --teacher openclip_l14_openai_qgelu=teacher_npzs\terra_incognita__openclip_l14_openai_qgelu__location_100.npz `
#   --teacher openclip_b16_datacomp=teacher_npzs\terra_incognita__openclip_b16_datacomp__location_100.npz `
#   --teacher openclip_so400m_siglip=teacher_npzs\terra_incognita__openclip_so400m_siglip__location_100.npz `
#   --teacher openclip_l14_dfn2b=teacher_npzs\terra_incognita__openclip_l14_dfn2b__location_100.npz `
#   --teacher openclip_h14_laion2b=teacher_npzs\terra_incognita__openclip_h14_laion2b__location_100.npz `
#   --teacher openclip_h14_378_dfn5b=teacher_npzs\terra_incognita__openclip_h14_378_dfn5b__location_100.npz `
#   --teacher openclip_convnext_xxlarge=teacher_npzs\terra_incognita__openclip_convnext_xxlarge__location_100.npz `
#   --outdir terra_analysis `
#   --tag terra