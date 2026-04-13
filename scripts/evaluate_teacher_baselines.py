#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# ============================================================
# IO
# ============================================================

def load_npz_dict(path: str | Path) -> Dict[str, np.ndarray]:
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


def parse_teacher_arg(item: str) -> Tuple[str, str]:
    """
    Expected:
      --teacher alias=path_to_npz
    """
    if "=" not in item:
        raise ValueError(f"--teacher must be alias=path, got: {item}")
    alias, path = item.split("=", 1)
    alias = alias.strip()
    path = path.strip()
    if not alias or not path:
        raise ValueError(f"Invalid --teacher argument: {item}")
    return alias, path


def infer_domain_and_teacher(alias: str, data: Dict[str, np.ndarray], path: str) -> Tuple[str, str]:
    meta = data.get("_meta", {})
    domain = meta.get("domain")
    teacher = meta.get("teacher_name")

    if teacher is None:
        teacher = alias

    if domain is None:
        # Fallback to filename convention:
        # dataset__teacher__domain.npz
        stem = Path(path).stem
        parts = stem.split("__")
        if len(parts) >= 3:
            domain = parts[-1]
        else:
            domain = "unknown_domain"

    return str(domain), str(teacher)


# ============================================================
# Metrics / checks
# ============================================================

def get_preds_from_logits(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=1).astype(np.int64)


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


def ensure_domain_alignment(domain_to_teacher_data: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> None:
    """
    Within each domain:
      - all teachers must have identical sample count
      - identical labels
      - identical path ordering
      - identical class count
    """
    for domain, teacher_map in domain_to_teacher_data.items():
        teacher_names = sorted(teacher_map.keys())
        if not teacher_names:
            continue

        ref_teacher = teacher_names[0]
        ref = teacher_map[ref_teacher]

        ref_labels = ref["labels"].astype(np.int64)
        ref_paths = ref["paths"]
        ref_logits = ref["logits"]
        ref_num_classes = ref_logits.shape[1]

        for teacher in teacher_names[1:]:
            cur = teacher_map[teacher]

            cur_labels = cur["labels"].astype(np.int64)
            cur_paths = cur["paths"]
            cur_logits = cur["logits"]
            cur_num_classes = cur_logits.shape[1]

            if len(cur_labels) != len(ref_labels):
                raise ValueError(
                    f"Sample count mismatch in domain={domain}: "
                    f"{teacher} has {len(cur_labels)} vs {ref_teacher} has {len(ref_labels)}"
                )

            if not np.array_equal(cur_labels, ref_labels):
                raise ValueError(
                    f"Label mismatch in domain={domain} between {ref_teacher} and {teacher}"
                )

            if len(cur_paths) != len(ref_paths):
                raise ValueError(
                    f"Path count mismatch in domain={domain} between {ref_teacher} and {teacher}"
                )

            if not np.array_equal(cur_paths, ref_paths):
                raise ValueError(
                    f"Path ordering mismatch in domain={domain} between {ref_teacher} and {teacher}"
                )

            if cur_num_classes != ref_num_classes:
                raise ValueError(
                    f"Logit dimension mismatch in domain={domain}: "
                    f"{teacher} has {cur_num_classes} vs {ref_teacher} has {ref_num_classes}"
                )


def ensure_teacher_set_consistency(domain_to_teacher_data: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> List[str]:
    domains = sorted(domain_to_teacher_data.keys())
    if not domains:
        raise ValueError("No domains found.")

    ref_teachers = set(domain_to_teacher_data[domains[0]].keys())
    for domain in domains[1:]:
        cur = set(domain_to_teacher_data[domain].keys())
        if cur != ref_teachers:
            raise ValueError(
                f"Teacher set mismatch.\n"
                f"{domains[0]}: {sorted(ref_teachers)}\n"
                f"{domain}: {sorted(cur)}"
            )

    return sorted(ref_teachers)


# ============================================================
# Baselines
# ============================================================

@dataclass
class DomainPredictionResult:
    domain: str
    method: str
    selected_teacher: str | None
    accuracy: float
    balanced_accuracy: float
    n_samples: int


def evaluate_teacher_logits(y: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    pred = get_preds_from_logits(logits)
    return compute_metrics(y, pred)


def evaluate_uniform_ensemble(y: np.ndarray, logits_list: List[np.ndarray]) -> Dict[str, float]:
    mean_logits = np.mean(np.stack(logits_list, axis=0), axis=0)
    pred = get_preds_from_logits(mean_logits)
    return compute_metrics(y, pred)


def evaluate_confidence_hard_selection(
    y: np.ndarray,
    teacher_to_logits: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], np.ndarray]:
    teacher_names = sorted(teacher_to_logits.keys())
    probs_stack = []
    logits_stack = []

    for teacher in teacher_names:
        logits = teacher_to_logits[teacher]
        probs = softmax_np(logits, axis=1)
        probs_stack.append(probs)
        logits_stack.append(logits)

    probs_stack = np.stack(probs_stack, axis=0)   # [T, N, C]
    logits_stack = np.stack(logits_stack, axis=0) # [T, N, C]

    conf = probs_stack.max(axis=2)                # [T, N]
    best_teacher_idx = conf.argmax(axis=0)        # [N]

    chosen_logits = logits_stack[best_teacher_idx, np.arange(logits_stack.shape[1])]
    pred = get_preds_from_logits(chosen_logits)
    metrics = compute_metrics(y, pred)
    return metrics, best_teacher_idx


def evaluate_confidence_soft_ensemble(
    y: np.ndarray,
    teacher_to_logits: Dict[str, np.ndarray],
    temperature: float = 1.0,
) -> Dict[str, float]:
    teacher_names = sorted(teacher_to_logits.keys())
    probs_stack = []
    logits_stack = []

    for teacher in teacher_names:
        logits = teacher_to_logits[teacher]
        probs = softmax_np(logits, axis=1)
        probs_stack.append(probs)
        logits_stack.append(logits)

    probs_stack = np.stack(probs_stack, axis=0)   # [T, N, C]
    logits_stack = np.stack(logits_stack, axis=0) # [T, N, C]

    conf = probs_stack.max(axis=2)                # [T, N]
    conf = conf / max(temperature, 1e-8)

    weights = softmax_np(conf, axis=0)            # [T, N]
    weighted_logits = (weights[:, :, None] * logits_stack).sum(axis=0)

    pred = get_preds_from_logits(weighted_logits)
    return compute_metrics(y, pred)


# ============================================================
# Ranking instability analysis
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate standard teacher-selection baselines from extractor-produced NPZ files.\n\n"
            "Expected use:\n"
            "  python evaluate_teacher_baselines.py \\\n"
            "    --teacher openclip_l14_openai_qgelu=...location_38.npz \\\n"
            "    --teacher openclip_b16_datacomp=...location_38.npz \\\n"
            "    ... repeat for all teacher/domain files ..."
        )
    )
    parser.add_argument(
        "--teacher",
        action="append",
        required=True,
        help="Teacher spec in the form alias=path_to_npz. Repeat for all teacher/domain NPZs.",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--tag", type=str, default="baseline_eval")
    parser.add_argument(
        "--confidence_soft_temperature",
        type=float,
        default=1.0,
        help="Temperature for confidence-weighted soft ensemble."
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load NPZs into nested dict: domain -> teacher -> data
    # --------------------------------------------------------
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
    teacher_names = ensure_teacher_set_consistency(domain_to_teacher_data)
    ensure_domain_alignment(domain_to_teacher_data)

    # --------------------------------------------------------
    # Per-teacher / per-domain metrics
    # --------------------------------------------------------
    domain_teacher_rows = []
    acc_map: Dict[str, Dict[str, float]] = {}
    ranking_map: Dict[str, List[str]] = {}

    for domain in domains:
        teacher_accs = {}
        for teacher in teacher_names:
            data = domain_to_teacher_data[domain][teacher]
            y = data["labels"].astype(np.int64)
            logits = data["logits"].astype(np.float32)
            metrics = evaluate_teacher_logits(y, logits)

            meta = data.get("_meta", {})
            domain_teacher_rows.append({
                "domain": domain,
                "teacher": teacher,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n_samples": metrics["n_samples"],
                "dataset_name": meta.get("dataset_name", "unknown"),
                "model_tag": meta.get("model_tag", "unknown"),
                "zero_shot_top1_acc_meta": meta.get("zero_shot_top1_acc", None),
                "npz_path": data.get("_path", ""),
            })
            teacher_accs[teacher] = metrics["accuracy"]

        acc_map[domain] = teacher_accs
        ranking_map[domain] = [
            t for t, _ in sorted(teacher_accs.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    domain_teacher_df = pd.DataFrame(domain_teacher_rows).sort_values(
        ["domain", "accuracy", "teacher"], ascending=[True, False, True]
    )
    domain_teacher_df.to_csv(outdir / f"domain_teacher_metrics_{args.tag}.csv", index=False)

    domain_pivot_df = domain_teacher_df.pivot(index="domain", columns="teacher", values="accuracy").reset_index()
    domain_pivot_df.to_csv(outdir / f"domain_teacher_accuracy_pivot_{args.tag}.csv", index=False)

    domain_oracle_df = (
        domain_teacher_df.loc[domain_teacher_df.groupby("domain")["accuracy"].idxmax()]
        .sort_values("domain")
        .reset_index(drop=True)
    )
    domain_oracle_df["method"] = "oracle_domain_best_teacher"
    domain_oracle_df.to_csv(outdir / f"oracle_domain_best_teacher_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Global best single teacher
    # --------------------------------------------------------
    mean_acc_per_teacher = (
        domain_teacher_df.groupby("teacher", as_index=False)["accuracy"]
        .mean()
        .sort_values(["accuracy", "teacher"], ascending=[False, True])
        .reset_index(drop=True)
    )
    mean_acc_per_teacher.to_csv(outdir / f"teacher_global_mean_accuracy_{args.tag}.csv", index=False)

    global_best_teacher = str(mean_acc_per_teacher.iloc[0]["teacher"])

    baseline_rows: List[Dict] = []

    for domain in domains:
        ref_teacher = teacher_names[0]
        y = domain_to_teacher_data[domain][ref_teacher]["labels"].astype(np.int64)

        # Global best
        logits_global = domain_to_teacher_data[domain][global_best_teacher]["logits"].astype(np.float32)
        m_global = evaluate_teacher_logits(y, logits_global)
        baseline_rows.append({
            "domain": domain,
            "method": "global_best_teacher",
            "selected_teacher": global_best_teacher,
            **m_global,
        })

    # --------------------------------------------------------
    # Leave-one-domain-out best teacher
    # --------------------------------------------------------
    lodo_rows = []
    for target_domain in domains:
        source_domains = [d for d in domains if d != target_domain]
        teacher_source_mean = {}

        for teacher in teacher_names:
            vals = [acc_map[d][teacher] for d in source_domains]
            teacher_source_mean[teacher] = float(np.mean(vals))

        selected_teacher = sorted(
            teacher_source_mean.items(), key=lambda kv: (-kv[1], kv[0])
        )[0][0]

        ref_teacher = teacher_names[0]
        y = domain_to_teacher_data[target_domain][ref_teacher]["labels"].astype(np.int64)
        logits = domain_to_teacher_data[target_domain][selected_teacher]["logits"].astype(np.float32)
        m_lodo = evaluate_teacher_logits(y, logits)

        oracle_teacher = ranking_map[target_domain][0]
        oracle_acc = acc_map[target_domain][oracle_teacher]

        row = {
            "domain": target_domain,
            "method": "leave_one_domain_out_best_teacher",
            "selected_teacher": selected_teacher,
            "selected_teacher_source_mean_acc": teacher_source_mean[selected_teacher],
            "oracle_teacher": oracle_teacher,
            "oracle_target_acc": oracle_acc,
            "gap_to_oracle": oracle_acc - m_lodo["accuracy"],
            **m_lodo,
        }
        lodo_rows.append(row)
        baseline_rows.append(row)

    lodo_df = pd.DataFrame(lodo_rows).sort_values("domain").reset_index(drop=True)
    lodo_df.to_csv(outdir / f"leave_one_domain_out_best_teacher_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Uniform ensemble / confidence hard / confidence soft
    # --------------------------------------------------------
    for domain in domains:
        ref_teacher = teacher_names[0]
        y = domain_to_teacher_data[domain][ref_teacher]["labels"].astype(np.int64)

        teacher_to_logits = {
            teacher: domain_to_teacher_data[domain][teacher]["logits"].astype(np.float32)
            for teacher in teacher_names
        }

        # Uniform ensemble
        m_uniform = evaluate_uniform_ensemble(y, list(teacher_to_logits.values()))
        baseline_rows.append({
            "domain": domain,
            "method": "uniform_logit_ensemble",
            "selected_teacher": None,
            **m_uniform,
        })

        # Confidence hard selection
        m_conf_hard, chosen_teacher_idx = evaluate_confidence_hard_selection(y, teacher_to_logits)
        teacher_array = np.array(sorted(teacher_to_logits.keys()), dtype=object)
        chosen_teachers = teacher_array[chosen_teacher_idx]
        teacher_counts = pd.Series(chosen_teachers).value_counts().sort_index()

        baseline_rows.append({
            "domain": domain,
            "method": "confidence_hard_selection",
            "selected_teacher": None,
            "selection_histogram_json": teacher_counts.to_json(),
            **m_conf_hard,
        })

        # Confidence soft ensemble
        m_conf_soft = evaluate_confidence_soft_ensemble(
            y=y,
            teacher_to_logits=teacher_to_logits,
            temperature=args.confidence_soft_temperature,
        )
        baseline_rows.append({
            "domain": domain,
            "method": "confidence_soft_ensemble",
            "selected_teacher": None,
            "confidence_soft_temperature": args.confidence_soft_temperature,
            **m_conf_soft,
        })

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(outdir / f"baseline_results_{args.tag}.csv", index=False)

    baseline_summary_df = (
        baseline_df.groupby("method", as_index=False)[["accuracy", "balanced_accuracy"]]
        .mean()
        .sort_values(["accuracy", "method"], ascending=[False, True])
        .reset_index(drop=True)
    )
    baseline_summary_df.to_csv(outdir / f"baseline_summary_{args.tag}.csv", index=False)

    baseline_pivot_df = baseline_df.pivot(index="domain", columns="method", values="accuracy").reset_index()
    baseline_pivot_df.to_csv(outdir / f"baseline_accuracy_pivot_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Ranking instability analysis
    # --------------------------------------------------------
    pair_rows = []
    for d1, d2 in combinations(domains, 2):
        pair_rows.append({
            "domain_a": d1,
            "domain_b": d2,
            "winner_a": ranking_map[d1][0],
            "winner_b": ranking_map[d2][0],
            "same_winner": int(ranking_map[d1][0] == ranking_map[d2][0]),
            "winner_changed": int(ranking_map[d1][0] != ranking_map[d2][0]),
            "rank_corr": average_rank_correlation(ranking_map[d1], ranking_map[d2]),
            "pairwise_flip_fraction": pairwise_order_flip_fraction(acc_map[d1], acc_map[d2]),
        })

    pair_df = pd.DataFrame(pair_rows).sort_values(["domain_a", "domain_b"])
    pair_df.to_csv(outdir / f"domain_pair_ranking_flips_{args.tag}.csv", index=False)

    # --------------------------------------------------------
    # Print compact summary
    # --------------------------------------------------------
    print("\n=== Per-domain oracle teachers ===")
    print(domain_oracle_df[["domain", "teacher", "accuracy", "balanced_accuracy"]].to_string(index=False))

    print("\n=== Global mean accuracy by teacher ===")
    print(mean_acc_per_teacher.to_string(index=False))

    print(f"\n=== Global best teacher: {global_best_teacher} ===")

    print("\n=== Baseline summary (mean across domains) ===")
    print(baseline_summary_df.to_string(index=False))

    print("\n=== Baseline accuracy by domain ===")
    print(baseline_pivot_df.to_string(index=False))

    if len(pair_df) > 0:
        print("\n=== Domain-pair ranking instability ===")
        print(pair_df.to_string(index=False))

    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

# $domains = @("location_38", "location_43", "location_46", "location_100")
# $teachers = @(
#     "openclip_l14_openai_qgelu",
#     "openclip_b16_datacomp",
#     "openclip_so400m_siglip",
#     "openclip_l14_dfn2b",
#     "openclip_h14_laion2b",
#     "openclip_h14_378_dfn5b",
#     "openclip_convnext_xxlarge"
# )

# $argsList = @()

# foreach ($d in $domains) {
#     foreach ($t in $teachers) {
#         $path = "teacher_npzs\terra_incognita__${t}__${d}.npz"
#         $argsList += "--teacher"
#         $argsList += "${t}=${path}"
#     }
# }

# python evaluate_teacher_baselines.py `
#     @argsList `
#     --outdir baseline_eval `
#     --tag terra_incognita `
#     --confidence_soft_temperature 1.0
