#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Teacher registry
# ============================================================

TEACHERS_DEFAULT = [
    "openclip_l14_openai_qgelu",
    "openclip_b16_datacomp",
    "openclip_so400m_siglip",
    "openclip_l14_dfn2b",
    "openclip_h14_laion2b",
    "openclip_h14_378_dfn5b",
    "openclip_convnext_xxlarge",
]


# ============================================================
# IO helpers
# ============================================================

def load_npz_with_meta(npz_path: Path) -> Dict[str, Any]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    out: Dict[str, Any] = {
        "feats": data["feats"],
        "labels": data["labels"],
        "paths": data["paths"],
    }

    if "logits" in data:
        out["logits"] = data["logits"]
    if "preds" in data:
        out["preds"] = data["preds"]

    meta_json = data["meta_json"].item() if hasattr(data["meta_json"], "item") else data["meta_json"]
    out["meta"] = json.loads(meta_json)
    return out


def save_json(out_path: Path, obj: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"saved: {out_path}")


def save_probe_outputs_npz(
    out_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: np.ndarray,
    proba: np.ndarray | None,
    decision_function: np.ndarray | None,
    meta: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "y_true": y_true.astype(np.int64),
        "y_pred": y_pred.astype(np.int64),
        "paths": np.array([str(p) for p in paths], dtype=object),
        "meta_json": json.dumps(meta),
    }

    if proba is not None:
        payload["proba"] = proba.astype(np.float32)

    if decision_function is not None:
        payload["decision_function"] = np.asarray(decision_function).astype(np.float32)

    np.savez_compressed(out_path, **payload)
    print(f"saved: {out_path}")


# ============================================================
# Domain / class helpers
# ============================================================

def canonicalize_domain_name(name: str) -> str:
    name = name.strip().lower()
    if name.endswith("_subset"):
        name = name[:-7]
    return name


def resolve_domain_names(npz_root: Path, domain_names_arg: List[str] | None) -> List[str]:
    if domain_names_arg and len(domain_names_arg) > 0:
        return list(domain_names_arg)

    # infer from filenames: dataset__teacher__domain.npz
    found = set()
    for p in npz_root.glob("*.npz"):
        parts = p.stem.split("__")
        if len(parts) >= 3:
            found.add(parts[-1])

    domain_names = sorted(found)
    if not domain_names:
        raise RuntimeError(
            f"Could not infer domain names from NPZ directory: {npz_root}\n"
            "Pass --domain_names explicitly."
        )
    return domain_names


def npz_path_for(npz_root: Path, dataset_name: str, teacher_name: str, domain_name: str) -> Path:
    safe_dataset = dataset_name.replace(" ", "_").lower()
    safe_domain = canonicalize_domain_name(domain_name).replace(" ", "_").lower()
    return npz_root / f"{safe_dataset}__{teacher_name}__{safe_domain}.npz"


def load_teacher_domain_npzs(
    npz_root: Path,
    dataset_name: str,
    teacher_name: str,
    domain_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    domain_data: Dict[str, Dict[str, Any]] = {}

    for domain_name in domain_names:
        npz_path = npz_path_for(npz_root, dataset_name, teacher_name, domain_name)
        data = load_npz_with_meta(npz_path)
        domain_data[domain_name] = data

    return domain_data


def verify_class_alignment(
    domain_data: Dict[str, Dict[str, Any]],
    domain_names: List[str],
    class_mode: str,
    train_domain: str,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    metas = {d: domain_data[d]["meta"] for d in domain_names}
    classes_by_domain = {d: metas[d]["class_names"] for d in domain_names}

    if class_mode == "strict":
        ref = classes_by_domain[domain_names[0]]
        for d in domain_names[1:]:
            if classes_by_domain[d] != ref:
                raise ValueError(
                    "Class names differ across domains under strict mode.\n"
                    f"{domain_names[0]}: {ref[:10]} ... ({len(ref)} classes)\n"
                    f"{d}: {classes_by_domain[d][:10]} ... ({len(classes_by_domain[d])} classes)"
                )
        return list(ref), {d: np.arange(len(classes_by_domain[d])) for d in domain_names}

    if class_mode == "from_train":
        ref = classes_by_domain[train_domain]
        for d in domain_names:
            if classes_by_domain[d] != ref:
                raise ValueError(
                    f"Class names for domain {d} do not match train domain {train_domain}.\n"
                    "Re-extract with a shared class vocabulary or use strict-compatible NPZs."
                )
        return list(ref), {d: np.arange(len(classes_by_domain[d])) for d in domain_names}

    if class_mode == "intersection":
        common = set(classes_by_domain[domain_names[0]])
        for d in domain_names[1:]:
            common &= set(classes_by_domain[d])

        common = sorted(common)
        if not common:
            raise RuntimeError("No common classes across NPZ metadata.")

        index_maps: Dict[str, np.ndarray] = {}
        for d in domain_names:
            cls = classes_by_domain[d]
            name_to_idx = {name: i for i, name in enumerate(cls)}
            index_maps[d] = np.array([name_to_idx[name] for name in common], dtype=np.int64)

        return common, index_maps

    raise ValueError(f"Unknown class_mode: {class_mode}")


# ============================================================
# Probe helpers
# ============================================================

def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    max_iter: int,
    random_state: int,
):
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logreg",
                LogisticRegression(
                    C=c_value,
                    max_iter=max_iter,
                    solver="lbfgs",
                    n_jobs=None,
                    random_state=random_state,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    return clf


def evaluate_probe_full(
    clf,
    x: np.ndarray,
    y: np.ndarray,
    paths: np.ndarray,
) -> Dict[str, Any]:
    pred = clf.predict(x)

    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n_samples": int(len(y)),
        "y_true": y.astype(int).tolist(),
        "y_pred": pred.astype(int).tolist(),
        "paths": [str(p) for p in paths],
    }

    proba_np = None
    decision_np = None

    if hasattr(clf, "predict_proba"):
        try:
            proba_np = clf.predict_proba(x)
            out["proba"] = np.asarray(proba_np).astype(float).tolist()
        except Exception:
            proba_np = None

    if hasattr(clf, "decision_function"):
        try:
            decision_np = clf.decision_function(x)
            out["decision_function"] = np.asarray(decision_np).astype(float).tolist()
        except Exception:
            decision_np = None

    return out


def subset_to_class_indices(
    feats: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    keep_old_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_set = set(int(x) for x in keep_old_indices.tolist())
    old_to_new = {int(old): new for new, old in enumerate(keep_old_indices.tolist())}

    keep_mask = np.array([int(y) in keep_set for y in labels], dtype=bool)
    if not keep_mask.any():
        raise RuntimeError("No samples left after class intersection filtering.")

    feats2 = feats[keep_mask]
    labels_old = labels[keep_mask]
    paths2 = paths[keep_mask]
    labels2 = np.array([old_to_new[int(y)] for y in labels_old], dtype=np.int64)
    return feats2, labels2, paths2


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train frozen-feature linear probes for DomainNet (or similar) from precomputed NPZs.\n\n"
            "Expected NPZ naming:\n"
            "  dataset__teacher__domain.npz\n\n"
            "Protocol:\n"
            "  1) load precomputed teacher NPZs for all domains\n"
            "  2) choose one train domain\n"
            "  3) split train domain into 80/20 train/val\n"
            "  4) fit linear probe on train split only\n"
            "  5) evaluate on held-out val split + all other domains"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--npz_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="domainnet")
    parser.add_argument("--train_domain", type=str, required=True)
    parser.add_argument("--domain_names", nargs="*", default=None)
    parser.add_argument(
        "--class_mode",
        choices=["strict", "intersection", "from_train"],
        default="strict",
        help=(
            "strict       -> all NPZ class lists must match exactly\n"
            "intersection -> keep only common classes across domains\n"
            "from_train   -> require all domains to match train-domain class order"
        ),
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=TEACHERS_DEFAULT,
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--probe_C", type=float, default=1.0)
    parser.add_argument("--probe_max_iter", type=int, default=2000)
    parser.add_argument("--save_probe_outputs", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    npz_root = Path(args.npz_root)
    outdir = Path(args.outdir)

    if not npz_root.exists():
        raise FileNotFoundError(f"NPZ root not found: {npz_root}")

    domain_names = resolve_domain_names(npz_root, args.domain_names)
    if args.train_domain not in domain_names:
        raise ValueError(
            f"--train_domain={args.train_domain!r} not found in domain_names={domain_names}"
        )

    print(f"NPZ root: {npz_root}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Train domain: {args.train_domain}")
    print(f"Domains: {domain_names}")
    print(f"Class mode: {args.class_mode}")

    summary: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "npz_root": str(npz_root),
        "train_domain": args.train_domain,
        "domain_names": domain_names,
        "class_mode": args.class_mode,
        "teachers": {},
    }

    for teacher_name in args.teachers:
        print("\n" + "=" * 80)
        print(f"Teacher: {teacher_name}")
        print("=" * 80)

        teacher_outdir = outdir / args.dataset_name / args.train_domain / teacher_name
        metrics_json_path = teacher_outdir / "linear_probe_metrics.json"

        if args.skip_existing and metrics_json_path.exists():
            print(f"Skipping existing: {metrics_json_path}")
            continue

        domain_data = load_teacher_domain_npzs(
            npz_root=npz_root,
            dataset_name=args.dataset_name,
            teacher_name=teacher_name,
            domain_names=domain_names,
        )

        class_names, class_index_maps = verify_class_alignment(
            domain_data=domain_data,
            domain_names=domain_names,
            class_mode=args.class_mode,
            train_domain=args.train_domain,
        )

        print(f"Resolved {len(class_names)} classes")

        prepared: Dict[str, Dict[str, np.ndarray]] = {}
        for domain_name in domain_names:
            arr = domain_data[domain_name]

            feats = arr["feats"]
            labels = arr["labels"]
            paths = arr["paths"]

            if args.class_mode == "intersection":
                feats, labels, paths = subset_to_class_indices(
                    feats=feats,
                    labels=labels,
                    paths=paths,
                    keep_old_indices=class_index_maps[domain_name],
                )

            prepared[domain_name] = {
                "feats": feats,
                "labels": labels,
                "paths": paths,
            }

            print(
                f"  [{domain_name}] n={len(labels)} feat_dim={feats.shape[1]}"
            )

        train_arr = prepared[args.train_domain]
        x_all = train_arr["feats"]
        y_all = train_arr["labels"]
        p_all = train_arr["paths"]

        x_train, x_val, y_train, y_val, p_train, p_val = train_test_split(
            x_all,
            y_all,
            p_all,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y_all,
        )

        print("\nFitting linear probe...")
        print(f"  train samples: {len(y_train)}")
        print(f"  val samples:   {len(y_val)}")
        print(f"  feat dim:      {x_train.shape[1]}")

        clf = fit_linear_probe(
            x_train=x_train,
            y_train=y_train,
            c_value=args.probe_C,
            max_iter=args.probe_max_iter,
            random_state=args.random_state,
        )

        teacher_meta = domain_data[args.train_domain]["meta"].get("teacher_meta", {})
        model_tag = domain_data[args.train_domain]["meta"].get("model_tag", teacher_name)

        results: Dict[str, Any] = {
            "teacher_name": teacher_name,
            "model_tag": model_tag,
            "teacher_meta": teacher_meta,
            "probe": {
                "type": "LogisticRegression",
                "standardize_features": True,
                "C": args.probe_C,
                "max_iter": args.probe_max_iter,
                "test_size": args.test_size,
                "random_state": args.random_state,
            },
            "train_domain": args.train_domain,
            "class_names": class_names,
            "domains": {},
        }

        # held-out source split
        val_metrics = evaluate_probe_full(clf, x_val, y_val, p_val)
        results["domains"][args.train_domain] = {
            "split": "heldout_20_percent",
            **val_metrics,
        }

        print(
            f"[{args.train_domain}] "
            f"acc={val_metrics['accuracy']:.4f}  "
            f"bal_acc={val_metrics['balanced_accuracy']:.4f}"
        )

        if args.save_probe_outputs:
            proba_np = np.asarray(val_metrics["proba"], dtype=np.float32) if "proba" in val_metrics else None
            decision_np = (
                np.asarray(val_metrics["decision_function"], dtype=np.float32)
                if "decision_function" in val_metrics else None
            )

            save_probe_outputs_npz(
                teacher_outdir / f"probe_outputs_{args.train_domain}.npz",
                y_true=np.asarray(val_metrics["y_true"], dtype=np.int64),
                y_pred=np.asarray(val_metrics["y_pred"], dtype=np.int64),
                paths=np.asarray(val_metrics["paths"], dtype=object),
                proba=proba_np,
                decision_function=decision_np,
                meta={
                    "dataset_name": args.dataset_name,
                    "train_domain": args.train_domain,
                    "target_domain": args.train_domain,
                    "split": "heldout_20_percent",
                    "teacher_name": teacher_name,
                    "model_tag": model_tag,
                    "class_names": class_names,
                },
            )

        # full OOD target domains
        for domain_name in domain_names:
            if domain_name == args.train_domain:
                continue

            arr = prepared[domain_name]
            test_metrics = evaluate_probe_full(clf, arr["feats"], arr["labels"], arr["paths"])
            results["domains"][domain_name] = {
                "split": "full_domain_ood",
                **test_metrics,
            }

            print(
                f"[{domain_name}] "
                f"acc={test_metrics['accuracy']:.4f}  "
                f"bal_acc={test_metrics['balanced_accuracy']:.4f}"
            )

            if args.save_probe_outputs:
                proba_np = np.asarray(test_metrics["proba"], dtype=np.float32) if "proba" in test_metrics else None
                decision_np = (
                    np.asarray(test_metrics["decision_function"], dtype=np.float32)
                    if "decision_function" in test_metrics else None
                )

                save_probe_outputs_npz(
                    teacher_outdir / f"probe_outputs_{domain_name}.npz",
                    y_true=np.asarray(test_metrics["y_true"], dtype=np.int64),
                    y_pred=np.asarray(test_metrics["y_pred"], dtype=np.int64),
                    paths=np.asarray(test_metrics["paths"], dtype=object),
                    proba=proba_np,
                    decision_function=decision_np,
                    meta={
                        "dataset_name": args.dataset_name,
                        "train_domain": args.train_domain,
                        "target_domain": domain_name,
                        "split": "full_domain_ood",
                        "teacher_name": teacher_name,
                        "model_tag": model_tag,
                        "class_names": class_names,
                    },
                )

        save_json(metrics_json_path, results)
        summary["teachers"][teacher_name] = results["domains"]

    save_json(outdir / args.dataset_name / args.train_domain / "summary.json", summary)
    print("\nDone.")


if __name__ == "__main__":
    main()

# $domains = @("quickdraw", "real", "infograph", "sketch")

# python scripts\DomainNet\linear_probe_domainnet.py `
#   --npz_root "C:\Users\racs2019\Documents\NIPS-KD\teacher_npzs_domainnet" `
#   --dataset_name "domainnet" `
#   --train_domain "quickdraw" `
#   --domain_names $domains `
#   --class_mode strict `
#   --outdir "quickdraw" `
#   --save_probe_outputs