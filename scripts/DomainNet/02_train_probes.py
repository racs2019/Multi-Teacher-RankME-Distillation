#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TEACHERS_DEFAULT = [
    "openclip_l14_openai_qgelu",
    "openclip_b16_datacomp",
    "openclip_so400m_siglip",
    "openclip_l14_dfn2b",
    "openclip_h14_laion2b",
    "openclip_h14_378_dfn5b",
    "openclip_convnext_xxlarge",
]


def load_features(feature_root: Path, domain: str, split: str, teacher: str):
    path = feature_root / domain / split / f"{teacher}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    z = np.load(path, allow_pickle=True)

    return {
        "feats": z["feats"].astype(np.float32),
        "labels": z["labels"].astype(np.int64),
        "paths": z["paths"],
    }


def fit_probe(x_train, y_train, seed: int, c: float, max_iter: int):
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    solver="lbfgs",
                    random_state=seed,
                    n_jobs=None,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    return clf


def evaluate(clf, x, y):
    proba = clf.predict_proba(x).astype(np.float32)
    pred = proba.argmax(axis=1).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "pred": pred,
        "proba": proba,
    }


def save_probe_outputs(out_path: Path, y_true, y_pred, proba, paths, meta):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        y_true=y_true.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        proba=proba.astype(np.float32),
        paths=np.asarray(paths, dtype=object),
        meta_json=json.dumps(meta),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dataset", default="domainnet")
    parser.add_argument("--source", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--teachers", nargs="+", default=TEACHERS_DEFAULT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--probe_C", type=float, default=1.0)
    parser.add_argument("--probe_max_iter", type=int, default=2000)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    outdir = Path(args.outdir) / args.dataset / args.source
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in args.seeds:
        for teacher in args.teachers:
            print("\n" + "=" * 80)
            print(f"seed={seed} teacher={teacher}")
            print("=" * 80)

            train = load_features(
                feature_root=feature_root,
                domain=args.source,
                split=args.train_split,
                teacher=teacher,
            )

            clf = fit_probe(
                train["feats"],
                train["labels"],
                seed=seed,
                c=args.probe_C,
                max_iter=args.probe_max_iter,
            )

            for target in args.targets:
                test = load_features(
                    feature_root=feature_root,
                    domain=target,
                    split=args.test_split,
                    teacher=teacher,
                )

                out_npz = (
                    outdir
                    / "probe_outputs"
                    / f"seed_{seed}"
                    / target
                    / f"{teacher}.npz"
                )

                if args.skip_existing and out_npz.exists():
                    print(f"Skipping existing: {out_npz}")
                    continue

                result = evaluate(clf, test["feats"], test["labels"])

                save_probe_outputs(
                    out_npz,
                    y_true=test["labels"],
                    y_pred=result["pred"],
                    proba=result["proba"],
                    paths=test["paths"],
                    meta={
                        "dataset": args.dataset,
                        "source": args.source,
                        "target": target,
                        "seed": seed,
                        "teacher": teacher,
                        "train_split": args.train_split,
                        "test_split": args.test_split,
                        "method": "linear_probe",
                    },
                )

                rows.append(
                    {
                        "dataset": args.dataset,
                        "source": args.source,
                        "target": target,
                        "seed": seed,
                        "teacher": teacher,
                        "method": "linear_probe",
                        "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"],
                        "n_samples": int(len(test["labels"])),
                        "output_npz": str(out_npz),
                    }
                )

                print(
                    f"[{target}] acc={result['accuracy']:.4f} "
                    f"macro_f1={result['macro_f1']:.4f}"
                )

    results_csv = outdir / "linear_probe_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    print(f"\nSaved results: {results_csv}")


if __name__ == "__main__":
    main()