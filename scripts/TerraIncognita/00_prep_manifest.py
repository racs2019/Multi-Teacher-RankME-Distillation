#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset_folder", default="terra_incognita")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["location_38", "location_43", "location_46", "location_100"],
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.data_dir) / args.dataset_folder
    if not root.exists():
        raise FileNotFoundError(f"Missing TerraIncognita root: {root}")

    # Build global class map across all domains.
    class_names = set()
    for domain in args.domains:
        domain_dir = root / domain
        if not domain_dir.exists():
            raise FileNotFoundError(f"Missing domain folder: {domain_dir}")

        for p in domain_dir.iterdir():
            if p.is_dir():
                class_names.add(p.name)

    class_names = sorted(class_names)
    class_to_label = {c: i for i, c in enumerate(class_names)}

    rows = []

    for domain in args.domains:
        domain_dir = root / domain
        domain_rows = []

        for class_name in class_names:
            class_dir = domain_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in sorted(class_dir.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                    domain_rows.append(
                        {
                            "domain": domain,
                            "label": class_to_label[class_name],
                            "class_name": class_name,
                            "abs_path": str(img_path.resolve()),
                        }
                    )

        if not domain_rows:
            raise RuntimeError(f"No images found for domain={domain}")

        df = pd.DataFrame(domain_rows)

        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=df["label"],
        )

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["split"] = "train"
        test_df["split"] = "test"

        rows.append(train_df)
        rows.append(test_df)

        print(
            f"{domain}: train={len(train_df)} test={len(test_df)} "
            f"classes={df['label'].nunique()}"
        )

    out = pd.concat(rows, ignore_index=True)
    out = out[["domain", "split", "label", "class_name", "abs_path"]]
    out = out.sort_values(["domain", "split", "label", "abs_path"])

    out_path = root / "master_manifest.csv"
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Total rows: {len(out)}")
    print(f"Classes: {len(class_names)}")


if __name__ == "__main__":
    main()