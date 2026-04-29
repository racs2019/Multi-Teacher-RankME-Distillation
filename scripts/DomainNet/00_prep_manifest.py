#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


VALID_DOMAINS = {
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
}


def parse_split_file(txt_path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"Bad line in {txt_path} at line {line_idx}: {line}")

            rel_path, label_str = parts
            rows.append((rel_path.replace("\\", "/"), int(label_str)))

    return rows


def infer_class_name_from_relpath(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if len(parts) >= 3:
        return parts[1]
    return f"class_unknown"


def build_label_to_name_map(
    parsed: Dict[Tuple[str, str], List[Tuple[str, int]]]
) -> Dict[int, str]:
    votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rows in parsed.values():
        for rel_path, label in rows:
            class_name = infer_class_name_from_relpath(rel_path)
            votes[label][class_name] += 1

    label_to_name: Dict[int, str] = {}
    for label, counts in votes.items():
        label_to_name[label] = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    return label_to_name


def build_trunk_filename_index(domain_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}

    for split_name in ["train", "test"]:
        split_dir = domain_dir / split_name
        if not split_dir.exists():
            continue

        for trunk_dir in split_dir.iterdir():
            if not trunk_dir.is_dir():
                continue

            for file_path in trunk_dir.iterdir():
                if not file_path.is_file():
                    continue

                if file_path.name in index:
                    raise ValueError(
                        f"Duplicate filename found in trunked domain {domain_dir}: "
                        f"{file_path.name}"
                    )

                index[file_path.name] = file_path.resolve()

    return index


def build_special_domain_indexes(
    dataset_root: Path,
    domains: List[str],
) -> Dict[str, Dict[str, Path]]:
    """
    Handles cleaned clipart/painting layouts like:

        clipart/train/trunkXX/image.jpg
        clipart/test/trunkYY/image.jpg

    while official txt files may contain:

        clipart/class_name/image.jpg
    """
    indexes: Dict[str, Dict[str, Path]] = {}

    for domain in domains:
        if domain not in {"clipart", "painting"}:
            continue

        domain_dir = dataset_root / domain
        if domain_dir.exists():
            print(f"Indexing trunked domain: {domain}")
            indexes[domain] = build_trunk_filename_index(domain_dir)
            print(f"  indexed {len(indexes[domain])} files")

    return indexes


def resolve_abs_path(
    dataset_root: Path,
    rel_path: str,
    special_indexes: Dict[str, Dict[str, Path]],
) -> Path:
    rel = Path(rel_path)
    parts = rel.parts

    if len(parts) < 2:
        raise ValueError(f"Unexpected DomainNet relative path: {rel_path}")

    domain = parts[0].lower()

    direct_path = dataset_root / rel
    if direct_path.exists():
        return direct_path.resolve()

    if domain in special_indexes:
        filename = rel.name
        if filename in special_indexes[domain]:
            return special_indexes[domain][filename]

    return direct_path.resolve()


def write_csv(out_path: Path, rows: List[Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "domain",
        "split",
        "label",
        "class_name",
        "abs_path",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a clean DomainNet master manifest."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing the domainnet folder.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["real", "sketch", "infograph", "quickdraw"],
        help="DomainNet domains to include.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Splits to include.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to data_dir/domainnet/master_manifest.csv.",
    )
    args = parser.parse_args()

    domains = [d.lower() for d in args.domains]
    invalid = [d for d in domains if d not in VALID_DOMAINS]
    if invalid:
        raise ValueError(f"Invalid domains: {invalid}. Valid domains: {sorted(VALID_DOMAINS)}")

    dataset_root = Path(args.data_dir).expanduser().resolve() / "domainnet"
    if not dataset_root.exists():
        raise FileNotFoundError(f"DomainNet folder not found: {dataset_root}")

    meta_dir = dataset_root / "manifests" / "official_txts"
    if not meta_dir.exists():
        raise FileNotFoundError(
            f"Official split txt directory not found: {meta_dir}\n"
            "Expected files like real_train.txt, real_test.txt, sketch_train.txt, etc."
        )

    parsed: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}

    for domain in domains:
        for split in args.splits:
            txt_path = meta_dir / f"{domain}_{split}.txt"
            if not txt_path.exists():
                raise FileNotFoundError(f"Missing split file: {txt_path}")

            parsed[(domain, split)] = parse_split_file(txt_path)

    label_to_name = build_label_to_name_map(parsed)
    special_indexes = build_special_domain_indexes(dataset_root, domains)

    rows: List[Dict[str, object]] = []
    missing_count = 0

    for domain in domains:
        for split in args.splits:
            for rel_path, label in parsed[(domain, split)]:
                abs_path = resolve_abs_path(dataset_root, rel_path, special_indexes)

                if not abs_path.exists():
                    missing_count += 1
                    continue

                rows.append(
                    {
                        "domain": domain,
                        "split": split,
                        "label": int(label),
                        "class_name": label_to_name.get(int(label), f"class_{int(label):03d}"),
                        "abs_path": str(abs_path),
                    }
                )

    rows = sorted(
        rows,
        key=lambda r: (
            str(r["domain"]),
            str(r["split"]),
            int(r["label"]),
            str(r["abs_path"]),
        ),
    )

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else dataset_root / "master_manifest.csv"
    )

    write_csv(out_path, rows)

    print("\n=== DomainNet manifest summary ===")
    print(f"Domains: {domains}")
    print(f"Splits: {args.splits}")
    print(f"Existing rows written: {len(rows)}")
    print(f"Missing rows dropped: {missing_count}")

    labels_by_domain: Dict[str, set[int]] = defaultdict(set)
    counts_by_domain_split: Dict[Tuple[str, str], int] = defaultdict(int)

    for row in rows:
        domain = str(row["domain"])
        split = str(row["split"])
        label = int(row["label"])
        labels_by_domain[domain].add(label)
        counts_by_domain_split[(domain, split)] += 1

    for domain in domains:
        split_counts = " ".join(
            f"{split}={counts_by_domain_split[(domain, split)]}"
            for split in args.splits
        )
        print(
            f"{domain:10s} {split_counts} "
            f"labels={len(labels_by_domain[domain])}"
        )

    common_labels = set.intersection(
        *(labels_by_domain[d] for d in domains)
    ) if domains else set()

    print(f"Common label count across included domains: {len(common_labels)}")

    if len(common_labels) == 0:
        raise RuntimeError("No common labels across selected domains.")

    print("\nDone.")


if __name__ == "__main__":
    main()