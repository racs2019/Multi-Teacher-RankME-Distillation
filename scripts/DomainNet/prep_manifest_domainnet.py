#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import requests


SPLIT_BASE_URL = "https://ai.bu.edu/M3SDA/DomainNet"
VALID_DOMAINS = {
    "clipart",
    "infograph",
    "painting",
    "quickdraw",
    "real",
    "sketch",
}


def download(url: str, dst: Path, timeout: int = 60) -> None:
    if dst.exists():
        print(f"Skipping existing file: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")

    with requests.get(url, timeout=timeout) as response:
        response.raise_for_status()
        dst.write_bytes(response.content)

    print(f"Saved: {dst}")


def split_txt_url(domain: str, split: str) -> str:
    return f"{SPLIT_BASE_URL}/{domain}_{split}.txt"


def split_txt_path(meta_dir: Path, domain: str, split: str) -> Path:
    return meta_dir / f"{domain}_{split}.txt"


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


def infer_class_name_from_relpath(rel_path: str) -> str | None:
    parts = Path(rel_path).parts
    if len(parts) >= 3:
        return parts[1]
    return None


def build_label_to_name_map(
    parsed_by_domain_split: Dict[Tuple[str, str], List[Tuple[str, int]]]
) -> Dict[int, str]:
    votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for (_, _), rows in parsed_by_domain_split.items():
        for rel_path, label in rows:
            cname = infer_class_name_from_relpath(rel_path)
            if cname is not None:
                votes[label][cname] += 1

    label_to_name: Dict[int, str] = {}
    for label, name_counts in votes.items():
        best_name = sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        label_to_name[label] = best_name

    return label_to_name


def build_trunk_filename_index(domain_dir: Path) -> Dict[str, str]:
    """
    For cleaned clipart/painting:
      domain/train/trunkXX/<filename>
      domain/test/trunkYY/<filename>

    Build filename -> absolute path map.
    """
    index: Dict[str, str] = {}
    for split_name in ["train", "test"]:
        split_dir = domain_dir / split_name
        if not split_dir.exists():
            continue

        for trunk_dir in split_dir.iterdir():
            if not trunk_dir.is_dir():
                continue

            for f in trunk_dir.iterdir():
                if not f.is_file():
                    continue
                if f.name in index:
                    raise ValueError(
                        f"Duplicate filename detected while indexing {domain_dir}: {f.name}"
                    )
                index[f.name] = str(f.resolve())

    return index


def build_special_domain_indexes(dataset_root: Path, domains: List[str]) -> Dict[str, Dict[str, str]]:
    special_indexes: Dict[str, Dict[str, str]] = {}

    for domain in domains:
        if domain not in {"clipart", "painting"}:
            continue

        domain_dir = dataset_root / domain
        if not domain_dir.exists():
            continue

        print(f"Building filename index for {domain} trunks...")
        idx = build_trunk_filename_index(domain_dir)
        print(f"{domain}: indexed {len(idx)} filenames")
        special_indexes[domain] = idx

    return special_indexes


def resolve_abs_path(
    dataset_root: Path,
    rel_path: str,
    special_indexes: Dict[str, Dict[str, str]],
) -> Path:
    """
    Resolve official DomainNet rel_path to actual local file path.

    Works for:
      real/sketch/infograph/quickdraw:
        dataset_root / rel_path

      cleaned clipart/painting:
        rel_path = clipart/class_name/file.jpg
        actual file lives in clipart/train|test/trunkXX/file.jpg
        so resolve by filename lookup.
    """
    rel_path_obj = Path(rel_path)
    parts = rel_path_obj.parts
    if len(parts) < 2:
        raise ValueError(f"Unexpected rel_path: {rel_path}")

    domain = parts[0].lower()
    direct = dataset_root / rel_path_obj
    if direct.exists():
        return direct.resolve()

    if domain in {"clipart", "painting"}:
        fname = rel_path_obj.name
        domain_index = special_indexes.get(domain, {})
        if fname in domain_index:
            return Path(domain_index[fname])

    return direct.resolve()


def write_manifest_csv(out_path: Path, rows: List[Dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "domain",
        "split",
        "label",
        "class_name",
        "rel_path",
        "abs_path",
        "exists",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved manifest: {out_path}")


def write_json(out_path: Path, obj: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved JSON: {out_path}")


def subset_rows_per_label(
    rows: List[Dict],
    max_per_class: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    by_label: Dict[int, List[Dict]] = defaultdict(list)

    for row in rows:
        by_label[int(row["label"])].append(row)

    subset: List[Dict] = []
    for label in sorted(by_label.keys()):
        cur = by_label[label]
        if len(cur) > max_per_class:
            cur = rng.sample(cur, max_per_class)
        subset.extend(cur)

    subset = sorted(subset, key=lambda r: (int(r["label"]), r["rel_path"]))
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare official DomainNet manifests from official split txt files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing domainnet/ with extracted image folders.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["real", "sketch", "clipart", "painting"],
        help="Subset of DomainNet domains to prepare.",
    )
    parser.add_argument(
        "--download_txts",
        action="store_true",
        help="Download official *_train.txt and *_test.txt files.",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Also write subset manifests with up to --max_per_class per label per domain.",
    )
    parser.add_argument("--max_per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    requested_domains = [d.lower() for d in args.domains]
    invalid = [d for d in requested_domains if d not in VALID_DOMAINS]
    if invalid:
        raise ValueError(f"Invalid domains: {invalid}. Valid: {sorted(VALID_DOMAINS)}")

    dataset_root = Path(args.data_dir).expanduser().resolve() / "domainnet"
    if not dataset_root.exists():
        raise FileNotFoundError(f"DomainNet root not found: {dataset_root}")

    meta_dir = dataset_root / "manifests" / "official_txts"
    out_dir = dataset_root / "manifests"

    if args.download_txts:
        for domain in requested_domains:
            for split in ["train", "test"]:
                url = split_txt_url(domain, split)
                dst = split_txt_path(meta_dir, domain, split)
                download(url, dst)

    parsed_by_domain_split: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}

    for domain in requested_domains:
        for split in ["train", "test"]:
            txt_path = split_txt_path(meta_dir, domain, split)
            if not txt_path.exists():
                raise FileNotFoundError(
                    f"Missing official split file: {txt_path}\n"
                    f"Download it manually or rerun with --download_txts."
                )
            parsed_by_domain_split[(domain, split)] = parse_split_file(txt_path)

    label_to_name = build_label_to_name_map(parsed_by_domain_split)
    special_indexes = build_special_domain_indexes(dataset_root, requested_domains)

    labels_per_domain: Dict[str, set[int]] = {}
    summary: Dict[str, Dict] = {}

    for domain in requested_domains:
        train_rows: List[Dict] = []
        test_rows: List[Dict] = []
        all_rows: List[Dict] = []

        domain_labels: set[int] = set()
        missing_files = 0

        for split in ["train", "test"]:
            rows = parsed_by_domain_split[(domain, split)]

            cur_out: List[Dict] = []
            for rel_path, label in rows:
                abs_path = resolve_abs_path(dataset_root, rel_path, special_indexes)
                exists = abs_path.exists()
                if not exists:
                    missing_files += 1

                domain_labels.add(label)

                cur_out.append(
                    {
                        "domain": domain,
                        "split": split,
                        "label": int(label),
                        "class_name": label_to_name.get(label, f"class_{label:03d}"),
                        "rel_path": rel_path,
                        "abs_path": str(abs_path),
                        "exists": int(exists),
                    }
                )

            if split == "train":
                train_rows = cur_out
            else:
                test_rows = cur_out

            all_rows.extend(cur_out)

        labels_per_domain[domain] = domain_labels

        write_manifest_csv(out_dir / f"{domain}_train_manifest.csv", train_rows)
        write_manifest_csv(out_dir / f"{domain}_test_manifest.csv", test_rows)
        write_manifest_csv(out_dir / f"{domain}_all_manifest.csv", all_rows)

        if args.subset:
            subset_rows = subset_rows_per_label(
                [r for r in all_rows if int(r["exists"]) == 1],
                max_per_class=args.max_per_class,
                seed=args.seed,
            )
            write_manifest_csv(
                out_dir / f"{domain}_all_subset_{args.max_per_class}_manifest.csv",
                subset_rows,
            )

        summary[domain] = {
            "num_train_rows": len(train_rows),
            "num_test_rows": len(test_rows),
            "num_all_rows": len(all_rows),
            "num_existing_files": int(sum(int(r["exists"]) for r in all_rows)),
            "num_missing_files": int(sum(1 - int(r["exists"]) for r in all_rows)),
            "num_unique_labels": len(domain_labels),
        }

    common_labels = set.intersection(*(labels_per_domain[d] for d in requested_domains))
    common_label_names = {
        int(label): label_to_name.get(int(label), f"class_{int(label):03d}")
        for label in sorted(common_labels)
    }

    global_report = {
        "dataset_root": str(dataset_root),
        "domains": requested_domains,
        "per_domain_summary": summary,
        "common_label_count": len(common_labels),
        "common_labels": [int(x) for x in sorted(common_labels)],
        "common_label_names": common_label_names,
    }

    write_json(out_dir / "domainnet_manifest_summary.json", global_report)

    print("\n=== DomainNet manifest summary ===")
    for domain in requested_domains:
        s = summary[domain]
        print(
            f"{domain:10s} "
            f"train={s['num_train_rows']:7d} "
            f"test={s['num_test_rows']:7d} "
            f"all={s['num_all_rows']:7d} "
            f"labels={s['num_unique_labels']:4d} "
            f"missing={s['num_missing_files']:6d}"
        )

    print(f"\nCommon label count across domains: {len(common_labels)}")
    if len(common_labels) <= 20:
        print("Common labels:", [common_label_names[x] for x in sorted(common_label_names.keys())])

    if len(common_labels) == 0:
        raise RuntimeError(
            "No common labels across selected domains according to the official txt lists."
        )

    print("\nDone.")
    print(f"Manifests written under: {out_dir}")


if __name__ == "__main__":
    main()

# python scripts/DomainNet/prep_manifest_domainnet.py `
#   --data_dir "C:\Users\racs2019\Documents\NIPS-KD\data" `
#   --domains real sketch infograph quickdraw `
#   --subset `
#   --max_per_class 100