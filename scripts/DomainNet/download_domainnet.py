#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Set

import requests


DOMAINNET_URLS = {
    "clipart":   "https://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip",
    "infograph": "https://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
    "painting":  "https://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip",
    "quickdraw": "https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
    "real":      "https://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
    "sketch":    "https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def download(url: str, dst: Path, timeout: int = 60) -> None:
    if dst.exists():
        print(f"Skipping existing archive: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(dst, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = 100.0 * downloaded / total
                    print(
                        f"\r  {downloaded / (1024**3):.2f} GB / {total / (1024**3):.2f} GB "
                        f"({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

    if total > 0:
        print()
    print(f"Saved to {dst}")


def safe_extract(zip_path: Path, extract_to: Path) -> None:
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted into {extract_to}")


def find_domain_dir(root: Path, domain: str) -> Path:
    direct = root / domain
    if direct.exists() and direct.is_dir():
        return direct

    matches = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == domain.lower():
            matches.append(p)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches = sorted(matches, key=lambda x: (len(x.parts), str(x)))
        return matches[0]

    raise FileNotFoundError(f"Could not find extracted folder for domain '{domain}' under {root}")


def resolve_class_root(domain_dir: Path) -> Path:
    """
    Return the directory whose immediate children are class folders.

    Handles:
      domain/class_name/...
      domain/train/class_name/...
    """
    direct_subdirs = sorted([p for p in domain_dir.iterdir() if p.is_dir()])
    direct_names = {p.name.lower() for p in direct_subdirs}

    if "train" in direct_names:
        train_dir = domain_dir / "train"
        train_subdirs = [p for p in train_dir.iterdir() if p.is_dir()]
        if len(train_subdirs) > 10:
            return train_dir

    if len(direct_subdirs) > 10:
        return domain_dir

    raise RuntimeError(
        f"Could not resolve class root for domain directory: {domain_dir}\n"
        f"Immediate subdirectories: {[p.name for p in direct_subdirs[:20]]}"
    )


def count_images_in_class(class_dir: Path) -> int:
    return sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def get_domain_class_map(domain_dir: Path) -> Dict[str, int]:
    class_root = resolve_class_root(domain_dir)
    out: Dict[str, int] = {}

    for cls_dir in sorted([p for p in class_root.iterdir() if p.is_dir()]):
        n = count_images_in_class(cls_dir)
        if n > 0:
            out[cls_dir.name] = n

    return out


def validate_domains(root: Path, domains: List[str]) -> Dict[str, Dict[str, int]]:
    class_maps: Dict[str, Dict[str, int]] = {}

    print("\n=== Validating extracted domains ===")
    for domain in domains:
        domain_dir = find_domain_dir(root, domain)
        class_root = resolve_class_root(domain_dir)
        class_map = get_domain_class_map(domain_dir)
        class_maps[domain] = class_map

        print(
            f"{domain:14s} classes={len(class_map):5d} "
            f"images={sum(class_map.values()):8d} "
            f"domain_path={domain_dir} "
            f"class_root={class_root}"
        )

        if len(class_map) == 0:
            raise RuntimeError(f"Domain '{domain}' has zero non-empty classes: {domain_dir}")

    common: Set[str] = set(class_maps[domains[0]].keys())
    for d in domains[1:]:
        common &= set(class_maps[d].keys())

    print(f"\nCommon classes across {domains}: {len(common)}")
    if len(common) <= 20:
        print("Common class names:", sorted(common))
    else:
        preview = sorted(common)[:20]
        print("Common class preview:", preview, "...")

    if len(common) <= 1:
        print("\n[ERROR] Common class count is <= 1. Folder structure is still not usable.")
    else:
        print("[OK] Shared class vocabulary looks plausible.")

    return class_maps


def copy_subset_for_shared_classes(
    root: Path,
    domains: List[str],
    class_maps: Dict[str, Dict[str, int]],
    max_per_class: int,
    seed: int,
) -> None:
    rng = random.Random(seed)

    common: Set[str] = set(class_maps[domains[0]].keys())
    for d in domains[1:]:
        common &= set(class_maps[d].keys())
    common = set(sorted(common))

    if len(common) == 0:
        raise RuntimeError("No common classes across domains; cannot build subsets.")

    print(f"\nBuilding shared-class subsets using {len(common)} common classes")

    for domain in domains:
        src_domain = resolve_class_root(find_domain_dir(root, domain))
        dst_domain = root / f"{domain}_subset"

        if dst_domain.exists():
            print(f"Removing existing subset folder: {dst_domain}")
            shutil.rmtree(dst_domain)

        copied_images = 0
        copied_classes = 0

        for class_name in sorted(common):
            src_cls = src_domain / class_name
            if not src_cls.exists():
                continue

            imgs = [p for p in src_cls.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if not imgs:
                continue

            if len(imgs) > max_per_class:
                imgs = rng.sample(imgs, max_per_class)

            dst_cls = dst_domain / class_name
            dst_cls.mkdir(parents=True, exist_ok=True)

            for img in imgs:
                shutil.copy2(img, dst_cls / img.name)

            copied_images += len(imgs)
            copied_classes += 1

        print(
            f"{domain:14s} subset_classes={copied_classes:5d} "
            f"subset_images={copied_images:8d} "
            f"path={dst_domain}"
        )


def save_audit_report(root: Path, domains: List[str], class_maps: Dict[str, Dict[str, int]]) -> None:
    common = set(class_maps[domains[0]].keys())
    for d in domains[1:]:
        common &= set(class_maps[d].keys())

    report = {
        "domains": domains,
        "per_domain_num_classes": {d: len(class_maps[d]) for d in domains},
        "per_domain_num_images": {d: int(sum(class_maps[d].values())) for d in domains},
        "common_class_count": len(common),
        "common_classes_preview": sorted(list(common))[:100],
    }

    out_path = root / "domainnet_audit_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved audit report: {out_path}")


def maybe_remove_existing(root: Path, domain: str) -> None:
    candidates = [
        root / domain,
        root / f"{domain}_subset",
    ]

    for candidate in candidates:
        if candidate.exists():
            print(f"Removing existing folder: {candidate}")
            shutil.rmtree(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, validate, and optionally subset DomainNet.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["real", "sketch", "clipart", "painting"],
        help="Allowed: clipart infograph painting quickdraw real sketch",
    )
    parser.add_argument("--subset", action="store_true", help="Create shared-class subset folders.")
    parser.add_argument("--max_per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep_archives", action="store_true")
    parser.add_argument("--force_reextract", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--clean_domains_first", action="store_true")
    args = parser.parse_args()

    requested_domains = [d.lower() for d in args.domains]
    invalid = [d for d in requested_domains if d not in DOMAINNET_URLS]
    if invalid:
        raise ValueError(
            f"Invalid domain(s): {invalid}. Valid options: {sorted(DOMAINNET_URLS.keys())}"
        )

    root = Path(args.data_dir).expanduser().resolve() / "domainnet"
    root.mkdir(parents=True, exist_ok=True)

    print(f"Target root: {root}")
    print(f"Domains: {requested_domains}")

    for domain in requested_domains:
        if args.clean_domains_first:
            maybe_remove_existing(root, domain)

        zip_path = root / f"{domain}.zip"

        if not args.skip_download:
            download(DOMAINNET_URLS[domain], zip_path)

        extracted_dir = root / domain
        if args.force_reextract or not extracted_dir.exists():
            if not zip_path.exists():
                raise FileNotFoundError(
                    f"Archive not found for extraction: {zip_path}\n"
                    f"Either re-download, or use --skip_download only when extracted folders already exist."
                )
            safe_extract(zip_path, root)
        else:
            print(f"Skipping extraction, folder already exists: {extracted_dir}")

    class_maps = validate_domains(root, requested_domains)
    save_audit_report(root, requested_domains, class_maps)

    if args.subset:
        copy_subset_for_shared_classes(
            root=root,
            domains=requested_domains,
            class_maps=class_maps,
            max_per_class=args.max_per_class,
            seed=args.seed,
        )

        subset_domains = [f"{d}_subset" for d in requested_domains]
        print("\nRe-validating subset folders...")
        subset_maps = validate_domains(root, subset_domains)
        save_audit_report(root, subset_domains, subset_maps)

    if not args.keep_archives:
        for domain in requested_domains:
            zip_path = root / f"{domain}.zip"
            if zip_path.exists():
                zip_path.unlink()
                print(f"Removed archive: {zip_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()