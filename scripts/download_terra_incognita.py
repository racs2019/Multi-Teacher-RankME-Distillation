#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path


IMAGES_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/eccv_18_all_images_sm.tar.gz"
)
ANNOTATIONS_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/eccv_18_annotations.tar.gz"
)

INCLUDE_LOCATIONS = {"38", "46", "100", "43"}
INCLUDE_CATEGORIES = {
    "bird",
    "bobcat",
    "cat",
    "coyote",
    "dog",
    "empty",
    "opossum",
    "rabbit",
    "raccoon",
    "squirrel",
}


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"Saved to {dst}")


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    print(f"Extracting {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)
    print(f"Extracted into {extract_to}")


def load_and_merge_annotations(annotation_dir: Path) -> dict:
    annotation_files = [
        annotation_dir / "cis_test_annotations.json",
        annotation_dir / "cis_val_annotations.json",
        annotation_dir / "train_annotations.json",
        annotation_dir / "trans_test_annotations.json",
        annotation_dir / "trans_val_annotations.json",
    ]

    merged = defaultdict(list)
    for file in annotation_files:
        with open(file, "r") as f:
            ann = json.load(f)
        for k, v in ann.items():
            merged[k].extend(v)

    return merged


def build_category_dict(categories: list[dict]) -> dict[int, str]:
    return {item["id"]: item["name"] for item in categories}


def build_image_to_categories(annotations: list[dict], category_dict: dict[int, str]) -> dict[int, list[str]]:
    image_to_categories = defaultdict(list)
    for ann in annotations:
        cat_name = category_dict[ann["category_id"]]
        image_to_categories[ann["image_id"]].append(cat_name)
    return image_to_categories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare the DomainBed-style TerraIncognita subset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory where terra_incognita will be created.",
    )
    parser.add_argument(
        "--keep_archives",
        action="store_true",
        help="Keep downloaded .tar.gz files after extraction.",
    )
    parser.add_argument(
        "--keep_raw",
        action="store_true",
        help="Keep raw extracted folders after restructuring.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_root = data_dir / "terra_incognita"
    out_root.mkdir(parents=True, exist_ok=True)

    images_tar = out_root / "terra_incognita_images.tar.gz"
    ann_tar = out_root / "terra_incognita_annotations.tar.gz"

    if not images_tar.exists():
        download_file(IMAGES_URL, images_tar)
    else:
        print(f"Skipping existing archive: {images_tar}")

    if not ann_tar.exists():
        download_file(ANNOTATIONS_URL, ann_tar)
    else:
        print(f"Skipping existing archive: {ann_tar}")

    images_dir = out_root / "eccv_18_all_images_sm"
    ann_dir = out_root / "eccv_18_annotation_files"

    if not images_dir.exists():
        extract_tar(images_tar, out_root)
    else:
        print(f"Skipping extraction, folder already exists: {images_dir}")

    if not ann_dir.exists():
        extract_tar(ann_tar, out_root)
    else:
        print(f"Skipping extraction, folder already exists: {ann_dir}")

    merged = load_and_merge_annotations(ann_dir)
    category_dict = build_category_dict(merged["categories"])
    image_to_categories = build_image_to_categories(merged["annotations"], category_dict)

    copied = 0
    skipped_location = 0
    skipped_category = 0

    for image in merged["images"]:
        image_id = image["id"]
        image_location = str(image["location"])
        image_fname = image["file_name"]

        if image_location not in INCLUDE_LOCATIONS:
            skipped_location += 1
            continue

        src_path = images_dir / image_fname
        if not src_path.exists():
            print(f"Warning: missing source image {src_path}")
            continue

        categories = image_to_categories.get(image_id, [])
        for category in categories:
            if category not in INCLUDE_CATEGORIES:
                skipped_category += 1
                continue

            dst_dir = out_root / f"location_{image_location}" / category
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / Path(image_fname).name

            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1

    print("\nDone.")
    print(f"Created dataset at: {out_root}")
    print(f"Copied files: {copied}")
    print(f"Skipped images from other locations: {skipped_location}")
    print(f"Skipped annotations from other categories: {skipped_category}")

    if not args.keep_archives:
        if images_tar.exists():
            images_tar.unlink()
        if ann_tar.exists():
            ann_tar.unlink()
        print("Removed downloaded archives.")

    if not args.keep_raw:
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if ann_dir.exists():
            shutil.rmtree(ann_dir)
        print("Removed raw extracted folders.")

    print("\nExpected final structure:")
    print(out_root / "location_38")
    print(out_root / "location_46")
    print(out_root / "location_100")
    print(out_root / "location_43")


if __name__ == "__main__":
    main()

# python download_terra_incognita.py --data_dir "C:\Users\racs2019\Documents\NIPS-KD\data"