#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "This script requires open_clip_torch.\n"
        "Install with: pip install open_clip_torch"
    ) from e


# ============================================================
# Prompt templates
# ============================================================

PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a picture of a {}.",
    "a picture of the {}.",
    "an image of a {}.",
    "an image of the {}.",
    "a cropped photo of a {}.",
    "a close-up photo of a {}.",
    "a clean photo of a {}.",
    "a bright photo of a {}.",
    "a good photo of a {}.",
    "a centered photo of a {}.",
    "a studio photo of a {}.",
    "a product photo of a {}.",
]


def canonicalize_class_name(name: str) -> str:
    text = name.replace("_", " ").replace("/", " ").replace("-", " ")
    return " ".join(text.split()).lower()


# ============================================================
# Dataset wrapper
# ============================================================

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = default_loader(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label, path


# ============================================================
# Model wrapper
# ============================================================

class ZeroShotVLMWrapper(nn.Module):
    """
    Standard interface:
      forward(x) -> (features, logits)
    """

    def __init__(
        self,
        model: nn.Module,
        text_features: torch.Tensor,
        logit_scale: float | None = None,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("text_features", text_features)
        self.logit_scale_value = 100.0 if logit_scale is None else float(logit_scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.model.encode_image(x).float()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        text_features = self.text_features.float()
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits = self.logit_scale_value * image_features @ text_features.t()
        return image_features, logits


# ============================================================
# Teacher registry
# ============================================================

TEACHER_SPECS = {
    "openclip_l14_openai_qgelu": {
        "model_name": "ViT-L-14-quickgelu",
        "pretrained": "openai",
        "family": "openai_clip",
    },
    "openclip_b16_datacomp": {
        "model_name": "ViT-B-16",
        "pretrained": "datacomp_xl_s13b_b90k",
        "family": "datacomp",
    },
    "openclip_so400m_siglip": {
        "model_name": "ViT-SO400M-14-SigLIP",
        "pretrained": "webli",
        "family": "siglip",
    },
    "openclip_l14_dfn2b": {
        "model_name": "ViT-L-14",
        "pretrained": "dfn2b",
        "family": "dfn",
    },
    # New teachers
    "openclip_h14_laion2b": {
        "model_name": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
        "family": "laion_generalist",
    },
    "openclip_h14_378_dfn5b": {
        "model_name": "ViT-H-14-378-quickgelu",
        "pretrained": "dfn5b",
        "family": "dfn_specialist",
    },
    "openclip_convnext_xxlarge": {
        "model_name": "convnext_xxlarge",
        "pretrained": "laion2b_s34b_b82k_augreg_soup",
        "family": "convnext",
    },
}


# ============================================================
# Builders
# ============================================================

def print_requested_teacher_availability() -> None:
    wanted = [
        ("ViT-L-14-quickgelu", "openai"),
        ("ViT-B-16", "datacomp_xl_s13b_b90k"),
        ("ViT-SO400M-14-SigLIP", "webli"),
        ("ViT-L-14", "dfn2b"),
        ("ViT-H-14", "laion2b_s32b_b79k"),
        ("ViT-H-14-378-quickgelu", "dfn5b"),
        ("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"),
    ]

    try:
        available = set(open_clip.list_pretrained())
    except Exception as e:
        print(f"Warning: could not query open_clip.list_pretrained(): {e}")
        return

    print("\n=== Checking requested teachers in open_clip registry ===")
    for pair in wanted:
        status = "OK" if pair in available else "MISSING"
        print(f"{status:8s} model={pair[0]!r} pretrained={pair[1]!r}")
    print("=========================================================\n")


def build_openclip_model(
    class_names: List[str],
    model_name: str,
    pretrained: str,
    device: torch.device,
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    prompt_class_names = [canonicalize_class_name(c) for c in class_names]
    text_features_per_template = []

    with torch.no_grad():
        for template in PROMPT_TEMPLATES:
            prompts = [template.format(c) for c in prompt_class_names]
            tokens = tokenizer(prompts).to(device)

            feats = model.encode_text(tokens).float()
            feats = feats / feats.norm(dim=1, keepdim=True)
            text_features_per_template.append(feats)

        text_features = torch.stack(text_features_per_template, dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if hasattr(model, "logit_scale"):
            try:
                logit_scale = float(model.logit_scale.exp().item())
            except Exception:
                logit_scale = 100.0
        else:
            logit_scale = 100.0

    wrapper = ZeroShotVLMWrapper(
        model=model,
        text_features=text_features,
        logit_scale=logit_scale,
    )

    meta = {
        "model_family": "openclip",
        "model_name": model_name,
        "pretrained": pretrained,
        "prompt_class_names": prompt_class_names,
        "prompt_templates": PROMPT_TEMPLATES,
        "logit_scale": logit_scale,
    }
    return wrapper, preprocess, meta


def build_teacher(
    teacher_name: str,
    class_names: List[str],
    device: torch.device,
):
    if teacher_name not in TEACHER_SPECS:
        raise ValueError(
            f"Unknown teacher '{teacher_name}'. "
            f"Available: {sorted(TEACHER_SPECS.keys())}"
        )

    spec = TEACHER_SPECS[teacher_name]
    model, preprocess, meta = build_openclip_model(
        class_names=class_names,
        model_name=spec["model_name"],
        pretrained=spec["pretrained"],
        device=device,
    )
    meta["teacher_family_group"] = spec["family"]
    model_tag = f"{meta['model_family']}::{meta['model_name']}::{meta['pretrained']}"
    return model, preprocess, model_tag, meta


# ============================================================
# Extraction
# ============================================================

@torch.no_grad()
def extract_teacher_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, np.ndarray]:
    all_feats = []
    all_logits = []
    all_preds = []
    all_labels = []
    all_paths = []

    model.eval()

    for batch_idx, (images, labels, paths) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats, logits = model(images)
        preds = logits.argmax(dim=1)

        all_feats.append(feats.detach().cpu().numpy().astype(np.float32))
        all_logits.append(logits.detach().cpu().numpy().astype(np.float32))
        all_preds.append(preds.detach().cpu().numpy().astype(np.int64))
        all_labels.append(labels.detach().cpu().numpy().astype(np.int64))
        all_paths.extend(list(paths))

        if (batch_idx + 1) % 20 == 0:
            print(f"  processed {batch_idx + 1} batches")

    return {
        "feats": np.concatenate(all_feats, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "preds": np.concatenate(all_preds, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "paths": np.array(all_paths, dtype=object),
    }


def save_npz(out_path: Path, arrays: Dict[str, np.ndarray], meta: Dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        feats=arrays["feats"],
        logits=arrays["logits"],
        preds=arrays["preds"],
        labels=arrays["labels"],
        paths=arrays["paths"],
        meta_json=json.dumps(meta),
    )
    print(f"saved: {out_path}")


# ============================================================
# Helpers
# ============================================================

def build_dataset(domain_dir: Path, preprocess) -> ImageFolderWithPaths:
    return ImageFolderWithPaths(str(domain_dir), transform=preprocess)


def discover_domain_dirs(dataset_root: Path, domain_names: List[str] | None) -> List[Path]:
    if domain_names:
        domain_dirs = [dataset_root / d for d in domain_names]
    else:
        domain_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    if not domain_dirs:
        raise RuntimeError(f"No domain directories found under: {dataset_root}")

    for d in domain_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing domain folder: {d}")

    return domain_dirs


def get_classes_for_domain(domain_dir: Path) -> List[str]:
    return ImageFolder(str(domain_dir)).classes


def resolve_class_names(domain_dirs: List[Path], class_mode: str) -> List[str]:
    domain_classes = {str(d): get_classes_for_domain(d) for d in domain_dirs}

    if class_mode == "strict":
        reference = None
        for d in domain_dirs:
            names = domain_classes[str(d)]
            if reference is None:
                reference = names
            elif names != reference:
                raise RuntimeError(
                    f"Class mismatch in {d}\n"
                    f"Expected: {reference[:5]} ... ({len(reference)} classes)\n"
                    f"Got:      {names[:5]} ... ({len(names)} classes)\n\n"
                    f"Use --class_mode intersection if your dataset has only a common subset."
                )
        return reference

    if class_mode == "intersection":
        sets = [set(v) for v in domain_classes.values()]
        common = sorted(set.intersection(*sets))
        if not common:
            raise RuntimeError("No common classes across domains.")
        return common

    if class_mode == "from_target":
        return domain_classes[str(domain_dirs[0])]

    raise ValueError(f"Unknown class_mode: {class_mode}")


def filter_dataset_to_class_subset(dataset: ImageFolderWithPaths, class_names: List[str]) -> None:
    class_to_new_idx = {name: i for i, name in enumerate(class_names)}
    keep_classes = set(class_names)

    filtered_samples = []
    filtered_targets = []

    for path, old_label in dataset.samples:
        class_name = dataset.classes[old_label]
        if class_name in keep_classes:
            new_label = class_to_new_idx[class_name]
            filtered_samples.append((path, new_label))
            filtered_targets.append(new_label)

    if not filtered_samples:
        raise RuntimeError("No samples left after filtering to class subset.")

    dataset.classes = list(class_names)
    dataset.class_to_idx = class_to_new_idx
    dataset.samples = filtered_samples
    dataset.imgs = filtered_samples
    dataset.targets = filtered_targets


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract zero-shot teacher features/logits from arbitrary ImageFolder-style "
            "multi-domain datasets.\n\n"
            "Expected layout:\n"
            "  dataset_root/domain_name/class_name/image.xxx"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument(
        "--target_domain",
        type=str,
        required=True,
        help="Name of the domain folder to extract, relative to dataset_root",
    )
    parser.add_argument(
        "--domain_names",
        nargs="*",
        default=None,
        help=(
            "Optional list of all domain folder names. "
            "If omitted, all subdirectories under dataset_root are treated as domains."
        ),
    )
    parser.add_argument(
        "--class_mode",
        choices=["strict", "intersection", "from_target"],
        default="strict",
        help=(
            "How to define class vocabulary across domains:\n"
            "  strict       -> all domain class lists must match exactly\n"
            "  intersection -> use only classes common to all domains\n"
            "  from_target  -> use the target domain's class list only"
        ),
    )
    parser.add_argument("--dataset_name", type=str, default="custom")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--teachers",
        nargs="+",
        default=[
            "openclip_l14_openai_qgelu",
            "openclip_b16_datacomp",
            "openclip_so400m_siglip",
            "openclip_l14_dfn2b",
            "openclip_h14_laion2b",
            "openclip_h14_378_dfn5b",
            "openclip_convnext_xxlarge",
        ],
        choices=sorted(TEACHER_SPECS.keys()),
    )
    parser.add_argument("--max_batches", type=int, default=None, help="Debug only")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--check_registry_only", action="store_true")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    device = torch.device(args.device if use_cuda or args.device == "cpu" else "cpu")
    print(f"using device: {device}")

    print_requested_teacher_availability()
    if args.check_registry_only:
        print("Exiting after registry check because --check_registry_only was set.")
        return

    dataset_root = Path(args.dataset_root)
    outdir = Path(args.outdir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    domain_dirs = discover_domain_dirs(dataset_root, args.domain_names)
    domain_map = {d.name: d for d in domain_dirs}

    if args.target_domain not in domain_map:
        raise FileNotFoundError(
            f"Target domain '{args.target_domain}' not found.\n"
            f"Available domains: {sorted(domain_map.keys())}"
        )

    target_domain_dir = domain_map[args.target_domain]

    ordered_domain_dirs = [target_domain_dir] + [d for d in domain_dirs if d != target_domain_dir]
    class_names = resolve_class_names(ordered_domain_dirs, args.class_mode)
    num_classes = len(class_names)

    print(f"resolved {num_classes} classes using class_mode={args.class_mode}")
    print(f"target domain: {target_domain_dir}")

    safe_dataset = args.dataset_name.replace(" ", "_").lower()
    safe_domain = args.target_domain.replace(" ", "_").lower()

    for teacher_name in args.teachers:
        save_path = outdir / f"{safe_dataset}__{teacher_name}__{safe_domain}.npz"

        if args.skip_existing and save_path.exists():
            print(f"\n=== skipping existing teacher: {teacher_name} ===")
            print(f"  exists: {save_path}")
            continue

        print(f"\n=== extracting teacher: {teacher_name} ===")
        model, preprocess, model_tag, teacher_meta = build_teacher(
            teacher_name=teacher_name,
            class_names=class_names,
            device=device,
        )

        model = model.to(device)
        model.eval()

        dataset = build_dataset(target_domain_dir, preprocess)
        if args.class_mode != "strict":
            filter_dataset_to_class_subset(dataset, class_names)

        loader = make_loader(dataset, args.batch_size, args.num_workers)

        arrays = extract_teacher_outputs(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )

        top1_acc = float((arrays["preds"] == arrays["labels"]).mean())

        meta = {
            "dataset_name": args.dataset_name,
            "teacher_name": teacher_name,
            "model_tag": model_tag,
            "domain": args.target_domain,
            "all_domains": [d.name for d in domain_dirs],
            "class_mode": args.class_mode,
            "num_classes": num_classes,
            "class_names": class_names,
            "num_samples": int(arrays["labels"].shape[0]),
            "feature_dim": int(arrays["feats"].shape[1]),
            "logit_dim": int(arrays["logits"].shape[1]),
            "zero_shot_top1_acc": top1_acc,
            "teacher_meta": teacher_meta,
        }

        print(f"  zero-shot top1 acc: {top1_acc:.4f}")
        save_npz(save_path, arrays, meta)

    print("\ndone.")


if __name__ == "__main__":
    main()

# $domains = @("location_38", "location_43", "location_46", "location_100")

# foreach ($d in $domains) {
#     python extract_teachers.py `
#         --dataset_root "C:\Users\racs2019\Documents\NIPS-KD\data\terra_incognita" `
#         --dataset_name terra_incognita `
#         --target_domain $d `
#         --domain_names location_38 location_43 location_46 location_100 `
#         --class_mode strict `
#         --outdir teacher_npzs `
#         --batch_size 64 `
#         --num_workers 4 `
#         --teachers `
#             openclip_l14_openai_qgelu `
#             openclip_b16_datacomp `
#             openclip_so400m_siglip `
#             openclip_l14_dfn2b `
#             openclip_h14_laion2b `
#             openclip_h14_378_dfn5b `
#             openclip_convnext_xxlarge `
#         --skip_existing
# }