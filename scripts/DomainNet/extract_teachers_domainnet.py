#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "This script requires open_clip_torch.\n"
        "Install with: pip install open_clip_torch"
    ) from e


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
    text = str(name).replace("_", " ").replace("/", " ").replace("-", " ")
    return " ".join(text.split()).lower()


def canonicalize_domain_name(name: str) -> str:
    name = str(name).strip().lower()
    if name.endswith("_subset"):
        name = name[:-7]
    return name


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, transform):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform

        required_cols = {"abs_path", "label", "class_name", "domain", "exists"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        self.df = self.df[self.df["exists"] == 1].copy().reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No existing samples in manifest: {manifest_csv}")

        self.class_names = [
            name for _, name in sorted(
                self.df[["label", "class_name"]].drop_duplicates().values.tolist(),
                key=lambda x: int(x[0]),
            )
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = str(row["abs_path"])
        label = int(row["label"])

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label, path


class ZeroShotVLMWrapper(nn.Module):
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


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def load_manifest_summary(manifest_summary_json: Path) -> Dict:
    with open(manifest_summary_json, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract zero-shot teacher features/logits from DomainNet manifest CSVs."
    )
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--manifest_summary_json", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="domainnet")
    parser.add_argument("--target_domain", type=str, required=True)
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
    parser.add_argument("--max_batches", type=int, default=None)
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

    manifest_csv = Path(args.manifest_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    manifest_summary = load_manifest_summary(Path(args.manifest_summary_json))
    common_labels = manifest_summary["common_labels"]
    common_label_names = manifest_summary["common_label_names"]

    class_names = [common_label_names[str(i)] if str(i) in common_label_names else common_label_names[i] for i in common_labels]
    num_classes = len(class_names)

    print(f"resolved {num_classes} classes from manifest summary")
    print(f"target domain: {args.target_domain}")
    print(f"manifest: {manifest_csv}")

    safe_dataset = args.dataset_name.replace(" ", "_").lower()
    safe_domain = canonicalize_domain_name(args.target_domain).replace(" ", "_").lower()

    for teacher_name in args.teachers:
        save_path = Path(args.outdir) / f"{safe_dataset}__{teacher_name}__{safe_domain}.npz"

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

        dataset = ManifestDataset(manifest_csv=manifest_csv, transform=preprocess)
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
            "domain": canonicalize_domain_name(args.target_domain),
            "raw_domain_name": args.target_domain,
            "class_mode": "official_manifest_common_labels",
            "num_classes": num_classes,
            "class_names": class_names,
            "common_labels": common_labels,
            "num_samples": int(arrays["labels"].shape[0]),
            "feature_dim": int(arrays["feats"].shape[1]),
            "logit_dim": int(arrays["logits"].shape[1]),
            "zero_shot_top1_acc": top1_acc,
            "teacher_meta": teacher_meta,
            "manifest_csv": str(manifest_csv),
        }

        print(f"  zero-shot top1 acc: {top1_acc:.4f}")
        save_npz(save_path, arrays, meta)

    print("\ndone.")


if __name__ == "__main__":
    main()

# $domains = @("real", "sketch", "infograph", "quickdraw")

# foreach ($d in $domains) {
#     python scripts/DomainNet/extract_teachers_domainnet.py `
#         --manifest_csv "C:\Users\racs2019\Documents\NIPS-KD\data\domainnet\manifests\${d}_all_subset_100_manifest.csv" `
#         --manifest_summary_json "C:\Users\racs2019\Documents\NIPS-KD\data\domainnet\manifests\domainnet_manifest_summary.json" `
#         --dataset_name domainnet `
#         --target_domain $d `
#         --outdir "teacher_npzs_domainnet" `
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