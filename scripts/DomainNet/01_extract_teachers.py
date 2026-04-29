#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import open_clip


TEACHER_SPECS = {
    "openclip_l14_openai_qgelu": ("ViT-L-14-quickgelu", "openai"),
    "openclip_b16_datacomp": ("ViT-B-16", "datacomp_xl_s13b_b90k"),
    "openclip_so400m_siglip": ("ViT-SO400M-14-SigLIP", "webli"),
    "openclip_l14_dfn2b": ("ViT-L-14", "dfn2b"),
    "openclip_h14_laion2b": ("ViT-H-14", "laion2b_s32b_b79k"),
    "openclip_h14_378_dfn5b": ("ViT-H-14-378-quickgelu", "dfn5b"),
    "openclip_convnext_xxlarge": ("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"),
}


def clean_class_name(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ").replace("/", " ").lower()


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["abs_path"]).convert("RGB")
        img = self.transform(img)
        return img, int(row["label"]), str(row["abs_path"])


@torch.no_grad()
def build_text_features(model, tokenizer, class_names, device):
    prompts = [f"a photo of a {clean_class_name(c)}." for c in class_names]
    tokens = tokenizer(prompts).to(device)

    text_features = model.encode_text(tokens).float()
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale.exp()
    else:
        logit_scale = torch.tensor(100.0, device=device)

    return text_features, logit_scale


@torch.no_grad()
def extract(model, loader, device, text_features, logit_scale):
    feats, logits, preds, labels, paths = [], [], [], [], []

    model.eval()

    for batch_idx, (images, y, p) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        f = model.encode_image(images).float()
        f = f / f.norm(dim=1, keepdim=True)

        l = logit_scale * (f @ text_features.T)
        pred = l.argmax(dim=1)

        feats.append(f.detach().cpu().numpy().astype(np.float32))
        logits.append(l.detach().cpu().numpy().astype(np.float32))
        preds.append(pred.detach().cpu().numpy().astype(np.int64))
        labels.append(y.detach().cpu().numpy().astype(np.int64))
        paths.extend(list(p))

        if (batch_idx + 1) % 20 == 0:
            print(f"  processed {batch_idx + 1} batches")

    return {
        "feats": np.concatenate(feats, axis=0),
        "logits": np.concatenate(logits, axis=0),
        "preds": np.concatenate(preds, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "paths": np.asarray(paths, dtype=object),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher", required=True, choices=sorted(TEACHER_SPECS.keys()))
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    manifest_csv = Path(args.manifest_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_csv}")

    out_path = Path(args.outdir) / args.domain / args.split / f"{args.teacher}.npz"
    if args.skip_existing and out_path.exists():
        print(f"Skipping existing: {out_path}")
        return

    df = pd.read_csv(manifest_csv)
    df = df[(df["domain"] == args.domain) & (df["split"] == args.split)].copy()

    if df.empty:
        raise RuntimeError(f"No rows found for domain={args.domain}, split={args.split}")

    df = df.sort_values(["label", "abs_path"]).reset_index(drop=True)

    class_pairs = (
        df[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
        .values
        .tolist()
    )
    class_names = [name for _, name in class_pairs]

    model_name, pretrained = TEACHER_SPECS[args.teacher]

    print(f"Teacher: {args.teacher}")
    print(f"OpenCLIP: model={model_name}, pretrained={pretrained}")
    print(f"Domain/split: {args.domain}/{args.split}")
    print(f"Samples: {len(df)}")
    print(f"Classes: {len(class_names)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    text_features, logit_scale = build_text_features(
        model=model,
        tokenizer=tokenizer,
        class_names=class_names,
        device=device,
    )

    dataset = ManifestDataset(df, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    outputs = extract(
        model=model,
        loader=loader,
        device=device,
        text_features=text_features,
        logit_scale=logit_scale,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **outputs)

    acc = float((outputs["preds"] == outputs["labels"]).mean())
    print(f"Zero-shot top1: {acc:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()