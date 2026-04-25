#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "This script requires open_clip_torch.\n"
        "Install with: pip install open_clip_torch"
    ) from e


# ============================================================
# Prompt templates (kept for compatibility / optional zero-shot use)
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

    We mainly use `features` for frozen-feature probing.
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


def save_probe_outputs_npz(
    out_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: np.ndarray,
    proba: np.ndarray | None,
    decision_function: np.ndarray | None,
    meta: Dict[str, Any],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
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


def save_json(out_path: Path, obj: Dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"saved: {out_path}")


# ============================================================
# Helpers
# ============================================================

def build_dataset(domain_dir: Path, transform) -> ImageFolderWithPaths:
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain path does not exist: {domain_dir}")
    if not domain_dir.is_dir():
        raise NotADirectoryError(f"Domain path is not a directory: {domain_dir}")
    return ImageFolderWithPaths(str(domain_dir), transform=transform)


def get_domain_names(dataset_root: Path, domain_names_arg: List[str] | None) -> List[str]:
    if domain_names_arg is not None and len(domain_names_arg) > 0:
        return list(domain_names_arg)

    domain_names = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
    if not domain_names:
        raise RuntimeError(f"No domain directories found under: {dataset_root}")
    return domain_names


def compute_class_names(
    dataset_root: Path,
    domain_names: List[str],
    train_domain: str,
    class_mode: str,
) -> List[str]:
    domain_to_classes = {}

    for d in domain_names:
        ds = ImageFolder(str(dataset_root / d))
        domain_to_classes[d] = list(ds.classes)

    if class_mode == "strict":
        ref = domain_to_classes[domain_names[0]]
        for d in domain_names[1:]:
            if domain_to_classes[d] != ref:
                raise ValueError(
                    "Class lists differ across domains under strict mode.\n"
                    f"{domain_names[0]}: {ref}\n"
                    f"{d}: {domain_to_classes[d]}"
                )
        return ref

    if class_mode == "intersection":
        common = set(domain_to_classes[domain_names[0]])
        for d in domain_names[1:]:
            common &= set(domain_to_classes[d])
        common = sorted(common)
        if not common:
            raise RuntimeError("No common classes across domains.")
        return common

    if class_mode == "from_train":
        return domain_to_classes[train_domain]

    raise ValueError(f"Unknown class_mode: {class_mode}")


def filter_dataset_to_classes(dataset: ImageFolderWithPaths, class_names: List[str]) -> None:
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-feature linear probe on ImageFolder-style multi-domain datasets.\n\n"
            "Expected layout:\n"
            "  dataset_root/domain_name/class_name/image.xxx\n\n"
            "Protocol:\n"
            "  1) choose one train domain\n"
            "  2) extract frozen teacher embeddings for all domains\n"
            "  3) split train domain into 80/20 train/val\n"
            "  4) fit linear probe on train split only\n"
            "  5) evaluate on held-out val split + all other domains"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument(
        "--train_domain",
        type=str,
        required=True,
        help="Domain used to train the linear probe (80:20 split within this domain).",
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
        choices=["strict", "intersection", "from_train"],
        default="strict",
        help=(
            "How to define class vocabulary across domains:\n"
            "  strict       -> all domain class lists must match exactly\n"
            "  intersection -> use only classes common to all domains\n"
            "  from_train   -> use only the train domain's class list"
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
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--probe_C", type=float, default=1.0)
    parser.add_argument("--probe_max_iter", type=int, default=2000)
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument(
        "--save_probe_outputs",
        action="store_true",
        help="Save per-domain probe predictions/probabilities to NPZ for ensemble/routing baselines.",
    )
    parser.add_argument("--max_batches", type=int, default=None, help="Debug only")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--check_registry_only", action="store_true")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and args.device.lower().startswith("cuda")
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_root = Path(args.dataset_root)
    outdir = Path(args.outdir)
    domain_names = get_domain_names(dataset_root, args.domain_names)

    if args.train_domain not in domain_names:
        raise ValueError(
            f"--train_domain={args.train_domain!r} not in discovered domain names: {domain_names}"
        )

    if args.check_registry_only:
        print_requested_teacher_availability()
        return

    class_names = compute_class_names(
        dataset_root=dataset_root,
        domain_names=domain_names,
        train_domain=args.train_domain,
        class_mode=args.class_mode,
    )

    print(f"Using domains: {domain_names}")
    print(f"Train domain: {args.train_domain}")
    print(f"Class mode: {args.class_mode}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    summary = {
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root),
        "train_domain": args.train_domain,
        "domain_names": domain_names,
        "class_mode": args.class_mode,
        "class_names": class_names,
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

        model, preprocess, model_tag, teacher_meta = build_teacher(
            teacher_name=teacher_name,
            class_names=class_names,
            device=device,
        )
        model = model.to(device)
        model.eval()

        domain_features: Dict[str, Dict[str, np.ndarray]] = {}

        # ------------------------------------------------------------
        # Extract frozen features for every domain
        # ------------------------------------------------------------
        for domain_name in domain_names:
            print(f"\nExtracting features for domain: {domain_name}")
            ds = build_dataset(dataset_root / domain_name, preprocess)

            if args.class_mode != "strict":
                filter_dataset_to_classes(ds, class_names)

            loader = make_loader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
            arrays = extract_teacher_outputs(
                model=model,
                loader=loader,
                device=device,
                max_batches=args.max_batches,
            )

            domain_features[domain_name] = arrays

            if args.save_features:
                feature_meta = {
                    "dataset_name": args.dataset_name,
                    "domain_name": domain_name,
                    "train_domain": args.train_domain,
                    "class_names": class_names,
                    "class_mode": args.class_mode,
                    "n_samples": int(len(arrays["labels"])),
                    "feat_dim": int(arrays["feats"].shape[1]),
                    "teacher_name": teacher_name,
                    "model_tag": model_tag,
                    "teacher_meta": teacher_meta,
                }
                save_npz(
                    teacher_outdir / f"features_{domain_name}.npz",
                    arrays,
                    feature_meta,
                )

        # ------------------------------------------------------------
        # Train probe on train_domain split only
        # ------------------------------------------------------------
        train_arrays = domain_features[args.train_domain]
        x_all = train_arrays["feats"]
        y_all = train_arrays["labels"]
        p_all = train_arrays["paths"]

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

        results = {
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
            "domains": {},
        }

        # In-domain held-out validation
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
            proba_np = None
            decision_np = None

            if "proba" in val_metrics:
                proba_np = np.asarray(val_metrics["proba"], dtype=np.float32)
            if "decision_function" in val_metrics:
                decision_np = np.asarray(val_metrics["decision_function"], dtype=np.float32)

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

        # Other domains = full OOD test
        for domain_name in domain_names:
            if domain_name == args.train_domain:
                continue

            arr = domain_features[domain_name]
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
                proba_np = None
                decision_np = None

                if "proba" in test_metrics:
                    proba_np = np.asarray(test_metrics["proba"], dtype=np.float32)
                if "decision_function" in test_metrics:
                    decision_np = np.asarray(test_metrics["decision_function"], dtype=np.float32)

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

        del model
        if use_cuda:
            torch.cuda.empty_cache()

    save_json(outdir / args.dataset_name / args.train_domain / "summary.json", summary)
    print("\nDone.")


if __name__ == "__main__":
    main()

# python scripts\linear_probe_terra.py `
#   --dataset_root "C:\Users\racs2019\Documents\NIPS-KD\data\terra_incognita" `
#   --dataset_name "terra_incognita" `
#   --train_domain "location_38" `
#   --domain_names location_38 location_43 location_46 location_100 `
#   --class_mode strict `
#   --outdir "terra_probe_results" `
#   --batch_size 64 `
#   --num_workers 4 `
#   --device cuda `
#   --save_probe_outputs `
#   --save_features