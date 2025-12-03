#!/usr/bin/env python
"""Download or unpack Triton model repository assets.

Default behavior:
  - Download real ONNX models from Hugging Face into ./models/<name>/1/model.onnx

Optional:
  - Provide ``--source`` or set ``VISIONTIME_MODELS_URL`` to fetch a tar.gz
    archive over HTTP(S) or from a local path (legacy / custom bundle).
"""

import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Dict
from urllib.request import urlopen

# Base models that actually have ONNX weights to download.
BASE_MODEL_URLS: Dict[str, str] = {
    # CLIP vision encoder (image embeddings)
    "clip_encoder": (
        "https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/main/model.onnx"
    ),
    # Depth estimation (Distill-Any-Depth ONNX)
    "depth_fastdepth": (
        "https://huggingface.co/FuryTMP/Distill-Any-Depth-Base-onnx/resolve/main/model.onnx"
    ),
    # Optical flow (RAFT, Sintel, int8 block-quantized)
    "optical_flow": (
        "https://huggingface.co/opencv/optical_flow_estimation_raft/resolve/main/"
        "optical_flow_estimation_raft_2023aug_int8bq.onnx"
    ),
    # RTMDet-S object detector
    "rtmdet_s": (
        "https://huggingface.co/ziq/rtm/resolve/main/rtmdet-s.onnx"
    ),
    # SegFormer-B0 semantic segmentation
    "segformer": (
        # NVIDIA's repository does not host an ONNX export; use Xenova's
        # pre-converted weights instead (keeps the same architecture/weights
        # but stored under the onnx/ subfolder).
        "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model.onnx"
    ),
}

# Ensemble is a Triton ensemble model (no ONNX file, just a config.pbtxt you keep in the repo).
ENSEMBLE_MODELS = ["ensemble_multi"]

MODEL_NAMES = list(BASE_MODEL_URLS.keys()) + ENSEMBLE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate the Triton model repository")
    parser.add_argument(
        "--source",
        help="HTTP(S) URL or local path to a tar.gz archive containing the models "
             "(if omitted, individual ONNX models are downloaded from the internet)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite any existing model versions before extracting/downloading",
    )
    return parser.parse_args()


def safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = dest / member.name
        if not str(member_path.resolve()).startswith(str(dest)):
            raise RuntimeError(f"unsafe path detected in archive: {member.name}")
    tar.extractall(dest)


def download_to(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> downloading {url} -> {target}")
    with urlopen(url) as response, open(target, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def have_all_models(root: Path, names: Iterable[str]) -> bool:
    """Return True if all *base* models are present under root."""
    for name in BASE_MODEL_URLS.keys():
        model_path = root / name / "1" / "model.onnx"
        if not model_path.exists():
            return False
    # Ensembles are configs only; we don't require model.onnx for them.
    return True


def download_individual_models(models_root: Path, force: bool) -> None:
    print("No tarball source provided; downloading individual ONNX models...")
    for name, url in BASE_MODEL_URLS.items():
        version_dir = models_root / name / "1"
        model_path = version_dir / "model.onnx"

        if model_path.exists() and not force:
            print(f"{name}: model already exists at {model_path}, skipping (use --force to overwrite).")
            continue

        if version_dir.exists() and force:
            print(f"{name}: removing existing versioned directory {version_dir}")
            shutil.rmtree(version_dir)

        download_to(url, model_path)

    # Ensure ensemble directories exist (config.pbtxt should be in the repo).
    for ensemble_name in ENSEMBLE_MODELS:
        ensemble_dir = models_root / ensemble_name / "1"
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        # We don't create model.onnx here; Triton expects just config.pbtxt.


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    models_root = repo_root / "models"
    bundled_archive = repo_root / "placeholder" / "models-placeholder.tar.gz"

    source = args.source or os.environ.get("VISIONTIME_MODELS_URL")

    models_root.mkdir(exist_ok=True)

    # If a tarball source is specified, keep the legacy behavior.
    if source:
        if source.startswith("http://") or source.startswith("https://"):
            tmpdir = Path(tempfile.mkdtemp())
            archive_path = tmpdir / "models.tar.gz"
            print(f"Downloading model archive from {source} ...")
            download_to(source, archive_path)
        else:
            archive_path = Path(source)
            if not archive_path.exists():
                raise FileNotFoundError(f"{archive_path} does not exist")

        if args.force:
            for name in MODEL_NAMES:
                version_dir = models_root / name / "1"
                if version_dir.exists():
                    print(f"Removing existing versioned directory for {name}")
                    shutil.rmtree(version_dir)

        with tarfile.open(archive_path, "r:gz") as tar:
            print(f"Extracting models from {archive_path} ...")
            safe_extract(tar, models_root)

        print("Models ready under ./models (archive source).")
        return

    # No tarball source: use the real ONNX model URLs.
    if not args.force and have_all_models(models_root, MODEL_NAMES):
        print("All models already exist; nothing to do. Use --force to refresh.")
        return

    download_individual_models(models_root, args.force)
    print("Models ready under ./models (downloaded from Hugging Face).")


if __name__ == "__main__":
    main()
