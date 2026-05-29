#!/usr/bin/env python3
"""Download the SAM 2.1 tiny model checkpoint.

Usage:
    python3 models/download_model.py
    python3 models/download_model.py --model sam2.1_hiera_small  # for larger model

The checkpoint will be saved to the models/ directory.
"""

import argparse
import os
import sys
from pathlib import Path

import requests

# SAM 2 model checkpoint URLs (from Meta's official releases)
MODEL_URLS = {
    "sam2.1_hiera_tiny": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
        "sam2.1_hiera_tiny.pt"
    ),
    "sam2.1_hiera_small": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
        "sam2.1_hiera_small.pt"
    ),
    "sam2.1_hiera_base_plus": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
        "sam2.1_hiera_base_plus.pt"
    ),
    "sam2.1_hiera_large": (
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
        "sam2.1_hiera_large.pt"
    ),
}


MODEL_SIZES = {
    "sam2.1_hiera_tiny": "~80 MB",
    "sam2.1_hiera_small": "~150 MB",
    "sam2.1_hiera_base_plus": "~300 MB",
    "sam2.1_hiera_large": "~800 MB",
}


def download_model(model_name: str, output_dir: Path) -> Path:
    """Download a SAM 2 checkpoint. Returns the path to the downloaded file."""
    if model_name not in MODEL_URLS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(MODEL_URLS.keys())}")
        sys.exit(1)

    url = MODEL_URLS[model_name]
    filename = f"{model_name}.pt"
    output_path = output_dir / filename

    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {output_path} ({file_size:.1f} MB)")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name} ({MODEL_SIZES[model_name]})...")
    print(f"URL: {url}")
    print(f"Destination: {output_path}")
    print()

    # Stream download with progress (browser-like User-Agent to avoid 403)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Progress: {pct:.1f}% ({mb:.1f} MB)", end="")
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  Downloaded: {mb:.1f} MB", end="")

    print()
    final_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Download complete: {output_path} ({final_mb:.1f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download SAM 2 model checkpoint for Sticker Creator"
    )
    parser.add_argument(
        "--model",
        default="sam2.1_hiera_tiny",
        choices=list(MODEL_URLS.keys()),
        help="Model variant to download (default: sam2.1_hiera_tiny)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: models/ next to this script)",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent

    download_model(args.model, output_dir)
    print()
    print("You can now run the Sticker Creator:")
    print("  python3 main.py")


if __name__ == "__main__":
    main()
