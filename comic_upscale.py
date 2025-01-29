import os
import torch
from PIL import Image
import argparse
import platform
import zipfile
import rarfile
import shutil
import logging
from pathlib import Path
from typing import Union

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

def unpack_archive(file_path: Path, unpack_dir: Path):
    """
    Unpacks the cbz/cbr archive into a folder.
    """
    logger.info(f"Unpacking {file_path}...")
    if file_path.suffix == ".cbz":
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(unpack_dir)
    elif file_path.suffix == ".cbr":
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(unpack_dir)
    else:
        raise ValueError("Unsupported archive type. Only .cbz and .cbr are supported.")

def repack_archive(unpack_dir: Path, output_file: Path):
    """
    Repackages the unpacked files back into a cbz file.
    """
    logger.info(f"Repacking {unpack_dir} into {output_file}...")
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for root, _, files in os.walk(unpack_dir):
            for file in files:
                zip_ref.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), unpack_dir))

def upscale_comic_file(file_path: Path, model: torch.nn.Module):
    """
    Upscales a single comic archive (cbz or cbr).
    """
    # Step 1: Unpack the archive
    unpack_dir = file_path.parent / file_path.stem  # Create unpacked directory in the same directory as the original file
    if unpack_dir.exists():
        logger.warning(f"{unpack_dir} already exists, skipping unpacking.")
    else:
        unpack_dir.mkdir()
        unpack_archive(file_path, unpack_dir)

    # Step 2: Find and upscale images
    images = []
    for root, _, files in os.walk(unpack_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
                images.append(Path(root) / file)
    
    for img_path in images:
        logger.info(f"Upscaling image: {img_path}")
        upscale_image(img_path, img_path, model)  # Replaces the original image

    # Step 3: Repack into a new cbz file
    output_file = file_path.with_name(f"{file_path.stem} (upscaled).cbz")
    repack_archive(unpack_dir, output_file)

    # Step 4: Clean up unpacked directory
    shutil.rmtree(unpack_dir)
    logger.info(f"Upscaled archive saved to {output_file}")

def upscale_comics_in_directory(directory_path: Path, model: torch.nn.Module):
    """
    Upscales all comic files in a given directory.
    """
    files = directory_path.glob("*.{cbz,cbr}")
    total_files = len(list(files))  # Get total number of files in the directory
    processed_files = 0

    for file_path in files:
        # Skip already upscaled files
        if "(upscaled)" in file_path.stem:
            logger.info(f"Skipping already upscaled file: {file_path}")
            continue
        
        processed_files += 1
        logger.info(f"Processing {processed_files}/{total_files} - {file_path}")
        upscale_comic_file(file_path, model)

# Function to upscale the image
def upscale_image(in_path: Path, out_path: Path, model: torch.nn.Module):
    input_image = Image.open(in_path)
    result = model.infer(input_image)
    result.save(out_path)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Upscale comic pages using waifu2x")
    parser.add_argument("path", type=Path, help="Path to a comic file (cbz/cbr) or directory containing comic files")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps" if platform.system() == "Darwin" else "cuda", help="Device to use (default: mps on macOS, cuda on other systems)")
    parser.add_argument("--model_type", choices=["art", "art_scan", "photo"], default="art_scan", help="Model type to use (default: art_scan)")
    parser.add_argument("--method", choices=["scale2x", "scale4x", "noise"], default="scale2x", help="Method for upscaling (default: scale2x)")
    parser.add_argument("--noise_level", type=int, default=-1, choices=[-1, 0, 1, 2, 3], help="Noise level for denoising (default: -1, no denoising)")
    parser.add_argument("--tile_size", type=int, default=None, help="Tile size for processing large images (default: None)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for processing images (default: None)")

    args = parser.parse_args()

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Load the model
    model = torch.hub.load("nagadomi/nunif:master", "waifu2x", model_type=args.model_type, 
                           method=args.method, noise_level=args.noise_level, 
                           tile_size=args.tile_size, batch_size=args.batch_size, 
                           trust_repo=True)

    # Set the device (CPU, CUDA, or MPS)
    device = args.device
    model = model.to(device)

    # Check if path is a file or directory
    path = Path(args.path)
    if path.is_file():
        logger.info(f"Processing single file: {path}")
        upscale_comic_file(path, model)
    elif path.is_dir():
        logger.info(f"Processing all comic files in directory: {path}")
        upscale_comics_in_directory(path, model)
    else:
        logger.error(f"Invalid path: {path}. Please provide a valid file or directory.")

if __name__ == "__main__":
    main()
