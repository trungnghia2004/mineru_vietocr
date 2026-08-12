#!/usr/bin/env python3
"""
Script to download all MinerU pipeline models from HuggingFace and PyTorch models.
Run this on a machine with internet access before building the Docker image.

Usage:
    python download_models.py
"""

from huggingface_hub import snapshot_download
import os
import sys
import urllib.request
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_REPO = "opendatalab/PDF-Extract-Kit-1.0"
LOCAL_MODELS_DIR = REPO_ROOT / "models_cache" / "pipeline"
TORCH_MODELS_DIR = REPO_ROOT / "torch_models" / "checkpoints"

# PyTorch model for VietOCR backbone
PYTORCH_VGG19_URL = "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
PYTORCH_VGG19_FILE = "vgg19_bn-c79401a0.pth"

# Model paths needed for pipeline mode (from ModelPath enum)
MODEL_PATHS = [
    "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt",
    "models/MFD/YOLO/yolo_v8_ft.pt",
    "models/MFR/unimernet_hf_small_2503",
    "models/OCR/paddleocr_torch",
    "models/ReadingOrder/layout_reader",
    "models/TabRec/SlanetPlus/slanet-plus.onnx",
    "models/TabRec/UnetStructure/unet.onnx",
    "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx",
    "models/OriCls/paddle_orientation_classification/PP-LCNet_x1_0_doc_ori.onnx",
]

def download_pytorch_model():
    """Download PyTorch VGG19 model for VietOCR"""
    print("\n" + "=" * 70)
    print("Downloading PyTorch VGG19 Backbone Model")
    print("=" * 70)
    
    # Create directory
    os.makedirs(TORCH_MODELS_DIR, exist_ok=True)
    
    output_path = TORCH_MODELS_DIR / PYTORCH_VGG19_FILE
    
    # Check if already exists
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ VGG19 model already exists: {output_path}")
        print(f"  Size: {file_size:.1f} MB")
        return
    
    print(f"\nDownloading from: {PYTORCH_VGG19_URL}")
    print(f"Target: {output_path}")
    print("Please wait... (~549 MB)\n")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
        
        urllib.request.urlretrieve(PYTORCH_VGG19_URL, str(output_path), reporthook=progress_hook)
        print("\n\n✓ PyTorch VGG19 model downloaded successfully!")
        
    except Exception as e:
        print(f"\n✗ Error downloading PyTorch model: {e}")
        print("You can manually download it from:")
        print(f"  {PYTORCH_VGG19_URL}")
        print(f"And save it to: {output_path}")
        raise

def main():
    print("=" * 70)
    print("MinerU + VietOCR Models Downloader")
    print("=" * 70)
    print(f"\nPipeline Repository: {PIPELINE_REPO}")
    print(f"Pipeline Target: {LOCAL_MODELS_DIR}")
    print(f"PyTorch Models Target: {TORCH_MODELS_DIR}")
    print(f"\nNumber of MinerU components: {len(MODEL_PATHS)}")
    print("\nMinerU models to download:")
    for i, path in enumerate(MODEL_PATHS, 1):
        print(f"  {i}. {path}")
    print("\n" + "=" * 70)
    
    # Create patterns for all models
    patterns = []
    for path in MODEL_PATHS:
        path = path.strip('/')
        patterns.append(path)
        patterns.append(path + "/*")
    
    print("\n[1/2] Downloading MinerU pipeline models...")
    print("This may take a while depending on your connection.")
    print("(Pipeline models are approximately 2.5 GB total)\n")
    
    try:
        # Download all pipeline models at once
        cache_dir = snapshot_download(
            repo_id=PIPELINE_REPO,
            local_dir=str(LOCAL_MODELS_DIR),
            local_dir_use_symlinks=False,
            allow_patterns=patterns,
            resume_download=True,
        )
        
        print("\n" + "=" * 70)
        print("✓ MinerU pipeline models downloaded successfully!")
        print("=" * 70)
        print(f"\nModels location: {os.path.abspath(cache_dir)}")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Error downloading MinerU models:")
        print("=" * 70)
        print(f"{str(e)}")
        print("\nPlease check your internet connection and try again.")
        sys.exit(1)
    
    # Download PyTorch model
    print("\n[2/2] Downloading PyTorch VGG19 model for VietOCR...")
    try:
        download_pytorch_model()
    except Exception as e:
        print("\nWarning: PyTorch model download failed")
        print("The Docker build will fail without this model.")
        sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓ ALL MODELS DOWNLOADED SUCCESSFULLY!")
    print("=" * 70)
    print("\nDownloaded:")
    print(f"  • MinerU pipeline models: {os.path.abspath(LOCAL_MODELS_DIR)}")
    print(f"  • PyTorch VGG19 model: {os.path.abspath(TORCH_MODELS_DIR)}")
    print(f"\nYou can now build the Docker image with:")
    print("  ./build_offline_image.sh")
    print("\n")

if __name__ == "__main__":
    main()

