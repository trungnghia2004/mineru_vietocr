#!/usr/bin/env python3
"""
Script to download all MinerU pipeline models from HuggingFace.
Run this on a machine with internet access before building the Docker image.

Usage:
    python download_models.py
"""

from huggingface_hub import snapshot_download
import os
import sys

# Configuration
PIPELINE_REPO = "opendatalab/PDF-Extract-Kit-1.0"
LOCAL_MODELS_DIR = "./models_cache/pipeline"

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

def main():
    print("=" * 70)
    print("MinerU Pipeline Models Downloader")
    print("=" * 70)
    print(f"\nRepository: {PIPELINE_REPO}")
    print(f"Target directory: {LOCAL_MODELS_DIR}")
    print(f"Number of model components: {len(MODEL_PATHS)}")
    print("\nModels to download:")
    for i, path in enumerate(MODEL_PATHS, 1):
        print(f"  {i}. {path}")
    print("\n" + "=" * 70)
    
    # Create patterns for all models
    patterns = []
    for path in MODEL_PATHS:
        path = path.strip('/')
        patterns.append(path)
        patterns.append(path + "/*")
    
    print("\nStarting download... This may take a while depending on your connection.")
    print("(Models are approximately 5-10 GB total)\n")
    
    try:
        # Download all models at once
        cache_dir = snapshot_download(
            repo_id=PIPELINE_REPO,
            local_dir=LOCAL_MODELS_DIR,
            local_dir_use_symlinks=False,
            allow_patterns=patterns,
            resume_download=True,
        )
        
        print("\n" + "=" * 70)
        print("✓ Models downloaded successfully!")
        print("=" * 70)
        print(f"\nModels location: {os.path.abspath(cache_dir)}")
        print(f"\nYou can now build the Docker image with:")
        print("  docker build -f Dockerfile.offline -t mineru-vietocr-offline:latest .")
        print("\n")
        
        # Try to show directory structure
        if os.system("which tree > /dev/null 2>&1") == 0:
            print("Downloaded structure:")
            os.system(f"tree {LOCAL_MODELS_DIR} -L 3")
        else:
            print("\nTo see the directory structure, install 'tree' command or use:")
            print(f"  ls -lR {LOCAL_MODELS_DIR}")
            
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Error downloading models:")
        print("=" * 70)
        print(f"{str(e)}")
        print("\nPlease check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

