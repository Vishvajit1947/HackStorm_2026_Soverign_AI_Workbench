#!/usr/bin/env python3
"""Download required models for Kavach real model integration."""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Model definitions: (repo_id, filename, size_gb, purpose)
MODELS = [
    # R0: Test model
    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf", 1.3, "R0: Prove llama.cpp works"),
    
    # R2: Embedder (critical for real semantic search)
    ("BAAI/bge-m3-gguf", "bge-m3-q8_0.gguf", 2.0, "R2: Real embedder for routing & knowledge"),
    
    # R3: Main generation models for MID profile
    ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf", 5.0, "R3: General model (tier 2)"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", 4.6, "R3: Code generation"),
    ("Qwen/Qwen2-VL-7B-Instruct-GGUF", "qwen2-vl-7b-instruct-q4_k_m.gguf", 6.1, "R3: Vision model for scans"),
]

def download_model(repo_id: str, filename: str, size_gb: float, purpose: str) -> bool:
    """Download a single model file."""
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING: {filename}")
    print(f"   Repository: {repo_id}")
    print(f"   Size: ~{size_gb} GB")
    print(f"   Purpose: {purpose}")
    print(f"{'='*70}\n")
    
    try:
        local_dir = Path("models")
        local_dir.mkdir(exist_ok=True)
        
        # Check if already exists
        target_path = local_dir / filename
        if target_path.exists():
            print(f"✓ Already downloaded: {filename}")
            return True
        
        # Download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"\n✓ Successfully downloaded: {filename}")
        print(f"   Saved to: {downloaded_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}", file=sys.stderr)
        return False

def main():
    """Download all required models."""
    print("\n" + "="*70)
    print("🚀 KAVACH MODEL DOWNLOAD SCRIPT")
    print("   Downloading open-weight models for real integration")
    print("="*70)
    
    total_size = sum(m[2] for m in MODELS)
    print(f"\nTotal download size: ~{total_size} GB")
    print(f"Number of models: {len(MODELS)}")
    print(f"\nEstimated time: 1-3 hours (depending on connection speed)")
    
    print("\n⏳ Starting downloads in 3 seconds...")
    import time
    time.sleep(3)
    
    success_count = 0
    failed = []
    
    for i, (repo_id, filename, size_gb, purpose) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Starting download...")
        if download_model(repo_id, filename, size_gb, purpose):
            success_count += 1
        else:
            failed.append(filename)
    
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✓ Successfully downloaded: {success_count}/{len(MODELS)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
    else:
        print("🎉 All models downloaded successfully!")
    print("="*70)
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
