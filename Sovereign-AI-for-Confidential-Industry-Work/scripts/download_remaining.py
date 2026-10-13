#!/usr/bin/env python3
"""Download remaining models with corrected repository paths."""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Corrected model definitions
MODELS = [
    # R2: Embedder - CORRECTED PATH
    ("cstr/bge-m3-GGUF", "bge-m3-q8_0.gguf", 0.6, "R2: Real embedder (cstr repo has full weights)"),
    
    # R3: General model - CORRECTED PATH  
    ("QuantFactory/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct.Q4_K_M.gguf", 4.4, "R3: General model (QuantFactory)"),
    
    # R3: Vision model - CORRECTED TO QWEN2.5-VL
    ("Mungert/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-q4_k_m.gguf", 4.7, "R3: Vision model (Qwen2.5-VL)"),
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
    """Download remaining models."""
    print("\n" + "="*70)
    print("🚀 DOWNLOADING REMAINING MODELS")
    print("   Corrected repository paths")
    print("="*70)
    
    # Check what we already have
    models_dir = Path("models")
    existing = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
    
    if existing:
        print("\n✅ Already downloaded:")
        for f in existing:
            size_gb = f.stat().st_size / (1024**3)
            print(f"   ✓ {f.name} ({size_gb:.2f} GB)")
    
    total_size = sum(m[2] for m in MODELS)
    print(f"\nRemaining to download: ~{total_size} GB")
    print(f"Number of models: {len(MODELS)}")
    
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
        print("🎉 All remaining models downloaded successfully!")
    
    # Show final status
    all_models = list(Path("models").glob("*.gguf")) if Path("models").exists() else []
    if all_models:
        print(f"\n📦 Total models available: {len(all_models)}")
        total_gb = sum(f.stat().st_size for f in all_models) / (1024**3)
        print(f"📊 Total size: {total_gb:.2f} GB")
    
    print("="*70)
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
