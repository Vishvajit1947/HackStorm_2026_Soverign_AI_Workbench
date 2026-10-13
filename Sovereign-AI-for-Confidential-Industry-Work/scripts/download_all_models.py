#!/usr/bin/env python3
"""
Download all required models for Kavach real integration.
Fixed repository paths and filenames.
"""

import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

# Corrected model definitions with working repositories
MODELS = [
    # R0: Test model (1.3 GB)
    {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.3,
        "purpose": "R0: Prove llama.cpp works",
        "checkpoint": "10%",
        "config_maps_to": "qwen3-1_7b-q4"
    },
    
    # R2: Embedder (0.6 GB) - CRITICAL
    {
        "repo_id": "cstr/bge-m3-GGUF",
        "filename": "bge-m3-q8_0.gguf",
        "size_gb": 0.6,
        "purpose": "R2: Real embedder for routing & knowledge",
        "checkpoint": "50%",
        "config_maps_to": "bge-m3-emb",
        "critical": True
    },
    
    # R3: General model (4.4 GB)
    {
        "repo_id": "QuantFactory/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct.Q4_K_M.gguf",
        "size_gb": 4.4,
        "purpose": "R3: General Q&A, summarization",
        "checkpoint": "70%",
        "config_maps_to": "qwen3-8b-q4"
    },
    
    # R3: Code generation (4.6 GB)
    {
        "repo_id": "QuantFactory/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-7B-Instruct.Q4_K_M.gguf",
        "size_gb": 4.6,
        "purpose": "R3: Code generation specialist",
        "checkpoint": "70%",
        "config_maps_to": "qwen2_5-coder-7b-q4"
    },
    
    # R3: Vision model (4.7 GB)
    {
        "repo_id": "Mungert/Qwen2.5-VL-7B-Instruct-GGUF",
        "filename": "Qwen2.5-VL-7B-Instruct-q4_k_m.gguf",
        "size_gb": 4.7,
        "purpose": "R3: Vision model for scans/P&IDs",
        "checkpoint": "100%",
        "config_maps_to": "qwen3-vl-8b-q4"
    }
]

def download_model(model: dict) -> bool:
    """Download a single model file."""
    repo_id = model["repo_id"]
    filename = model["filename"]
    size_gb = model["size_gb"]
    purpose = model["purpose"]
    
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING: {filename}")
    print(f"   Repository: {repo_id}")
    print(f"   Size: ~{size_gb} GB")
    print(f"   Purpose: {purpose}")
    print(f"   Checkpoint: {model['checkpoint']}")
    if model.get("critical"):
        print(f"   ⚠️  CRITICAL: Required for real semantic search")
    print(f"{'='*70}\n")
    
    try:
        local_dir = Path("models")
        local_dir.mkdir(exist_ok=True)
        
        # Check if already exists
        target_path = local_dir / filename
        if target_path.exists():
            size_mb = target_path.stat().st_size / (1024**2)
            print(f"✓ Already downloaded: {filename} ({size_mb:.0f} MB)")
            return True
        
        # Download with progress
        print(f"⏳ Starting download...")
        start_time = time.time()
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        
        elapsed = time.time() - start_time
        downloaded_size = Path(downloaded_path).stat().st_size / (1024**3)
        
        print(f"\n✓ Successfully downloaded: {filename}")
        print(f"   Saved to: {downloaded_path}")
        print(f"   Size: {downloaded_size:.2f} GB")
        print(f"   Time: {elapsed:.1f}s ({downloaded_size*8/elapsed:.1f} Mbps)")
        print(f"   Maps to config: {model['config_maps_to']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}", file=sys.stderr)
        print(f"   Repository: {repo_id}")
        print(f"   Try manual download from:")
        print(f"   https://huggingface.co/{repo_id}/resolve/main/{filename}")
        return False

def main():
    """Download all required models."""
    print("\n" + "="*70)
    print("🚀 KAVACH MODEL DOWNLOAD - COMPLETE SETUP")
    print("   Downloading 5 open-weight models for real integration")
    print("="*70)
    
    # Check existing downloads
    models_dir = Path("models")
    existing = list(models_dir.glob("*.gguf")) if models_dir.exists() else []
    
    if existing:
        print("\n✅ Already downloaded:")
        total_existing = 0
        for f in existing:
            size_gb = f.stat().st_size / (1024**3)
            total_existing += size_gb
            print(f"   ✓ {f.name} ({size_gb:.2f} GB)")
        print(f"   Total: {total_existing:.2f} GB")
    
    # Calculate remaining
    total_size = sum(m["size_gb"] for m in MODELS)
    existing_size = sum(f.stat().st_size / (1024**3) for f in existing) if existing else 0
    remaining_size = total_size - existing_size
    
    print(f"\n📊 Download Plan:")
    print(f"   Total models: {len(MODELS)}")
    print(f"   Total size: ~{total_size} GB")
    print(f"   Already have: ~{existing_size:.1f} GB")
    print(f"   Remaining: ~{remaining_size:.1f} GB")
    
    if remaining_size <= 0:
        print("\n🎉 All models already downloaded!")
        print("\nNext steps:")
        print("  1. Stage models: python scripts/stage_model.py --write")
        print("  2. Set backend: KAVACH_BACKEND=llamacpp in .env")
        print("  3. Download llama-server.exe (see INSTALL.md)")
        return 0
    
    print(f"\n⏰ Estimated time: {remaining_size/5:.0f}-{remaining_size*3:.0f} minutes")
    print("   (depends on your internet speed)")
    
    print("\n⏳ Starting downloads in 3 seconds...")
    time.sleep(3)
    
    # Download each model
    success_count = 0
    failed = []
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Starting download...")
        if download_model(model):
            success_count += 1
        else:
            failed.append(model["filename"])
    
    # Final summary
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✓ Successfully downloaded: {success_count}/{len(MODELS)}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
        print("\n⚠️  Some downloads failed. See MODELS.md for manual download links.")
    else:
        print("\n🎉 All models downloaded successfully!")
    
    # Show final status
    all_models = list(Path("models").glob("*.gguf")) if Path("models").exists() else []
    if all_models:
        print(f"\n📦 Total models available: {len(all_models)}")
        total_gb = sum(f.stat().st_size for f in all_models) / (1024**3)
        print(f"📊 Total size: {total_gb:.2f} GB")
        
        print("\n✅ Next Steps:")
        print("   1. Stage models:")
        print("      python scripts/stage_model.py --write")
        print("   ")
        print("   2. Configure backend in .env:")
        print("      KAVACH_BACKEND=llamacpp")
        print("   ")
        print("   3. Download llama-server.exe:")
        print("      See INSTALL.md for instructions")
        print("   ")
        print("   4. Test R0:")
        print("      python scripts/prove_sandbox.py --llama")
        print("   ")
        print("   5. Verify:")
        print("      kavach models list")
    
    print("="*70)
    
    return 0 if not failed else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("   Run again to resume (already downloaded files will be skipped)")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
