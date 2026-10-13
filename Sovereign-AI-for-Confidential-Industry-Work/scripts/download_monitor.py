#!/usr/bin/env python3
"""Monitor download progress."""

from pathlib import Path
import time

def format_size(bytes):
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

def main():
    models_dir = Path("models")
    
    print("📊 Monitoring downloads in models/ directory...")
    print("="*60)
    
    while True:
        if not models_dir.exists():
            print("⏳ Waiting for downloads to start...")
            time.sleep(2)
            continue
        
        files = list(models_dir.rglob("*.gguf"))
        if not files:
            cache_files = list(models_dir.rglob("*"))
            print(f"⏳ Download initializing... ({len(cache_files)} temp files)")
        else:
            print(f"\n📥 Current status ({time.strftime('%H:%M:%S')}):")
            for f in sorted(files):
                size = f.stat().st_size
                print(f"  ✓ {f.name}: {format_size(size)}")
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
