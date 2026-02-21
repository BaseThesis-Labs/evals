#!/usr/bin/env python3
"""Download DNSMOS ONNX models from Microsoft DNS Challenge repository."""

import os
import urllib.request
from pathlib import Path

def download_dnsmos_models():
    """Download DNSMOS ONNX models."""
    cache_dir = Path.home() / '.cache' / 'dnsmos'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # DNSMOS models from Microsoft DNS Challenge
    # These are the official model URLs
    models = {
        'sig.onnx': 'https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/dnsmos_onns/sig_bak_ovr.onnx',
        'bak.onnx': 'https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/dnsmos_onns/sig_bak_ovr.onnx',
        'ovr.onnx': 'https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/dnsmos_onns/sig_bak_ovr.onnx'
    }
    
    # Actually, DNSMOS uses a single model file with multiple outputs
    # Let me check the actual repository structure
    base_url = 'https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS'
    
    # Try alternative URLs - DNSMOS v2 uses separate models
    urls = {
        'sig.onnx': f'{base_url}/dnsmos_onns/sig.onnx',
        'bak.onnx': f'{base_url}/dnsmos_onns/bak.onnx', 
        'ovr.onnx': f'{base_url}/dnsmos_onns/ovr.onnx'
    }
    
    print(f"Downloading DNSMOS models to {cache_dir}...")
    
    for filename, url in urls.items():
        filepath = cache_dir / filename
        if filepath.exists():
            print(f"  {filename} already exists, skipping...")
            continue
            
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            
            # Verify it's a valid file (not HTML)
            with open(filepath, 'rb') as f:
                header = f.read(4)
                if header == b'PK\x03\x04':  # ZIP/ONNX header
                    print(f"    ✓ {filename} downloaded successfully")
                else:
                    print(f"    ✗ {filename} appears to be corrupted (got HTML?)")
                    filepath.unlink()
        except Exception as e:
            print(f"    ✗ Failed to download {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
    
    # Check if we have all models
    all_exist = all((cache_dir / f).exists() for f in ['sig.onnx', 'bak.onnx', 'ovr.onnx'])
    
    if all_exist:
        print("\n✓ All DNSMOS models downloaded successfully!")
    else:
        print("\n⚠ Some models failed to download.")
        print("You may need to download them manually from:")
        print("https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS")

if __name__ == '__main__':
    download_dnsmos_models()
