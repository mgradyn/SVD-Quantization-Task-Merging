#!/usr/bin/env python3
"""
Convenience script to run SVD-Hybrid merging with common configurations.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.svd_hybrid.cli import main

if __name__ == "__main__":
    main()
