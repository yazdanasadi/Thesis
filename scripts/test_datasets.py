#!/usr/bin/env python3
"""
Test script to verify all datasets are accessible and loadable.

Usage:
    python scripts/test_datasets.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from types import SimpleNamespace

# Import after path is set
from lib.parse_datasets import _get_data_path

def test_data_paths():
    """Test that all dataset paths can be resolved."""
    print("=" * 60)
    print("Testing Dataset Path Resolution")
    print("=" * 60)

    datasets = ['physionet', 'mimic', 'ushcn', 'activity']

    for dataset_name in datasets:
        path = _get_data_path(dataset_name)
        exists = Path(path).exists()
        status = "✓ FOUND" if exists else "✗ NOT FOUND"

        print(f"\n{dataset_name:12} -> {path}")
        print(f"{'':12}    {status}")

        if exists:
            # Check for processed data
            processed_path = Path(path) / 'processed'
            if processed_path.exists():
                num_files = len(list(processed_path.glob('*')))
                print(f"{'':12}    Processed files: {num_files}")

    print("\n" + "=" * 60)

def test_dataset_loading():
    """Test loading a small sample from each dataset."""
    print("\nTesting Dataset Loading (small samples)")
    print("=" * 60)

    device = torch.device('cpu')

    test_configs = [
        ('physionet', {'quantization': 1.0, 'n': 100, 'device': device}),
        ('mimic', {'n': 50, 'device': device}),
        ('ushcn', {'n': 20, 'device': device}),
        ('activity', {'n': 20, 'device': device}),
    ]

    for dataset_name, config in test_configs:
        print(f"\n{dataset_name}:")
        try:
            # Create minimal args object
            args = SimpleNamespace(
                dataset=dataset_name,
                quantization=config.get('quantization', 1.0),
                n=config['n'],
                device=config['device'],
                batch_size=16,
                history=24,
            )

            from lib.parse_datasets import parse_datasets
            data_objects = parse_datasets(args, patch_ts=False, length_stat=False)

            print(f"  ✓ Successfully loaded")
            print(f"  Input dim: {data_objects['input_dim']}")
            print(f"  Train batches: {data_objects['n_train_batches']}")
            print(f"  Val batches: {data_objects['n_val_batches']}")
            print(f"  Test batches: {data_objects['n_test_batches']}")

        except Exception as e:
            print(f"  ✗ Failed to load: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("\nDataset Accessibility Test")
    print("Working directory:", Path.cwd())
    print("Project root:", PROJECT_ROOT)
    print()

    # Test path resolution
    test_data_paths()

    # Optionally test loading (can be slow)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-test', action='store_true',
                       help='Also test loading datasets (slower)')
    args = parser.parse_args()

    if args.load_test:
        test_dataset_loading()
    else:
        print("\nSkipping load test. Use --load-test to verify dataset loading.")

    print("\nDone!")
