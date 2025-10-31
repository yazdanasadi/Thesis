#!/usr/bin/env python3
"""
Simple script to verify dataset directories are accessible (no dependencies required).

Usage:
    python scripts/check_data_paths.py
"""

from pathlib import Path

def check_datasets():
    """Check if all dataset directories exist."""
    print("=" * 70)
    print("Dataset Directory Check")
    print("=" * 70)
    print(f"\nWorking directory: {Path.cwd()}")

    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    print(f"Project root:      {project_root.resolve()}")

    data_dir = project_root / 'data'
    print(f"Data directory:    {data_dir.resolve()}")
    print()

    if not data_dir.exists():
        print("[ERROR] 'data' directory not found!")
        print(f"  Expected at: {data_dir.resolve()}")
        return False

    datasets = ['physionet', 'mimic', 'ushcn', 'activity']

    all_found = True
    for dataset_name in datasets:
        dataset_path = data_dir / dataset_name
        processed_path = dataset_path / 'processed'
        raw_path = dataset_path / 'raw'

        exists = dataset_path.exists()
        has_processed = processed_path.exists()
        has_raw = raw_path.exists()

        status = "[OK]" if exists else "[MISSING]"
        print(f"{status:10} {dataset_name:12} -> {dataset_path}")

        if exists:
            if has_processed:
                num_files = len(list(processed_path.iterdir()))
                print(f"  {'':12}    Processed: {num_files} files")
            if has_raw:
                num_files = len(list(raw_path.iterdir()))
                print(f"  {'':12}    Raw: {num_files} files")
        else:
            print(f"  {'':12}    NOT FOUND")
            all_found = False

        print()

    print("=" * 70)

    if all_found:
        print("[SUCCESS] All datasets found!")
        print("\nDataset paths have been fixed in lib/parse_datasets.py")
        print("The code now automatically detects the correct data path")
        print("regardless of where you run the training scripts from.")
    else:
        print("[WARNING] Some datasets are missing!")
        print("\nPlease ensure your data directories are properly extracted.")

    print("=" * 70)

    return all_found

if __name__ == "__main__":
    check_datasets()
