"""Select the published DRKG-minus-MS features from the supplied matrices.

This script subsets existing features; it does not run or retrain encoders.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def require_materialized(path):
    with path.open("rb") as stream:
        if stream.read(43).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"{path} is a Git LFS pointer. Run git lfs pull first.")


def load_tensor(path):
    require_materialized(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_mapping(path):
    require_materialized(path)
    return np.load(path, allow_pickle=True).item()


def build_transfer_features(data_root):
    """Preserve the original DRKG ID order while excluding shared entity IDs."""
    features = {}
    mappings = {}
    for entity in ("dise", "drug", "gene"):
        source_ids = load_mapping(data_root / "drkg" / f"id2{entity}.npy")
        target_ids = set(load_mapping(data_root / "ms" / f"id2{entity}.npy").values())
        rows = [index for index, identifier in source_ids.items()
                if identifier not in target_ids]
        mappings[f"subgraph_{entity}.npy"] = {
            old_index: new_index for new_index, old_index in enumerate(rows)
        }

        for source_suffix, output_suffix in (("feat", "Base"), ("All", "Rev")):
            matrix = load_tensor(data_root / "drkg" / f"{entity}_{source_suffix}.pth")
            selected = matrix[rows, :]
            if entity == "dise" and source_suffix == "feat":
                # Baseline disease features are a square similarity matrix.
                selected = selected[:, rows]
            features[f"DRKG_MS_{entity}_{output_suffix}.pth"] = selected
    return features, mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path,
                        help="Destination or comparison directory (default: DATA_ROOT/drkg).")
    parser.add_argument("--check", action="store_true",
                        help="Compare against existing files without writing anything.")
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_root / "drkg"
    features, mappings = build_transfer_features(args.data_root)

    if args.check:
        for filename, tensor in features.items():
            existing = load_tensor(output_dir / filename)
            if tensor.dtype != existing.dtype or not torch.equal(tensor, existing):
                raise ValueError(f"Feature values differ: {filename}")
            print(f"MATCH {filename}: {tuple(tensor.shape)}")
        for filename, mapping in mappings.items():
            existing = load_mapping(output_dir / filename)
            if mapping != existing:
                raise ValueError(f"Entity mapping differs: {filename}")
            print(f"MATCH {filename}: {len(mapping)} entities")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, tensor in features.items():
        torch.save(tensor, output_dir / filename)
        print(f"WROTE {filename}: {tuple(tensor.shape)}")
    for filename, mapping in mappings.items():
        np.save(output_dir / filename, mapping)
        print(f"WROTE {filename}: {len(mapping)} entities")


if __name__ == "__main__":
    main()
