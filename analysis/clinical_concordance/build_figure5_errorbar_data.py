"""Reconstruct current Figure 5 case percentiles for all five methods and ten seeds.

The Rmd computes descriptive case SDs and exploratory paired tests. This script
checks the frozen source hashes, seed alignment and all released cohort means.
No training, cohort filtering, seed selection or case reselection occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_mean_rank_percentile_source import (
    DEFAULT_INPUT, DEFAULT_OUTPUT, METHOD_NAMES, SEEDS,
    ensure_output_directory, reconstruct, sha256,
)


def build(input_dir: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    manifest, cohort, cases, percentiles, checks = reconstruct(input_dir)
    rows = [
        {"panel": case.panel, "candidate_index": int(case.candidate_index),
         "method": METHOD_NAMES[key], "seed": seed,
         "rank_percentile": float(values[s, case.candidate_index])}
        for key, values in percentiles.items()
        for s, seed in enumerate(SEEDS)
        for case in cases.itertuples(index=False)
    ]
    result = pd.DataFrame(rows).sort_values(["panel", "method", "seed"])
    if len(result) != 250 or result.duplicated(["panel", "method", "seed"]).any():
        raise ValueError("Expected exactly 250 unique panel/method/seed observations")
    output_dir = ensure_output_directory(output_dir, input_dir)
    output = output_dir / "figure5_case_seed_percentiles.csv"
    result.to_csv(output, index=False, float_format="%.17g")
    audit = {
        "input_manifest_sha256": sha256(input_dir / "manifest.json"),
        "all_source_hashes_verified": True, "seed_and_candidate_alignment_verified": True,
        "seeds": SEEDS, "candidate_pool_size": 46704, "motifs": len(cohort),
        "pair_clusters": 785, "disease_labels": 109,
        "case_candidates": manifest["case_candidates"],
        "aggregation": "arithmetic mean across all ten seeds; no median",
        "case_error_bars": "sample SD across ten seed-specific percentiles, divisor n-1",
        "source_reproduction_checks": checks, "output_sha256": sha256(output),
    }
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"Verified all cohort means and saved 250 case/seed values: {build(args.input_dir, args.output_dir)}")


if __name__ == "__main__":
    main()
