#!/usr/bin/env python3
"""Build the frozen 405-pair clinical-concordance set (401 automatic + 4 manual).

This script does not query ClinicalTrials.gov and does not select new cases. It
combines the frozen automatic Phase II/III matches with the manually reviewed
positive records and records exact input hashes for provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_value(row: pd.Series, names: Iterable[str], default: Any = "") -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
            return row[name]
    return default


def parse_pool_rows(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value)
    tagged = [int(x) for x in re.findall(r"pool\s*:\s*(\d+)", text, flags=re.I)]
    if tagged:
        return sorted(set(tagged))
    return sorted(set(int(x) for x in re.findall(r"\d+", text)))


def pool_rows_from(row: pd.Series) -> list[int]:
    for name in (
        "pool_rows",
        "supporting_pool_rows",
        "gene_intermediaries",
        "pool_index_zero_based",
    ):
        if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
            rows = parse_pool_rows(row[name])
            if rows:
                return rows
    return []


def canonical_row(row: pd.Series, source: str) -> dict[str, Any]:
    pool_rows = pool_rows_from(row)
    return {
        "validation_pair_id": str(first_value(row, ["validation_pair_id"])),
        "evidence_source": source,
        "condition_group": first_value(row, ["condition_group", "disorder_type"]),
        "disease_local_global_id": first_value(
            row, ["disease_local_global_id", "disease_local_id"]
        ),
        "disease_label": first_value(row, ["disease_label", "disease"]),
        "disease_mesh": first_value(row, ["disease_mesh"]),
        "drug_local_id": first_value(row, ["drug_local_id"]),
        "drug_name": first_value(row, ["drug_name", "drug"]),
        "drugbank_id": first_value(row, ["drugbank_id"]),
        "gene_symbols": first_value(row, ["gene_symbols", "gene_symbol"]),
        "gene_entrez_ids": first_value(row, ["gene_entrez_ids", "gene_entrez_id"]),
        "pool_rows": ";".join(str(x) for x in pool_rows),
        "evidence_classification": first_value(
            row, ["classification"], "phase2_3_registry_match"
        ),
        "condition_match_kind": first_value(
            row, ["condition_match_kind", "disease_mapping"]
        ),
        "supporting_record_ids": first_value(
            row, ["phase2_3_nct_ids", "supporting_nct_ids"]
        ),
        "supporting_phases": first_value(
            row, ["phase2_3_phases", "supporting_phases"]
        ),
        "supporting_statuses": first_value(row, ["supporting_statuses"]),
        "source_urls": first_value(row, ["source_urls"]),
        "manual_reason": first_value(row, ["manual_reason"]),
        "automatic_trial_count": first_value(row, ["n_phase2_3_trials"]),
        "hanami_prefilter_median_top_percent": first_value(
            row, ["hanami_median_top_percent", "best_median_top_percent"]
        ),
    }


def enrich_from_universe(frame: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if "validation_pair_id" not in frame or "validation_pair_id" not in universe:
        raise ValueError("Both frozen inputs must contain validation_pair_id")
    if universe["validation_pair_id"].duplicated().any():
        raise ValueError("Frozen pair universe has duplicate validation_pair_id values")
    useful = [
        col
        for col in (
            "validation_pair_id",
            "condition_group",
            "disease_local_global_id",
            "disease_label",
            "disease_mesh",
            "drug_local_id",
            "drug_name",
            "drugbank_id",
            "gene_symbols",
            "gene_entrez_ids",
            "pool_rows",
            "gene_intermediaries",
            "best_median_top_percent",
        )
        if col in universe.columns
    ]
    merged = frame.merge(
        universe[useful], on="validation_pair_id", how="left", suffixes=("", "__universe")
    )
    for col in useful:
        other = f"{col}__universe"
        if other not in merged:
            continue
        if col not in merged:
            merged[col] = merged[other]
        else:
            empty = merged[col].isna() | merged[col].astype(str).str.strip().eq("")
            merged.loc[empty, col] = merged.loc[empty, other]
        merged = merged.drop(columns=other)
    return merged


def build(config_path: Path, force: bool = False) -> Path:
    config = load_config(config_path)
    paths = config["paths"]
    auto_path = resolve_path(paths["automatic_pairs"])
    manual_path = resolve_path(paths["manual_reviews"])
    universe_path = resolve_path(paths["frozen_pair_universe"])
    # Keep the publication-ready 405-row table immutable. Rebuild the cohort
    # from its frozen inputs into results/ and verify that its membership
    # matches the packaged table used by the analysis.
    canonical_path = resolve_path(paths["documented_pairs"])
    output_path = resolve_path(paths.get("rebuilt_documented_pairs", paths["documented_pairs"]))
    result_dir = resolve_path(paths["results_dir"])

    for path in (auto_path, manual_path, universe_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input is missing: {path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force")

    automatic = pd.read_csv(auto_path, low_memory=False)
    manual = pd.read_csv(manual_path, low_memory=False)
    universe = pd.read_csv(universe_path, low_memory=False)
    automatic = enrich_from_universe(automatic, universe)
    manual = enrich_from_universe(manual, universe)

    analysis = config.get("analysis", {})
    positive_col = analysis.get("manual_positive_column", "classification")
    positive_value = str(analysis.get("manual_positive_value", "strong")).casefold()
    if positive_col not in manual:
        raise ValueError(f"Manual-review file lacks required column: {positive_col}")
    manual_positive = manual[
        manual[positive_col].astype(str).str.casefold().eq(positive_value)
    ].copy()

    expected = config.get("expected", {})
    expected_auto = int(expected.get("automatic_pairs", len(automatic)))
    expected_manual = int(expected.get("manual_positive_pairs", len(manual_positive)))
    expected_total = int(expected.get("documented_pairs", len(automatic) + len(manual_positive)))
    if len(automatic) != expected_auto:
        raise ValueError(f"Expected {expected_auto} automatic pairs, found {len(automatic)}")
    if len(manual_positive) != expected_manual:
        raise ValueError(f"Expected {expected_manual} manual positives, found {len(manual_positive)}")

    rows = [canonical_row(row, "automatic_phase2_3") for _, row in automatic.iterrows()]
    rows += [canonical_row(row, "manual_review") for _, row in manual_positive.iterrows()]
    combined = pd.DataFrame(rows)
    if combined["validation_pair_id"].eq("").any():
        raise ValueError("At least one documented pair has no validation_pair_id")
    duplicates = combined.loc[
        combined["validation_pair_id"].duplicated(False), "validation_pair_id"
    ].unique()
    if len(duplicates):
        raise ValueError(f"Automatic/manual inputs overlap: {duplicates.tolist()[:10]}")
    if len(combined) != expected_total:
        raise ValueError(f"Expected {expected_total} documented pairs, found {len(combined)}")
    missing_pool = combined["pool_rows"].astype(str).str.strip().eq("")
    if missing_pool.any():
        bad = combined.loc[missing_pool, "validation_pair_id"].tolist()[:10]
        raise ValueError(f"Documented pairs lack candidate pool rows: {bad}")

    pool_size = int(expected.get("pool_size", 0))
    if pool_size:
        invalid = [
            (pair_id, index)
            for pair_id, text in zip(combined.validation_pair_id, combined.pool_rows)
            for index in parse_pool_rows(text)
            if index < 0 or index >= pool_size
        ]
        if invalid:
            raise ValueError(f"Candidate index outside [0,{pool_size}): {invalid[:10]}")

    combined = combined.sort_values(
        ["evidence_source", "condition_group", "disease_label", "drug_name", "validation_pair_id"]
    )
    if canonical_path.is_file() and canonical_path.resolve() != output_path.resolve():
        canonical = pd.read_csv(canonical_path, low_memory=False)
        if "validation_pair_id" not in canonical:
            raise ValueError(f"Canonical table lacks validation_pair_id: {canonical_path}")
        rebuilt_ids = set(combined["validation_pair_id"].astype(str))
        canonical_ids = set(canonical["validation_pair_id"].astype(str))
        if rebuilt_ids != canonical_ids:
            missing = sorted(canonical_ids - rebuilt_ids)[:10]
            extra = sorted(rebuilt_ids - canonical_ids)[:10]
            raise ValueError(
                "Rebuilt 405-set does not match the packaged table; "
                f"missing={missing}, extra={extra}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    manifest = {
        "stage": "build_clinical_concordance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": manifest_path(config_path),
        "inputs": {
            "automatic_pairs": {"path": manifest_path(auto_path), "sha256": sha256(auto_path)},
            "manual_reviews": {"path": manifest_path(manual_path), "sha256": sha256(manual_path)},
            "frozen_pair_universe": {
                "path": manifest_path(universe_path),
                "sha256": sha256(universe_path),
            },
        },
        "selection": {
            "automatic_rows": len(automatic),
            "manual_positive_column": positive_col,
            "manual_positive_value": positive_value,
            "manual_positive_rows": len(manual_positive),
            "combined_rows": len(combined),
        },
        "output": {"path": manifest_path(output_path), "sha256": sha256(output_path)},
        "canonical_membership_check": {
            "path": manifest_path(canonical_path),
            "matched": canonical_path.is_file(),
        },
    }
    (result_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(combined)} documented pairs to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite the combined 405 CSV")
    args = parser.parse_args()
    build(args.config.resolve(), force=args.force)


if __name__ == "__main__":
    main()
