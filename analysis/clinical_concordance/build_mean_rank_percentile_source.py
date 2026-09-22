"""Reproduce current Figure 5 motif means from frozen, checksummed scores.

Rank the complete MS pool separately for each method and seed, then take the
arithmetic mean of ten rank percentiles for each motif. This does not train
models or reselect the clinical cohort. Only the output directory is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/clinical_concordance/current"
DEFAULT_OUTPUT = REPO_ROOT / "results/clinical_concordance/figure5"
SEEDS = list(range(0, 100, 10))
METHOD_COLUMNS = {
    "rf": "RF_motif_percentile",
    "trinet": "TriNet_motif_percentile",
    "n2v_mlp": "N2V_MLP_motif_percentile",
    "trimogcl": "TriMoGCL_motif_percentile",
    "hanami": "HANAMI_motif_percentile",
}
METHOD_NAMES = dict(zip(METHOD_COLUMNS, ["RF", "TriNet", "N2V-MLP", "TriMoGCL", "HANAMI"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(base: Path, record: dict) -> Path:
    path = (base / record["path"]).resolve()
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Input checksum mismatch: {path}")
    return path


def rank_percentiles(score_matrix: np.ndarray) -> np.ndarray:
    """Descending, average-tie rank / pool size, in percent (lower is better)."""
    if score_matrix.ndim != 2 or not np.isfinite(score_matrix).all():
        raise ValueError("Scores must be a finite seed-by-candidate matrix")
    pool_size = score_matrix.shape[1]
    if not pool_size:
        raise ValueError("Candidate pool is empty")
    return np.vstack([
        100.0 * pd.Series(-scores).rank(method="average").to_numpy() / pool_size
        for scores in score_matrix
    ])


def load_sources(input_dir: Path = DEFAULT_INPUT) -> tuple:
    input_dir = input_dir.resolve()
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["seeds"] != SEEDS or manifest["candidate_pool_size"] != 46704:
        raise ValueError("The released Figure 5 requires the fixed ten seeds and full 46,704 pool")
    paths = {key: checked_path(input_dir, record) for key, record in manifest["files"].items()}
    with np.load(paths["baseline_scores"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["seeds"], SEEDS)
        np.testing.assert_array_equal(archive["candidate_index"], np.arange(46704))
        # Never use the obsolete HANAMI array in this historical archive.
        arrays = {key: archive[key].copy() for key in METHOD_COLUMNS if key != "hanami"}
    if [entry["seed"] for entry in manifest["hanami_scores"]] != SEEDS:
        raise ValueError("HANAMI files must be aligned to the declared seed order")
    arrays["hanami"] = np.vstack([
        np.load(checked_path(input_dir, entry), allow_pickle=False)
        for entry in manifest["hanami_scores"]
    ])
    for method, array in arrays.items():
        if array.shape != (10, 46704) or not np.isfinite(array).all():
            raise ValueError(f"Invalid score matrix for {method}: {array.shape}")

    cohort = pd.read_csv(paths["cohort_metadata"], sep="\t")
    released = pd.read_csv(paths["released_means"], sep="\t")
    candidates = pd.read_csv(paths["candidate_metadata"])
    cases = pd.read_csv(paths["cases"])
    indices = cohort["candidate_index"].to_numpy(dtype=np.int64)
    if len(cohort) != 1630 or len(np.unique(indices)) != 1630:
        raise ValueError("Expected exactly 1,630 uniquely identified motifs")
    if np.any(indices < 0) or np.any(indices >= 46704):
        raise ValueError("Cohort candidate index outside full score pool")
    for pair_columns in (["disease_local_id", "drug_local_id"], ["disease_mesh_id", "drugbank_id"]):
        if len(cohort[pair_columns].drop_duplicates()) != 785:
            raise ValueError("Expected 785 drug-disease pair clusters")
    if cohort["disease_mesh_id"].nunique() != 109:
        raise ValueError("Expected 109 disease labels")
    if cohort["supporting_trials"].isna().any() or cohort["evidence_source"].isna().any():
        raise ValueError("Clinical evidence metadata is missing")
    # These checks prevent the clinical metadata from being aligned by row order
    # to the wrong candidate scores, or a non-gene-star entering the cohort.
    if len(candidates) != 46704 or candidates["candidate_index"].duplicated().any():
        raise ValueError("Candidate pool metadata is incomplete or duplicated")
    pool_rows = candidates.set_index("candidate_index").loc[indices].reset_index()
    for column in ["candidate_index", "disease_local_id", "drug_local_id", "gene_local_id"]:
        np.testing.assert_array_equal(cohort[column], pool_rows[column])
    for column, expected in [("drug_gene_edge_in_ms", 1), ("gene_disease_edge_in_ms", 1), ("drug_disease_edge_in_ms", 0)]:
        if not (pool_rows[column] == expected).all():
            raise ValueError(f"Cohort violates gene-star edge definition: {column}")
    pd.testing.assert_frame_equal(cohort, released[cohort.columns], check_dtype=False)
    if list(cases["panel"]) != list("cdefg") or cases["candidate_index"].tolist() != manifest["case_candidates"]:
        raise ValueError("Case panel mapping differs from the frozen release")
    if not cases["candidate_index"].isin(indices).all():
        raise ValueError("Every case must belong to the full clinical cohort")
    return manifest, arrays, cohort, released, cases


def reconstruct(input_dir: Path = DEFAULT_INPUT) -> tuple:
    manifest, arrays, cohort, released, cases = load_sources(input_dir)
    indices = cohort["candidate_index"].to_numpy(dtype=np.int64)
    checks, seed_percentiles = [], {}
    for key, column in METHOD_COLUMNS.items():
        percentiles = rank_percentiles(arrays[key])
        seed_percentiles[key] = percentiles
        cohort[column] = percentiles.mean(axis=0)[indices]
        error = float(np.max(np.abs(cohort[column] - released[column])))
        if error >= 1e-8:
            raise ValueError(f"{key} does not reproduce released means: {error} percentage points")
        checks.append({"method": METHOD_NAMES[key], "maximum_reproduction_error_pp": error})
    # Preserve the original published column order as well as the motif order.
    return manifest, cohort[released.columns], cases, seed_percentiles, checks


def ensure_output_directory(output_dir: Path, input_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    input_root = (input_dir.resolve().parent).resolve()
    if output_dir == input_root or input_root in output_dir.parents:
        raise ValueError("Outputs must not be written into the frozen input directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build(input_dir: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    manifest, cohort, _, _, checks = reconstruct(input_dir)
    output_dir = ensure_output_directory(output_dir, input_dir)
    output = output_dir / "validated_gene_star_motifs_1630_mean10.tsv"
    # Match the frozen release's CRLF bytes on Linux as well as Windows.
    cohort.to_csv(output, sep="\t", index=False, float_format="%.10f", lineterminator="\r\n")
    audit = {
        "input_manifest_sha256": sha256(input_dir / "manifest.json"),
        "all_source_hashes_verified": True,
        "seed_and_candidate_alignment_verified": True,
        "seeds": manifest["seeds"], "candidate_pool_size": 46704,
        "cohort_motifs": 1630, "cohort_pairs": 785, "disease_labels": 109,
        "aggregation": manifest["aggregation"], "median_used": False,
        "source_reproduction_checks": checks, "output_sha256": sha256(output),
    }
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"Verified frozen sources and reproduced 1,630 motif means: {build(args.input_dir, args.output_dir)}")


if __name__ == "__main__":
    main()
