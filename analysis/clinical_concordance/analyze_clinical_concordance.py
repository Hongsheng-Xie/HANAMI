#!/usr/bin/env python3
"""Rank the documented pairs, compare methods, and write Fig. 5 source data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, wilcoxon


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


def parse_indices(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    tagged = re.findall(r"pool\s*:\s*(\d+)", str(value), flags=re.I)
    values = tagged or re.findall(r"\d+", str(value))
    return sorted(set(int(x) for x in values))


def choose_column(frame: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"None of the required columns is present: {names}")
    return None


def load_score_archive(
    path: Path,
    score_keys: dict[str, str],
    expected_seeds: list[int],
    expected_pool_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Score archive is missing: {path}. It must contain seeds, candidate_index, "
            f"and one (10, {expected_pool_size}) array per method."
        )
    with np.load(path, allow_pickle=False) as archive:
        required = {"seeds", "candidate_index", *score_keys.values()}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Score archive lacks keys: {sorted(missing)}")
        seeds = np.asarray(archive["seeds"], dtype=np.int64)
        candidate_index = np.asarray(archive["candidate_index"], dtype=np.int64)
        if seeds.tolist() != expected_seeds:
            raise ValueError(f"Expected seeds {expected_seeds}, found {seeds.tolist()}")
        if candidate_index.shape != (expected_pool_size,):
            raise ValueError(f"Unexpected candidate_index shape: {candidate_index.shape}")
        if len(np.unique(candidate_index)) != expected_pool_size:
            raise ValueError("candidate_index must contain unique values")
        scores: dict[str, np.ndarray] = {}
        for method, key in score_keys.items():
            array = np.asarray(archive[key], dtype=np.float64)
            expected = (len(expected_seeds), expected_pool_size)
            if array.shape != expected:
                raise ValueError(f"{method} score shape {array.shape}; expected {expected}")
            if not np.isfinite(array).all():
                raise ValueError(f"{method} scores contain non-finite values")
            scores[method] = array
    return candidate_index, scores


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    """Return tie-aware descending midrank percentiles, independently per seed."""
    result = np.empty_like(scores, dtype=np.float64)
    denominator = scores.shape[1]
    for seed_index in range(scores.shape[0]):
        result[seed_index] = 100.0 * rankdata(-scores[seed_index], method="average") / denominator
    return result


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for sorted_index, original_index in enumerate(order):
        value = min(1.0, (count - sorted_index) * float(p_values[original_index]))
        running = max(running, value)
        adjusted[original_index] = running
    return adjusted.tolist()


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def candidate_metadata(pool: pd.DataFrame) -> tuple[str, dict[int, dict[str, Any]]]:
    index_col = choose_column(
        pool, ["pool_index_zero_based", "candidate_index", "pool_index"]
    )
    gene_symbol_col = choose_column(pool, ["gene_symbol", "gene", "gene_name"], False)
    gene_id_col = choose_column(pool, ["gene_entrez_id", "gene_id", "entrez_id"], False)
    metadata: dict[int, dict[str, Any]] = {}
    for _, row in pool.iterrows():
        index = int(row[index_col])
        if index in metadata:
            raise ValueError(f"Duplicate candidate index in pool CSV: {index}")
        metadata[index] = {
            "gene_symbol": row[gene_symbol_col] if gene_symbol_col else "",
            "gene_entrez_id": row[gene_id_col] if gene_id_col else "",
        }
    return index_col, metadata


def compute_pair_ranks(
    documented: pd.DataFrame,
    metadata: dict[int, dict[str, Any]],
    candidate_positions: dict[int, int],
    rank_percentiles: dict[str, np.ndarray],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "validation_pair_id" not in documented:
        raise ValueError("Documented-pair CSV must contain validation_pair_id")
    candidate_list_col = choose_column(
        documented, ["candidate_indices", "pool_rows", "supporting_pool_rows"]
    )
    gene_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for _, pair in documented.iterrows():
        pair_id = str(pair["validation_pair_id"])
        indices = parse_indices(pair[candidate_list_col])
        if not indices:
            raise ValueError(f"No pool rows for documented pair {pair_id}")
        missing = [index for index in indices if index not in candidate_positions]
        if missing:
            raise ValueError(f"Pair {pair_id} references absent candidates: {missing}")
        for method, matrix in rank_percentiles.items():
            method_candidates: list[dict[str, Any]] = []
            for candidate_index in indices:
                values = matrix[:, candidate_positions[candidate_index]]
                row = {
                    "validation_pair_id": pair_id,
                    "method": method,
                    "candidate_index": candidate_index,
                    **metadata.get(candidate_index, {}),
                    "median_rank_percentile": float(np.median(values)),
                    "mean_rank_percentile": float(np.mean(values)),
                    "q1_rank_percentile": float(np.percentile(values, 25)),
                    "q3_rank_percentile": float(np.percentile(values, 75)),
                    "min_rank_percentile": float(np.min(values)),
                    "max_rank_percentile": float(np.max(values)),
                }
                for seed, value in zip(seeds, values):
                    row[f"seed_{seed}_rank_percentile"] = float(value)
                gene_rows.append(row)
                method_candidates.append(row)
            # The publication rule is applied independently to each method.
            best = min(
                method_candidates,
                key=lambda row: (
                    row["median_rank_percentile"],
                    row["mean_rank_percentile"],
                    row["candidate_index"],
                ),
            )
            pair_rows.append(
                {
                    "validation_pair_id": pair_id,
                    "condition_group": pair.get("condition_group", ""),
                    "drug_name": pair.get("drug_name", ""),
                    "disease_label": pair.get("disease_label", ""),
                    "method": method,
                    "candidate_count": len(indices),
                    "best_candidate_index": best["candidate_index"],
                    "best_gene_symbol": best.get("gene_symbol", ""),
                    "best_gene_entrez_id": best.get("gene_entrez_id", ""),
                    "selection_rule": "lowest ten-seed median rank percentile; ties by mean then index",
                    **{
                        key: value
                        for key, value in best.items()
                        if key.endswith("rank_percentile")
                    },
                }
            )
    return pd.DataFrame(gene_rows), pd.DataFrame(pair_rows)


def aggregate(pair_ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in pair_ranks.groupby("method", sort=False, observed=True):
        values = group["median_rank_percentile"].to_numpy(float)
        rows.append(
            {
                "method": method,
                "n_pairs": len(values),
                "mean_pair_rank_percentile": float(np.mean(values)),
                "median_pair_rank_percentile": float(np.median(values)),
                "q1_pair_rank_percentile": float(np.percentile(values, 25)),
                "q3_pair_rank_percentile": float(np.percentile(values, 75)),
            }
        )
    return pd.DataFrame(rows)


def compare_methods(
    pair_ranks: pd.DataFrame, reference: str, alternative: str
) -> pd.DataFrame:
    pivot = pair_ranks.pivot(
        index="validation_pair_id", columns="method", values="median_rank_percentile"
    )
    if reference not in pivot:
        raise ValueError(f"Reference method absent from pair-level ranks: {reference}")
    records = []
    raw_p = []
    for method in pivot.columns:
        if method == reference:
            continue
        subset = pivot[[reference, method]].dropna()
        ref = subset[reference].to_numpy(float)
        other = subset[method].to_numpy(float)
        difference = ref - other
        if np.allclose(difference, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(
                ref,
                other,
                alternative=alternative,
                zero_method="wilcox",
                method="auto",
            )
            statistic, p_value = float(test.statistic), float(test.pvalue)
        raw_p.append(p_value)
        records.append(
            {
                "reference_method": reference,
                "comparison_method": method,
                "n_pairs": len(subset),
                "wilcoxon_statistic": statistic,
                "p_raw": p_value,
                "median_paired_difference_reference_minus_comparison": float(
                    np.median(difference)
                ),
                "mean_paired_difference_reference_minus_comparison": float(
                    np.mean(difference)
                ),
            }
        )
    adjusted = holm_adjust(raw_p)
    for record, value in zip(records, adjusted):
        record["p_holm"] = value
        record["significance"] = significance_label(value)
        record["test"] = "paired two-sided Wilcoxon signed-rank"
        record["multiplicity_correction"] = "Holm"
    return pd.DataFrame(records)


def compute_case_studies(
    cases: pd.DataFrame,
    candidate_positions: dict[int, int],
    rank_percentiles: dict[str, np.ndarray],
    methods: list[str],
) -> pd.DataFrame:
    index_col = choose_column(
        cases, ["pool_index_zero_based", "candidate_index", "pool_index"]
    )
    label_col = choose_column(cases, ["motif", "case_label", "label"])
    panel_col = choose_column(cases, ["panel", "panel_label"], False)
    records = []
    for position, (_, case) in enumerate(cases.iterrows()):
        candidate_index = int(case[index_col])
        if candidate_index not in candidate_positions:
            raise ValueError(f"Case-study candidate is absent from score archive: {candidate_index}")
        pool_position = candidate_positions[candidate_index]
        for method in methods:
            values = rank_percentiles[method][:, pool_position]
            records.append(
                {
                    "panel": str(case[panel_col]) if panel_col else chr(ord("b") + position),
                    "case_label": case[label_col],
                    "validation_pair_id": case.get("validation_pair_id", ""),
                    "candidate_index": candidate_index,
                    "method": method,
                    "median_rank_percentile": float(np.median(values)),
                    "q1_rank_percentile": float(np.percentile(values, 25)),
                    "q3_rank_percentile": float(np.percentile(values, 75)),
                }
            )
    return pd.DataFrame(records)


def make_figure_source(
    aggregate_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    case_ranks: pd.DataFrame,
    reference: str,
    significance_baseline: str,
    aggregate_panel: str,
    aggregate_method_order: list[str],
) -> pd.DataFrame:
    source_rows = []
    comparison = comparisons[
        comparisons["comparison_method"].eq(significance_baseline)
    ]
    p_holm = float(comparison.iloc[0]["p_holm"]) if len(comparison) else np.nan
    label = str(comparison.iloc[0]["significance"]) if len(comparison) else ""
    for _, row in aggregate_summary.iterrows():
        method = str(row["method"])
        source_rows.append(
            {
                "panel": aggregate_panel,
                "panel_type": "aggregate",
                "panel_title": f"Mean across {int(row['n_pairs'])} clinically documented associations",
                "method": method,
                "rank_percentile": row["mean_pair_rank_percentile"],
                "bar_order": aggregate_method_order.index(method),
                "comparison_method": significance_baseline,
                "reference_method": reference,
                "p_holm": p_holm,
                "significance": label,
            }
        )
    baselines = [method for method in case_ranks.method.unique() if method != reference]
    for (panel, case_label), group in case_ranks.groupby(["panel", "case_label"], sort=False):
        baseline_rows = group[group.method.isin(baselines)]
        strongest = baseline_rows.loc[baseline_rows.median_rank_percentile.idxmin(), "method"]
        selected = group[group.method.isin([reference, strongest])]
        for _, row in selected.iterrows():
            method = str(row["method"])
            source_rows.append(
                {
                    "panel": panel,
                    "panel_type": "case_study",
                    "panel_title": case_label,
                    "method": method,
                    "rank_percentile": row["median_rank_percentile"],
                    "bar_order": 1 if method == reference else 0,
                    "comparison_method": strongest,
                    "reference_method": reference,
                    "p_holm": np.nan,
                    "significance": "",
                }
            )
    return pd.DataFrame(source_rows)


def analyze(config_path: Path) -> Path:
    config = load_config(config_path)
    paths = config["paths"]
    expected = config["expected"]
    result_dir = resolve_path(paths["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    documented_path = resolve_path(paths["documented_pairs"])
    pool_path = resolve_path(paths["gene_star_pool"])
    score_path = resolve_path(paths["score_archive"])
    cases_path = resolve_path(paths["case_studies"])
    for path in (documented_path, pool_path, score_path, cases_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis input is missing: {path}")

    documented = pd.read_csv(documented_path, low_memory=False)
    pool = pd.read_csv(pool_path, low_memory=False)
    cases = pd.read_csv(cases_path, low_memory=False)
    if len(documented) != int(expected["documented_pairs"]):
        raise ValueError(f"Expected {expected['documented_pairs']} documented pairs, found {len(documented)}")
    if len(pool) != int(expected["pool_size"]):
        raise ValueError(f"Expected {expected['pool_size']} candidates, found {len(pool)}")
    if len(cases) != int(expected["case_studies"]):
        raise ValueError(f"Expected {expected['case_studies']} case studies, found {len(cases)}")

    _, metadata = candidate_metadata(pool)
    seeds = [int(seed) for seed in config["seeds"]]
    score_keys = dict(config["score_keys"])
    candidate_index, scores = load_score_archive(
        score_path, score_keys, seeds, int(expected["pool_size"])
    )
    if set(candidate_index.tolist()) != set(metadata):
        raise ValueError("Pool CSV and NPZ candidate_index contain different candidate sets")
    candidate_positions = {int(index): position for position, index in enumerate(candidate_index)}
    rank_percentiles = {method: percentile_ranks(array) for method, array in scores.items()}

    gene_ranks, pair_ranks = compute_pair_ranks(
        documented, metadata, candidate_positions, rank_percentiles, seeds
    )
    method_order = list(score_keys)
    pair_ranks["method"] = pd.Categorical(pair_ranks.method, method_order, ordered=True)
    pair_ranks = pair_ranks.sort_values(["validation_pair_id", "method"])
    aggregate_summary = aggregate(pair_ranks)
    aggregate_summary["method"] = pd.Categorical(
        aggregate_summary.method, method_order, ordered=True
    )
    aggregate_summary = aggregate_summary.sort_values("method")

    analysis_config = config.get("analysis", {})
    reference = analysis_config.get("reference_method", "HANAMI")
    comparisons = compare_methods(
        pair_ranks,
        reference,
        analysis_config.get("wilcoxon_alternative", "two-sided"),
    )
    case_ranks = compute_case_studies(
        cases, candidate_positions, rank_percentiles, method_order
    )
    figure_config = config.get("figure", {})
    aggregate_order = figure_config.get("aggregate_method_order", method_order)
    if set(aggregate_order) != set(method_order):
        raise ValueError("figure.aggregate_method_order must list every method exactly once")
    figure_source = make_figure_source(
        aggregate_summary,
        comparisons,
        case_ranks,
        reference,
        analysis_config.get("figure_significance_baseline", "TriMoGCL"),
        str(figure_config.get("aggregate_panel", "a")),
        list(aggregate_order),
    )

    outputs = {
        "pair_gene_rank_percentiles.csv": gene_ranks,
        "pair_level_rank_percentiles.csv": pair_ranks,
        "aggregate_method_summary.csv": aggregate_summary,
        "wilcoxon_holm_results.csv": comparisons,
        "case_study_rank_percentiles.csv": case_ranks,
        "figure5_source_data.csv": figure_source,
    }
    for name, frame in outputs.items():
        frame.to_csv(result_dir / name, index=False)

    manifest = {
        "stage": "analyze_clinical_concordance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": manifest_path(config_path),
        "inputs": {
            str(path.name): {"path": manifest_path(path), "sha256": sha256(path)}
            for path in (documented_path, pool_path, score_path, cases_path)
        },
        "score_archive_schema": {
            "seeds": seeds,
            "candidate_count": len(candidate_index),
            "methods": score_keys,
        },
        "best_gene_rule": analysis_config.get(
            "best_gene_rule", "lowest_ten_seed_median_rank_percentile"
        ),
        "statistical_test": "paired two-sided Wilcoxon signed-rank with Holm correction",
        "outputs": {
            name: {"rows": len(frame), "sha256": sha256(result_dir / name)}
            for name, frame in outputs.items()
        },
        "limitations": [
            "The 405 pairs were selected using HANAMI outputs and are not an independent benchmark.",
            "Clinical registry concordance does not establish efficacy or a causal role for the shared gene.",
            "The shared gene is an observed MS relation; HANAMI ranks the missing relation rather than discovering the gene.",
        ],
    }
    (result_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote analysis tables to {result_dir}")
    return result_dir / "figure5_source_data.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    args = parser.parse_args()
    analyze(args.config.resolve())


if __name__ == "__main__":
    main()
