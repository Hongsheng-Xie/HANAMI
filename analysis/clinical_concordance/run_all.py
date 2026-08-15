#!/usr/bin/env python3
"""Run the frozen-set build, statistical analysis, and Fig. 5 rendering."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_run_manifest(config_path: Path, stages: list[str]) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    results_dir = resolve_repo_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        "pair_gene_rank_percentiles.csv",
        "pair_level_rank_percentiles.csv",
        "aggregate_method_summary.csv",
        "wilcoxon_holm_results.csv",
        "case_study_rank_percentiles.csv",
        "figure5_source_data.csv",
        "figure5.pdf",
        "figure5.png",
        "build_manifest.json",
        "analysis_manifest.json",
    ]
    outputs = []
    for name in expected_outputs:
        path = results_dir / name
        if path.exists():
            outputs.append(
                {
                    "path": path.relative_to(REPO_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_path.relative_to(REPO_ROOT).as_posix()
        if config_path.is_relative_to(REPO_ROOT)
        else str(config_path),
        "config_sha256": sha256(config_path),
        "stages": stages,
        "outputs": outputs,
    }
    with (results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def run(script: str, config: Path, extra: list[str] | None = None) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script), "--config", str(config)]
    if extra:
        command.extend(extra)
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip reconstruction of the 405-set from its frozen inputs",
    )
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    stages: list[str] = []
    if not args.skip_build:
        # This writes a verification copy under results/; it never replaces
        # the packaged publication table under data/.
        run("build_clinical_concordance.py", config, ["--force"])
        stages.append("build_clinical_concordance")
    run("analyze_clinical_concordance.py", config)
    stages.append("analyze_clinical_concordance")
    if not args.skip_plot:
        run("plot_figure5.py", config)
        stages.append("plot_figure5")
    write_run_manifest(config, stages)


if __name__ == "__main__":
    main()
