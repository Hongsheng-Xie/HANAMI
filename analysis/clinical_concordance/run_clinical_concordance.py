#!/usr/bin/env python3
"""Run the complete post hoc clinical-concordance workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_run_manifest(config_path: Path, stages: list[str]) -> None:
    config = load_config(config_path)
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
        "figure5_render.html",
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


def run_python(script: str, config: Path, extra: list[str] | None = None) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script), "--config", str(config)]
    if extra:
        command.extend(extra)
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_figure5(config_path: Path) -> None:
    rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if rscript is None:
        raise RuntimeError(
            "Rscript was not found. Install R and the rmarkdown, ggplot2, and "
            "patchwork packages, "
            "or rerun with --skip-plot."
        )

    config = load_config(config_path)
    results_dir = resolve_repo_path(config["paths"]["results_dir"])
    figure = config.get("figure", {})
    def r_string(value: str | Path) -> str:
        return '"' + Path(value).as_posix().replace('"', '\\"') + '"'

    report_path = results_dir / "figure5_render.html"
    render_expression = (
        "rmarkdown::render("
        f"input={r_string(SCRIPT_DIR / 'plot_figure5.Rmd')},"
        f"output_file={r_string(report_path.name)},"
        f"output_dir={r_string(results_dir)},"
        "params=list("
        f"source={r_string(results_dir / 'figure5_source_data.csv')},"
        f"output_pdf={r_string(results_dir / 'figure5.pdf')},"
        f"output_png={r_string(results_dir / 'figure5.png')},"
        f"width={float(figure.get('width_inches', 13.5))},"
        f"height={float(figure.get('height_inches', 6.6))},"
        f"dpi={int(figure.get('dpi', 300))}"
        "),quiet=TRUE,envir=new.env(parent=globalenv()))"
    )
    command = [rscript, "-e", render_expression]
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
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Calculate the numerical results without rendering Fig. 5",
    )
    args = parser.parse_args()
    config = args.config.resolve()
    stages: list[str] = []

    if not args.skip_build:
        # Write a verification copy under results/ without replacing the
        # packaged publication table under data/.
        run_python("build_clinical_concordance.py", config, ["--force"])
        stages.append("build_clinical_concordance")

    run_python("analyze_clinical_concordance.py", config)
    stages.append("analyze_clinical_concordance")

    if not args.skip_plot:
        run_figure5(config)
        stages.append("plot_figure5")

    write_run_manifest(config, stages)


if __name__ == "__main__":
    main()
