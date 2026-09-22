#!/usr/bin/env python3
"""Reproduce the current complete-cohort Figure 5 from archived scores, without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis/clinical_concordance"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/clinical_concordance/figure5")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Reconstruct mean10 and case-seed tables only; tests and plots require R")
    parser.add_argument("--rscript", help="Path to Rscript if it is not on PATH")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = []
    for script in ("build_mean_rank_percentile_source.py", "build_figure5_errorbar_data.py"):
        command = [sys.executable, str(ANALYSIS / script), "--output-dir", str(output)]
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        stages.append(script)
    if not args.skip_plot:
        rscript = args.rscript or shutil.which("Rscript") or shutil.which("Rscript.exe")
        if not rscript:
            raise RuntimeError("Rscript not found. Supply --rscript PATH or use --skip-plot.")
        # knitr evaluates the canonical Rmd; no Pandoc or LaTeX installation is needed.
        command = [rscript, str(ANALYSIS / "render_figure5.R"), str(output)]
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        stages.append("plot_figure5.Rmd")
    names = ["validated_gene_star_motifs_1630_mean10.tsv", "figure5_case_seed_percentiles.csv"]
    if not args.skip_plot:
        names += ["figure5_plot_source.csv", "figure5_complete_cohort_cluster_robust_results.csv",
                  "figure5_case_rank_summary.csv", "figure5_case_seed_tests.csv", "figure5.pdf", "figure5.png"]
    outputs = [{"file": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
               for name in names]
    manifest = {"workflow": "current complete-cohort Figure 5", "stages": stages,
                "training_rerun": False, "mean_not_median": True, "outputs": outputs}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified outputs: {output}")


if __name__ == "__main__":
    main()
