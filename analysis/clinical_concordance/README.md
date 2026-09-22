# Clinical concordance plots

`plot_figure5.Rmd` summarizes the ten-seed motif rankings and prepares Figure 5 for the MS clinical evidence cohort of 1,630 motifs from 785 drug–disease pairs. Panels a–b show cohort performance, and panels c–g show five case studies.

Inputs are the saved scores, cohort metadata and case table described in the [data guide](../../data/clinical_concordance/). Run from the repository root:

```sh
pip install -r analysis/clinical_concordance/requirements.txt
python run_clinical_concordance.py
```

Required R packages: `knitr`, `yaml`, `ggplot2`, `tidyr`, `dplyr`, `patchwork` and `scales`.

The runner prepares the input tables and executes the Rmd. Outputs are saved in `results/clinical_concordance/figure5/`, including the PDF, PNG, plotted values and statistical results. Use `--rscript PATH` to locate R, `--output-dir PATH` to change the destination, or `--skip-plot` to prepare the input tables only.

Calculation notes accompany the relevant Rmd code. [Provenance](../../data/clinical_concordance/current/PROVENANCE.md) documents the data sources and evidence review.
