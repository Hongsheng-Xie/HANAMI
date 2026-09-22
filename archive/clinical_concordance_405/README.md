# Historical 405-pair analysis

These scripts preserve the previous 405-pair workflow: median rank percentiles
across ten seeds, per-method best gene configuration for each pair, and two-sided
Wilcoxon tests with Holm correction. They do **not** generate the current
1,630-motif Figure 5, which uses arithmetic means across seeds and the current
tests described in [the current workflow](../../analysis/clinical_concordance/).

The only execution change is the repository-root calculation after relocation.
The configuration retains historical input/output paths. Use an isolated checkout
with the corresponding historical inputs, or copy the configuration and change
its output paths before running; do not overwrite current Figure 5 outputs.

From the repository root, the historical Python stages are:

```sh
python archive/clinical_concordance_405/build_clinical_concordance.py --config archive/clinical_concordance_405/config.yaml
python archive/clinical_concordance_405/analyze_clinical_concordance.py --config archive/clinical_concordance_405/config.yaml
```

The historical figure renderer and root-level runner are preserved in Git
history at commit `a497768b43e79fd59502aaeebc094347ac658789`, respectively
`analysis/clinical_concordance/plot_figure5.Rmd` and
`run_clinical_concordance.py`. The current Figure 5 is rendered directly from
`analysis/clinical_concordance/plot_figure5.Rmd`; these archived stages belong
only to the historical analysis.
