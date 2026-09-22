# Optional Figure 5 tools

Normal plotting uses [`plot_figure5.Rmd`](../../analysis/clinical_concordance/plot_figure5.Rmd)
and the prepared inputs. These tools provide command-line rendering, raw-score
reconstruction and independent checks.

## Render without Pandoc

From the repository root, with the R packages listed in the
[plotting guide](../../analysis/clinical_concordance/README.md) installed:

```sh
Rscript tools/figure5/render_figure5.R
```

An optional output-directory argument changes where the figure and tables are
saved. Input paths remain those specified in the Rmd.

## Reconstruct or verify data

```sh
pip install -r tools/figure5/requirements.txt
python tools/figure5/prepare_data.py
python -m unittest discover -s tools/figure5/tests -v
python tools/figure5/verify_outputs.py
```

`prepare_data.py` checks the source hashes and rebuilds both the cohort means
and case-seed values from the saved scores in one pass. It verifies them against
the prepared inputs and writes to `results/clinical_concordance/figure5/`.
Use `--output-dir PATH` for another destination; source inputs are preserved.

Run `verify_outputs.py` after rendering to check the plotted values, significance
tests and error bars. It also accepts `--output-dir PATH`. The
[input provenance](../../data/clinical_concordance/current/PROVENANCE.md)
documents the sources and optional model-training records.
