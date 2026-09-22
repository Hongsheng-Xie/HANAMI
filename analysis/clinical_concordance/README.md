# Figure 5: complete-cohort clinical concordance

This is the current manuscript workflow. It evaluates 1,630 MS gene-star motifs
from 785 drug–disease pairs and 109 disease labels. The obsolete 405-pair,
seven-case workflow is retained separately in [legacy_405](legacy_405/).

## Reproduce without training

Python 3.10 or newer, NumPy, pandas and SciPy are required for reconstruction and tests.
Install the Python dependencies and the R packages below, then run from the repository root:

```sh
pip install -r analysis/clinical_concordance/requirements.txt
python run_clinical_concordance.py
```

```r
install.packages(c("knitr", "yaml", "ggplot2", "tidyr", "dplyr", "patchwork", "scales"))
```

The runner evaluates `plot_figure5.Rmd` using knitr. No Pandoc, LaTeX, GPU,
ClinicalTrials.gov request, or model retraining is needed. Supply
`--rscript PATH` if Rscript is not on PATH. `--output-dir PATH` changes the
output location. `--skip-plot` reconstructs only the mean10 and case-seed
tables; the statistical tests and figure are calculated in the Rmd.

The Rmd can also be rendered with rmarkdown, if Pandoc is installed. Its defaults
use the same inputs. There are no local-machine data paths or Top-5 dependencies.

## Inputs and provenance

The [current manifest](../../data/clinical_concordance/current/manifest.json)
records SHA256 checksums, identifiers, seed order, feature names and score sources.
The builders verify these before calculating results.

- All methods rank the same 46,704 candidates in `gene_star_candidates.csv`.
- The existing `gene_star_scores_10seeds.npz` supplies only the four baselines.
  Its historical HANAMI array is intentionally not used.
- `current/hanami_scores/` supplies the ten current HANAMI runs with
  `dise_All.pth`, `drug_All.pth` and `gene_All.pth`. The original disease-file
  name was `dise_Bio.pth`; it is byte-identical to the current `dise_All.pth`.
- Seeds are 0, 10, 20, 30, 40, 50, 60, 70, 80 and 90. Do not replace seed 0
  with seed 1 from the separate classification/cost experiments.
- `cohort_metadata.tsv` preserves the frozen strict-plus-manual clinical cohort
  and its supporting trial IDs. No model-score filter or disease Top-k filter is
  applied by this reproduction workflow.
- The cohort combines 353 manually reviewed pairs (965 motifs) and 432 additional
  strict automatic matches (665 motifs). The latter used the disease-directed
  Phase II/III trial-title rule. These are distinct evidence-source categories,
  not a claim that every manually reviewed row passed that same automatic rule.

This package starts from the completed evidence audit. It does not repeat a
ClinicalTrials.gov search or independently re-adjudicate every trial.
A registered trial supports clinical investigation, not established treatment
efficacy, FDA approval, or a causal role for the shared gene. Combination regimens
do not isolate one drug component's efficacy. Some MS disease labels are broader
than the trial condition, for example Hypersensitivity versus hypersensitivity
pneumonitis. Case-level trial details remain in the five-case input table.

The optional [HANAMI score-generation source](score_generation/) and
[original run metadata](../../data/clinical_concordance/current/training_provenance/)
are supplied separately. The default reproduction command never trains models.
The portable training source preserves the recorded algorithm, but a new training
run has not been validated for bitwise equality to the frozen scores.

## Calculations

For each method and seed, descending scores are ranked over the complete pool,
using average ranks for ties. Percentile is `100 * rank / 46704`; lower is better.
Each motif's value is the arithmetic mean of its ten seed-specific percentiles.
Neither a median nor a per-pair best-gene selection is used.

- Panel a: percentage of motifs for which each method has the unique lowest
  mean percentile. Ties are counted separately; there are none in this release.
- Panel b: arithmetic mean of the 1,630 motif values for each method.
- Panels c–g: five fixed cases, using the same ten-seed arithmetic means.

Panels a and b compare HANAMI with TriMoGCL using one-sided CR1 cluster-robust
t tests with 785 drug–disease pair clusters and 784 degrees of freedom. The
paired observations are motif-level winner indicators or percentiles. Error
bars are each method's marginal 95% CR1 confidence intervals, conditional on the
ten-seed means, not intervals for the paired contrast.

Cases c–g use one-sided paired t tests across ten matched seeds against the
baseline with the lowest mean percentile in that panel. Error bars are sample
standard deviations across seeds. These case tests are exploratory, unadjusted,
and do not account for case selection. No Holm correction is applied.
Shapiro–Wilk and Wilcoxon values are exported as diagnostics, not used to select
which test or significance symbol to report.

## Verified output

| Endpoint | HANAMI | TriMoGCL | One-sided P |
|---|---:|---:|---:|
| Motifs ranked best | 32.1472% (524) | 25.2761% (412) | 0.0151253 |
| Mean rank percentile | 35.6926% | 36.5774% | 0.0199483 |

The current cases are Lenvatinib–FLT1–Melanoma, Pirfenidone–TNF–Hypersensitivity,
Enalapril–ACE–Coronary Artery Disease, Roflumilast–PDE4D–Diabetes Mellitus,
and Scopolamine–CHRM2–Bipolar Disorder.

Outputs go to [results/clinical_concordance/figure5](../../results/clinical_concordance/figure5/):
the PDF and PNG, plot source data, cohort tests, case tests, case means and
seed values. The PDF is 14 × 8 inches (1008 × 576 points).
Red stars denote P < 0.05, 0.01 and 0.001; nonsignificant comparisons use n.s.
The five current cases each have one star. Statistical definitions belong in
the manuscript legend, not in a footnote on the figure.

The regenerated mean10 table matches the frozen released table byte-for-byte.
The output manifest hashes the generated files; figure file hashes can differ
between R versions or rendering platforms even when plotted numbers agree.

## Checks

```sh
python -m unittest discover -s analysis/clinical_concordance/tests -v
python analysis/clinical_concordance/verify_figure5_outputs.py
```

Run the second check after the full runner. It validates all 35 bar values,
both cohort tests, five case tests and error-bar definitions against the release.
