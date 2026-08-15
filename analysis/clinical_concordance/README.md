# Biological context and post hoc clinical concordance

This directory contains the analysis used for the biological-context examples and Fig. 5 of the HANAMI manuscript. The analysis starts from gene-star configurations in the MS dataset. In each configuration, MS contains a drug-gene relation and a gene-disease relation but no direct drug-disease relation. HANAMI ranks completion of the missing drug-disease relation as a clique.

The shared gene is an existing MS relation, not a gene discovered by HANAMI. It provides an explicit biological context in the model output and can be treated as a candidate intermediary for follow-up. This analysis does not provide model-level feature attribution and does not establish causality, a route of action, therapeutic efficacy, or prospective discovery.

## Analysis overview

1. Build the fixed pool of 46,704 MS gene-star configurations and verify the status of all three relations.
2. Load the full candidate scores from ten seeds for HANAMI, TriMoGCL, TriNet, N2V-MLP, and RF.
3. Rank all candidates within each method and seed. Lower rank percentiles indicate better prioritization.
4. For every drug-disease association, calculate the ten-seed median rank percentile for each gene configuration. If an association has more than one shared gene, retain the best-ranked gene separately for each method.
5. Construct the 405 clinically documented association set from 401 automatic Phase II/III matches and four manually reviewed additions.
6. Compare the 405 paired method-level percentiles with two-sided Wilcoxon signed-rank tests and apply Holm correction across baseline comparisons.
7. Generate the aggregate panel and the seven post hoc examples in Fig. 5. Five examples are members of the 405-set; aspirin-PTGS2-colorectal neoplasms and pentoxifylline-TNF-Parkinson disease are separately reviewed illustrations outside that set.

The 405 associations were selected through a post hoc process that depended on HANAMI predictions. They are therefore not an independent benchmark. Comparisons with baselines describe relative rankings within this selected set.

## Required inputs

The following files must be present before the analysis is run:

| Path | Purpose |
|---|---|
| `data/ms/Compound-Disease-feat-hierarchy.npy` | MS drug-disease adjacency matrix |
| `data/ms/Gene-Compound-feat-hierarchy.npy` | MS drug-gene adjacency matrix |
| `data/ms/Gene-Disease-feat-hierarchy.npy` | MS gene-disease adjacency matrix |
| `data/ms/id2drug.npy` | Drug identifier mapping |
| `data/ms/id2gene.npy` | Gene identifier mapping |
| `data/ms/id2dise.npy` | Disease identifier mapping |
| `data/clinical_concordance/gene_star_candidates.csv` | Fixed pool of 46,704 MS gene-star configurations |
| `data/clinical_concordance/hanami_consensus_gene_stars.csv` | HANAMI consensus selections used in the post hoc clinical match |
| `data/clinical_concordance/gene_star_scores_10seeds.npz` | Full candidate scores for five methods across ten seeds |
| `data/clinical_concordance/clinicaltrials_snapshot.csv.gz` | Frozen ClinicalTrials.gov records used for matching |
| `data/clinical_concordance/clinicaltrials_metadata.json` | Export date, query, source version, checksums, and matching settings |
| `data/clinical_concordance/frozen_inputs/automatic_phase23_pairs_401.csv` | Frozen automatic Phase II/III registry matches |
| `data/clinical_concordance/frozen_inputs/manually_reviewed_key_cases.csv` | Six manual reviews, including the four accepted additions |
| `data/clinical_concordance/frozen_inputs/consensus_drug_disease_pairs_hard9of10.csv` | Frozen HANAMI consensus pair universe used to recover candidate indices |

`gene_star_scores_10seeds.npz` must contain ten complete score arrays for each of HANAMI, TriMoGCL, TriNet, N2V-MLP, and RF, aligned row-for-row with `gene_star_candidates.csv`. The reported runs use seeds 0, 10, 20, 30, 40, 50, 60, 70, 80, and 90. Partial arrays, summary statistics, or scores only for the 405 associations cannot reproduce the ranks because every percentile is calculated against the full candidate pool.

## Generated data files

| Path | Contents |
|---|---|
| `data/clinical_concordance/clinically_documented_405.csv` | Final 405 associations, clinical identifiers, match provenance, and inclusion route |
| `data/clinical_concordance/figure5_cases_7.csv` | Seven examples displayed in Fig. 5b-h, with evidence, method values, and explicit 405-set membership |

`gene_star_candidates.csv` must explicitly record `drug_gene_in_ms = 1`, `gene_disease_in_ms = 1`, and `drug_disease_in_ms = 0`. These fields are the auditable basis for the topology-level biological context. `clinically_documented_405.csv` must distinguish the 401 automatic matches from the four manually reviewed additions so that the complete selection process remains traceable.

## Generated result files

| Path | Contents |
|---|---|
| `results/clinical_concordance/pair_level_rank_percentiles.csv` | Per-association method percentiles after the ten-seed median and per-method best-gene rule |
| `results/clinical_concordance/aggregate_method_summary.csv` | Group-level summary used in Fig. 5a |
| `results/clinical_concordance/wilcoxon_holm_results.csv` | Paired Wilcoxon tests and Holm-adjusted P values |
| `results/clinical_concordance/figure5_source_data.csv` | Complete source data for Fig. 5a-h |
| `results/clinical_concordance/figure5.pdf` | Manuscript figure |
| `results/clinical_concordance/figure5.png` | Raster preview |

## Reproduce the analysis

Run all commands from the repository root:

```bash
python analysis/clinical_concordance/run_all.py
```

The stages can also be run separately:

```bash
python analysis/clinical_concordance/build_clinical_concordance.py
python analysis/clinical_concordance/analyze_clinical_concordance.py
python analysis/clinical_concordance/plot_figure5.py
```

The pipeline must stop with an error if any method lacks one of the ten full candidate-score arrays. Do not replace missing seed outputs with group averages or values copied from the plotted figure.

## Interpretation and limitations

- Clinical documentation supports the drug-disease relation, not the shared gene as the causal route.
- A ClinicalTrials.gov record or publication does not by itself establish efficacy or regulatory approval.
- Broad MS disease categories may include conditions that are more general than the clinical record used for support.
- Highly connected genes such as PTGS2 and TNF may provide several alternative network paths.
- The seven displayed cases are manually reviewed illustrations and do not form an independent evaluation set. Five belong to the 405-set; two were reviewed separately and are identified in `figure5_cases_7.csv`.
- Only outputs from the final ten-seed models should be used. Results from the earlier single-seed helper run are exploratory and are not a source for Fig. 5.

See `data/clinical_concordance/data_dictionary.csv` and `VALIDATION_REPORT.md` for column definitions, source versions, checksums, and validation results.
