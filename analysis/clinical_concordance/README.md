# Post hoc clinical concordance analysis

This directory contains the code used to reproduce the MS gene-star ranking analysis reported in the HANAMI manuscript. A gene-star candidate contains an observed drug-gene relation and an observed gene-disease relation but no direct drug-disease relation in MS. The analysis ranks completion of that missing relation; it does not infer the shared gene or establish a causal mechanism.

## Scope

The workflow begins with a frozen set of 401 automatic Phase II/III registry matches and four manually reviewed additions. It reconstructs the 405-association table, calculates ten-seed rank percentiles for HANAMI and four baselines, performs paired Wilcoxon signed-rank tests with Holm correction, and prepares the source data for Fig. 5.

It does not rerun the original ClinicalTrials.gov condition and intervention matching from the raw registry exports. The frozen registry matches are the auditable starting point for this package. The 405 associations were selected through a post hoc process that depended on HANAMI predictions and therefore do not constitute an independent benchmark.

## Required inputs

All paths are relative to the repository root.

| Path | Purpose |
|---|---|
| `data/clinical_concordance/gene_star_candidates.csv` | Fixed pool of 46,704 MS gene-star candidates. The topology fields are `drug_gene_edge_in_ms`, `gene_disease_edge_in_ms`, and `drug_disease_edge_in_ms`. |
| `data/clinical_concordance/gene_star_scores_10seeds.npz` | Candidate scores for HANAMI, TriMoGCL, TriNet, N2V-MLP, and RF across ten seeds. |
| `data/clinical_concordance/frozen_inputs/automatic_phase23_pairs_401.csv` | Frozen automatic Phase II/III registry matches. |
| `data/clinical_concordance/frozen_inputs/manually_reviewed_key_cases.csv` | Manual reviews used to identify the four accepted additions. |
| `data/clinical_concordance/frozen_inputs/consensus_drug_disease_pairs_hard9of10.csv` | Frozen HANAMI consensus pair universe used to recover candidate identifiers. |
| `data/clinical_concordance/clinically_documented_405.csv` | Canonical 405-association table used to verify the reconstructed set. |
| `data/clinical_concordance/figure5_cases_7.csv` | Metadata for the seven illustrative cases. Five belong to the 405 set and two are separate examples. |

The score archive must contain the keys `seeds`, `candidate_index`, `hanami`, `trimogcl`, `trinet`, `n2v_mlp`, and `rf`. Each method array must have shape `(10, 46704)` and follow the row order in `gene_star_candidates.csv`. The reported seeds are 0, 10, 20, 30, 40, 50, 60, 70, 80, and 90.

The original MS matrices, identifier maps, ClinicalTrials.gov snapshot, and candidate-discovery tables are source-provenance materials rather than runtime inputs for this workflow.

## Run the analysis

From the repository root:

```bash
pip install -r analysis/clinical_concordance/requirements.txt
python analysis/clinical_concordance/run_all.py
```

To calculate the numerical results without rendering Fig. 5:

```bash
python analysis/clinical_concordance/run_all.py --skip-plot
```

The stages can also be run separately:

```bash
python analysis/clinical_concordance/build_clinical_concordance.py
python analysis/clinical_concordance/analyze_clinical_concordance.py
python analysis/clinical_concordance/plot_figure5.py
```

The plotting script is optional. It reads the generated `figure5_source_data.csv` and writes PDF and PNG versions of Fig. 5.

## Generated outputs

Running the workflow creates files under `results/clinical_concordance/`, including:

- the reconstructed 405-association table and its verification manifest;
- candidate- and pair-level rank percentiles;
- aggregate method summaries;
- paired Wilcoxon results with Holm-adjusted P values;
- seven-case rank summaries and Fig. 5 source data;
- optional PDF and PNG figure files.

These files are generated artifacts and do not need to be committed. The frozen inputs above are sufficient to reproduce them.

## Interpretation

Clinical registry concordance supports the presence of a drug-disease relation in an external record. It does not establish efficacy, regulatory approval, or the shared gene as the causal route. Broad MS disease categories may also be more general than the condition stated in an individual registry record.

See the [data dictionary](../../data/clinical_concordance/data_dictionary.csv) for field definitions and [provenance note](../../data/clinical_concordance/PROVENANCE.md) for frozen-input checksums and known scope limitations.
