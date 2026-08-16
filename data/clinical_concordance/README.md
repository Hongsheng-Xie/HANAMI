# Clinical concordance data

This directory contains the fixed inputs and provenance files for the post hoc MS gene-star analysis. Generated statistics, rank tables, and figures are written locally to `results/clinical_concordance/` and are not committed.

## Runtime inputs

| File | Role |
|---|---|
| `gene_star_candidates.csv` | Fixed, ordered pool of 46,704 MS gene-star candidates. |
| `gene_star_scores_10seeds.npz` | Full candidate-score arrays for HANAMI and four baselines across ten seeds. |
| `clinically_documented_405.csv` | Canonical 405-pair analysis table used to verify the reconstructed clinical set. |
| `figure5_cases_7.csv` | Identities, evidence, and set-membership metadata for the seven illustrative cases. Method ranks are recalculated from the score archive. |

## Frozen clinical-set inputs

The workflow reconstructs the 405-pair set from the following files in `frozen_inputs/`.

| File | Role |
|---|---|
| `automatic_phase23_pairs_401.csv` | Frozen set of 401 automatic Phase II or III registry matches. |
| `manually_reviewed_key_cases.csv` | Six manually reviewed records, including the four accepted additions. |
| `consensus_drug_disease_pairs_hard9of10.csv` | HANAMI consensus-pair universe used to recover candidate identifiers. |

The packaged workflow begins from these frozen tables. It does not repeat the original condition matching and disease crosswalk against the raw ClinicalTrials.gov exports.

## Documentation and provenance

| File | Role |
|---|---|
| `data_dictionary.csv` | Field definitions for the runtime and frozen clinical-set inputs. |
| `PROVENANCE.md` | Input checksums, construction boundary, and scope limitations. |
| `clinicaltrials_snapshot.csv.gz` | Frozen registry snapshot retained for audit purposes; it is not read by `run_all.py`. |
| `clinicaltrials_metadata.json` | Query dates, source information, and snapshot metadata. |
| `hanami_consensus_gene_stars.csv` | Supporting candidate-selection provenance; it is not read by `run_all.py`. |

For execution instructions, see [`analysis/clinical_concordance/README.md`](../../analysis/clinical_concordance/README.md). Clinical registry concordance does not establish efficacy, regulatory approval, or a causal role for the shared gene.
