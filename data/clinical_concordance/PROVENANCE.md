# Clinical concordance data provenance

This package uses final ten-seed score arrays, a fixed 46,704-row MS gene-star candidate pool, and frozen clinical-screen tables. The earlier provisional single-seed helper run and its candidate CSV files are not inputs.

## Frozen analysis boundary

The reproducible workflow begins with:

- 401 automatic Phase II/III ClinicalTrials.gov matches in `frozen_inputs/automatic_phase23_pairs_401.csv`;
- six manual review records in `frozen_inputs/manually_reviewed_key_cases.csv`, of which four are accepted additions;
- the frozen HANAMI consensus pair universe in `frozen_inputs/consensus_drug_disease_pairs_hard9of10.csv`.

The workflow reconstructs the 401+4 association set and verifies it against `clinically_documented_405.csv`. It does not recreate the 401 automatic matches from raw ClinicalTrials.gov exports because the original registry matching and disease-crosswalk code is not part of this package.

## Checksums

SHA-256 checksums for the required frozen inputs are listed below.

| File | SHA-256 |
|---|---|
| `gene_star_candidates.csv` | `34a20324567d94f2b4d7ddbf3c5a0f0a570685f0ffd06391aae757e97e0b1a6b` |
| `gene_star_scores_10seeds.npz` | `7504696c54b3353fdad316abe9b84d584fc2bc9f76e000b92c6dd7b20dba2d18` |
| `clinically_documented_405.csv` | `7bfb4e7207bd0e7a1649c514c0c4a8f35c21bd8285ed7eddd5b348dd2ae18d86` |
| `figure5_cases_7.csv` | `0cf6dd3ca6674f1812ca72371ddc793dae56d2036f638d04ae5b2210c9c9d0b8` |
| `data_dictionary.csv` | `e1605db36a367ab59b0849ba47c9791d87fe5c679e454a6f7a9175572f783108` |
| `frozen_inputs/automatic_phase23_pairs_401.csv` | `7041ea2771fbe343a533177d6b1147ba576f18d313f95d0ac15ffaafb582ac7e` |
| `frozen_inputs/manually_reviewed_key_cases.csv` | `984f16bc9a46bafcf1d8906d2d7ae4bbbd43ca32b6032f94f34561d6a544e715` |
| `frozen_inputs/consensus_drug_disease_pairs_hard9of10.csv` | `65f85b28159e5ba3c6b070a14148f69428c1aec1e302a48d82f5b72bd91d3711` |

The analysis scripts also write run-specific manifests containing checksums for all consumed inputs and generated outputs.

## Scope notes

- The 401 automatic matches were not all manually verified. They provide registry concordance rather than evidence of clinical efficacy or FDA approval.
- Five of the seven Fig. 5 examples belong to the 405-association set. Aspirin-PTGS2-Colorectal Neoplasms and Pentoxifylline-TNF-Parkinson Disease are separately reviewed illustrations outside that set.
- The shared gene is an existing MS relation. HANAMI ranks the missing drug-disease completion but does not discover or causally validate the intermediary gene.
- Some rows lack a gene symbol because the corresponding Entrez identifier had no symbol in the final MS supplementary mapping; the Entrez identifier remains available.
