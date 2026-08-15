# Clinical-concordance data package validation

Generated (UTC): 2026-08-15T07:01:29.020983+00:00

## Source provenance

- Only final ten-seed score arrays, the frozen 46,704-row MS gene-star pool, and frozen clinical-screen artifacts were used.
- The provisional single-seed helper extractor and its candidate CSVs were not read.
- Candidate-pool SHA-256: `96fb682138f648b377d14a55c02c75d2396746406995a6e00a06084a9858cbc2`.

## File validation

| File | Bytes | Data rows | SHA-256 |
|---|---:|---:|---|
| gene_star_candidates.csv | 5308021 | 46704 | `34a20324567d94f2b4d7ddbf3c5a0f0a570685f0ffd06391aae757e97e0b1a6b` |
| hanami_consensus_gene_stars.csv | 5088138 | 7734 | `7e01a195d9c0ae4e9868df4f3c214328eab194cd6ba318adccaeace61d3e1219` |
| gene_star_scores_10seeds.npz | 8493401 | n/a | `7504696c54b3353fdad316abe9b84d584fc2bc9f76e000b92c6dd7b20dba2d18` |
| clinicaltrials_metadata.json | 20019 | n/a | `63793028f2a1a9db8e051cc5ff794df7c1e6d074e4cb1b26bdf50faa44ff4bf2` |
| clinicaltrials_snapshot.csv.gz | 5421432 | 22760 | `67e14d61214147db2083988699361aec96cab8432a8802713c149ea4ae274d40` |
| clinically_documented_405.csv | 354621 | 405 | `7bfb4e7207bd0e7a1649c514c0c4a8f35c21bd8285ed7eddd5b348dd2ae18d86` |
| figure5_cases_7.csv | 2944 | 7 | `0cf6dd3ca6674f1812ca72371ddc793dae56d2036f638d04ae5b2210c9c9d0b8` |
| data_dictionary.csv | 20855 | 161 | `e1605db36a367ab59b0849ba47c9791d87fe5c679e454a6f7a9175572f783108` |

## Structural and numerical checks

- `gene_star_candidates.csv`: 46,704 candidates; every row has drug-gene = 1, gene-disease = 1, drug-disease = 0 in the final MS hierarchy arrays.
- Identifier resolution: {"disease_name": 0, "drug_name": 0, "gene_symbol": 1016}.
- `hanami_consensus_gene_stars.csv`: 7,734 hard-consensus rows; every candidate index and global triplet matches the fixed pool.
- `gene_star_scores_10seeds.npz`: exact keys `seeds`, `candidate_index`, `hanami`, `trimogcl`, `trinet`, `n2v_mlp`, `rf`; every method has shape (10, 46704), all values are finite, and values are exactly equal to the source arrays.
- `clinically_documented_405.csv`: 405 unique pairs = 401 automatic Phase II/III registry matches + 4 manually reviewed strong registry records; the two manually rejected key cases were not included.
- `clinicaltrials_snapshot.csv.gz`: 22,760 rows from 20 frozen disease-node exports; compressed size 5,421,432 bytes.
- `figure5_cases_7.csv`: 7 exact plotted cases; all candidate identities resolved uniquely. Recomputed-vs-source rounded rank mismatches: 0.
- `data_dictionary.csv`: 161 field definitions covering all packaged CSV and NPZ fields.

## Unresolved issues requiring manuscript/repository clarification

1. Only 5/7 Fig. 5 cases belong to the exact 405-association set. Outside the set: Aspirin-PTGS2-Colorectal Neoplasms, Pentoxifylline-TNF-Parkinson Disease. These two cases are outside the six-condition 401+4 universe and must be described as separate illustrative cases, not as members of the 405 set.
2. The 401 automatic registry matches were not individually manually verified. They support registry concordance, not efficacy, FDA approval, or a causal gene mechanism.
3. The source repository did not contain the script named `build_phase23_dataset.py`; this package reconstructs the documented 401+4 union directly from the frozen final source tables and records their hashes.
4. No saved 405-pair Wilcoxon/Holm script or result table was present in the source artifacts. Statistical-code publication remains a separate code-package task.
