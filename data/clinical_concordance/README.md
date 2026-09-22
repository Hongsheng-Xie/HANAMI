# Clinical concordance inputs

The current Figure 5 uses the complete cohort of 1,630 motifs from 785
drug–disease pairs and 109 disease labels. Its canonical inputs are under
[current](current/), with relative paths and SHA256 checks in
[current/manifest.json](current/manifest.json).

| Input | Purpose |
|---|---|
| `gene_star_candidates.csv` | Fixed 46,704-candidate MS gene-star pool and topology fields |
| `gene_star_scores_10seeds.npz` | Four baseline score arrays; its old HANAMI array is not used |
| `current/hanami_scores/` | Current ten HANAMI score arrays, seeds 0,10,…,90 |
| `current/cohort_metadata.tsv` | Frozen strict-plus-manual cohort, pair IDs and trial evidence |
| `current/validated_gene_star_motifs_1630_mean10.tsv` | Released arithmetic-mean percentiles, independently reconstructed by the builder |
| `current/figure5_cases_5.csv` | Current five cases and their qualified clinical evidence |

This is a reproduction from saved scores and the completed clinical review,
not a new clinical-trial audit. See the [workflow documentation](../../analysis/clinical_concordance/)
for aggregation, tests, evidence limits and commands.

## Historical materials

`clinically_documented_405.csv`, `figure5_cases_7.csv`,
`hanami_consensus_gene_stars.csv`, `frozen_inputs/`, `data_dictionary.csv`
and `PROVENANCE.md` describe the superseded 405-pair/consensus analysis.
They do not define the current cohort, cases or HANAMI scores.
The registry snapshot and its metadata are retained as historical provenance.
Historical scripts are under
[analysis/clinical_concordance/legacy_405](../../analysis/clinical_concordance/legacy_405/).
These inputs are retained for audit, not substituted for the current release.
