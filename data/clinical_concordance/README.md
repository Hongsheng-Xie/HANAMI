# Clinical concordance inputs

Figure 5 uses 1,630 motifs from 785 drug–disease pairs and 109 disease labels. Input paths, seed order and checksums are recorded in [current/manifest.json](current/manifest.json).

| Input | Purpose |
|---|---|
| `gene_star_candidates.csv` | Pool of 46,704 MS gene-star candidates |
| `gene_star_scores_10seeds.npz` | Four baseline score arrays |
| `current/hanami_scores/` | Ten HANAMI score arrays |
| `current/cohort_metadata.tsv` | Cohort identifiers and supporting trial records |
| `current/validated_gene_star_motifs_1630_mean10.tsv` | Mean rank percentiles across ten seeds |
| `current/figure5_cases_5.csv` | Five case-study motifs and clinical references |

See the [workflow instructions](../../analysis/clinical_concordance/) to generate Figure 5 and the [provenance note](current/PROVENANCE.md) for source details and historical materials.
