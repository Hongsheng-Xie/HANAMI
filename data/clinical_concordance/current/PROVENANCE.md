# Figure 5 input provenance

The current workflow uses the completed clinical evidence audit and saved model
scores. The [manifest](manifest.json) records the input paths, identifiers and
SHA256 checksums of the saved inputs.

## Candidate pool and scores

All methods score the same 46,704 MS gene-star candidates in
[`gene_star_candidates.csv`](../gene_star_candidates.csv). Each candidate has
drug–gene and gene–disease edges, with no direct drug–disease edge in MS.

[`gene_star_scores_10seeds.npz`](../gene_star_scores_10seeds.npz) supplies the
RF, TriNet, N2V-MLP and TriMoGCL arrays. HANAMI scores come from
[`hanami_scores/`](hanami_scores/), rather than the older HANAMI array in that
archive. The ten seeds are 0, 10, 20, 30, 40, 50, 60, 70, 80 and 90.

The HANAMI run used `dise_All.pth`, `drug_All.pth` and `gene_All.pth`. Its original
disease-feature filename, `dise_Bio.pth`, is byte-identical to `dise_All.pth`.
The [score-generation source](../../../run_bio_allgene_hanami.py)
and [completed-run records](training_provenance/) document the training pipeline.

## Clinical cohort

[`cohort_metadata.tsv`](cohort_metadata.tsv) contains 1,630 motifs from 785
distinct drug–disease pairs across 109 disease labels. It combines 353 manually
reviewed pairs (965 motifs) with 432 additional strict automatic matches
(665 motifs). Automatic matches used the disease-directed Phase II/III
trial-title criterion. The `evidence_source` field distinguishes these two
routes, and `supporting_trials` preserves their trial identifiers.

The frozen cohort retains all associated motifs, without further model-score
or disease-subset filtering during reproduction. Registry matching and manual
review are upstream steps, rather than stages repeated by the figure runner.

Trial evidence establishes clinical investigation, not efficacy, FDA approval
or a causal role for the shared gene. Combination studies concern the regimen
as a whole, and an MS disease label can be broader than the trial condition.
For example, the Hypersensitivity case concerns hypersensitivity pneumonitis.
The [five-case table](figure5_cases_5.csv) records the relevant clinical context.

The [registry snapshot](../clinicaltrials_snapshot.csv.gz) retains the available
archived records whose trial identifiers occur in this cohort. It covers at
least one cited trial for 381 of the 785 pairs, not all cohort evidence.
The [snapshot metadata](../clinicaltrials_metadata.json) preserves retrieval
details, checksums and four additional review notes for current pairs.
