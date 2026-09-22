# Provenance of the completed HANAMI scoring runs

These records describe the completed `bio_allgene_control` runs that produced
the ten frozen HANAMI score arrays used by the current Figure 5. They are
historical records, not evidence that training was rerun when this package was
prepared.

- `run_manifest.json` records the seeds, epochs, candidate-pool hash and runtime.
- `meta_seed*.json` retain each run's selected epoch, validation AUROC, feature
  names, optimizer settings and original diagnostic targets.
- `source_manifest.json` records the original script and metadata SHA256 hashes,
  the packaged source locations, and the limited portability changes.

Machine-specific paths have been sanitized. Score paths point to the frozen
packaged scores. Checkpoint and historical source basenames are provenance
labels; those historical checkpoints and auxiliary scripts are not included.
The historical `dise_Bio.pth` name is retained in these records and is
byte-identical to the packaged `dise_All.pth`. Original metrics and target lists
have not been recalculated or replaced by current Figure 5 cases.

The optional portable runner is documented in
[score_generation](../../../../analysis/clinical_concordance/score_generation/).
The frozen Figure 5 input checksums remain in [the current manifest](../manifest.json).
