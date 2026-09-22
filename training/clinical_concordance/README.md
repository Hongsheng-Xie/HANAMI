# Optional HANAMI score generation

This directory packages the specific `bio_allgene_control` runner and data loader
identified by the completed ten-seed run behind the current Figure 5 HANAMI
scores. It is optional: the normal Figure 5 workflow creates the figure and
statistics directly from the prepared ranking tables and does not retrain a model.

## Source and scope

The runner trains the clique binary classifier for 150 epochs for seeds
`0,10,20,30,40,50,60,70,80,90`. It selects the earliest strict maximum validation
AUROC and scores the fixed 46,704-candidate pool with the class-1 logit margin
`z1 - z0`. Model initialization, minibatch order, cumulative edge masking,
loss, sampling and checkpoint selection are retained from the original runner.

Only paths, dependency-path setup and the disease-file alias were made portable;
trailing whitespace was also removed without changing the executable code.
`dise_All.pth` replaces the byte-identical historical `dise_Bio.pth`.
The generated metadata points to the packaged main runner and labels the unused
historical extractor. Original source hashes and the complete list of packaging
changes are recorded in
[source_manifest.json](../../data/clinical_concordance/current/training_provenance/source_manifest.json).

The original six diagnostic targets are preserved in the runner and historical
metadata. They are not the five current Figure 5 cases and are not used to select
the current clinical cohort or calculate the current figure. Current Figure 5
aggregation remains in [the Figure 5 workflow](../../analysis/clinical_concordance/).

## Dependencies

The scorer uses the repository's `base_gcn.py` and `create_data.py`, the paired
`utils_our_bio_allgene.py` loader, and the supplied MS matrices and motif arrays.
It requires CUDA-enabled PyTorch, PyTorch Geometric, NumPy, scikit-learn and tqdm.
There is no TriMoGCL source-code or baseline-model dependency. The old local
TriMoGCL virtual-environment path was only used to locate installed PyG packages
and has been removed.

The recorded run used PyTorch `2.7.0+cu126`, CUDA 12.6 and an NVIDIA GeForce
RTX 4060 Laptop GPU. The historical metadata does not contain a complete package
lock. The repository's general requirements are not an exact environment lock
for this historical run.

## Optional training command

After installing the dependencies, run from the repository root with a
CUDA-capable environment:

```sh
python training/clinical_concordance/run_bio_allgene_hanami.py --output-dir results/clinical_concordance/score_generation --seeds 0,10,20,30,40,50,60,70,80,90 --epochs 150 --score-batch 2048
```

Use a separate output directory for new experiments. The command writes scores,
checkpoints, per-seed metadata and historical diagnostic summaries there; it does
not replace the frozen scores under `data/clinical_concordance/current/`.
Completed seeds are skipped unless `--force` is supplied.

The source package was validated without executing training. It does not claim
bitwise regeneration of the released scores across environments. Keep the frozen
scores as the source records underlying the prepared ranking tables.

## Completed-run provenance

[training_provenance](../../data/clinical_concordance/current/training_provenance/)
contains the sanitized original run manifest and ten per-seed records. Machine
paths were replaced by repository-relative paths or historical basenames;
original file hashes, metrics, seeds, feature names and diagnostic targets are
preserved. Historical checkpoints are not included. The separate
`current/manifest.json` continues to define the checksummed frozen Figure 5
inputs and has not been replaced by these training records.

The optional source files were moved from the analysis directory to
`training/clinical_concordance/` for organization. Only the repository-root depth
and documentation paths changed during this move; no training was executed.
