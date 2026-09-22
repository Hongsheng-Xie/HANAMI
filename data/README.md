# Feature matrices and transfer data

The repository supplies the precomputed feature matrices used by the training
scripts. Install Git LFS before cloning, or run the following in an existing clone
to fetch the binary files under `data/drkg/`:

```bash
git lfs install
git lfs pull
```

## Full HANAMI features

`main.py` loads `dise_All.pth`, `drug_All.pth`, and `gene_All.pth` from the selected
dataset directory. These are distinct from the baseline `*_feat.pth` inputs.

| Dataset | `dise_All.pth` | `drug_All.pth` | `gene_All.pth` |
| --- | --- | --- | --- |
| MS | 694 × 1792 | 1272 × 1452 | 4519 × 1024 |
| DRKG | 2157 × 1792 | 2908 × 1452 | 9809 × 1024 |

The within-dataset loader zero-pads drug and gene features to the disease width
of 1792. `data/ms/dise_Bio.pth` is a historical filename for the same bytes as
`data/ms/dise_All.pth`; use `dise_All.pth` for HANAMI.

## Baseline gene features

`ms/gene_feat.pth` and `drkg/gene_feat.pth` are copied unchanged from the original
TriMoGCL experiment data directories, completing the baseline `*_feat.pth` sets.
They are not reconstructed from HANAMI features. The MS file matches the input
fingerprint recorded by the original baseline scoring run.

| File | Shape | SHA256 |
| --- | --- | --- |
| `ms/gene_feat.pth` | 4519 × 1024 | `992516795fda359337a2e43bcbf4844eb20b90cf2319b8450fda2c3d573102ac` |
| `drkg/gene_feat.pth` | 9809 × 1024 | `8e845628d0a6f41a2da19245a888c24b342480495868ceb7d2ff2e1a5d567d0a` |

The MS baseline and HANAMI gene matrices have different values. The DRKG
baseline gene file happens to be byte-identical to `drkg/gene_All.pth` in this
release; its original filename is retained for baseline code.

## Rebuild the DRKG-minus-MS subsets

`transfer_embedding.py` selects rows from the supplied DRKG matrices after
excluding entity identifiers present in MS. It preserves the DRKG dictionary
order and writes the same original-index-to-subgraph-index mappings as the
published `subgraph_dise.npy`, `subgraph_drug.npy`, and `subgraph_gene.npy`.
Baseline disease similarity features are subset on both axes; all other
matrices are subset on the entity axis only.

| Subset | Disease | Drug | Gene |
| --- | --- | --- | --- |
| `DRKG_MS_*_Rev.pth` (HANAMI) | 1551 × 1792 | 1636 × 1452 | 5291 × 1024 |
| `DRKG_MS_*_Base.pth` (baseline) | 1551 × 1551 | 1636 × 1200 | 5291 × 1024 |

Verify the supplied six tensors and three mappings without changing files:

```bash
python transfer_embedding.py --check
```

Rebuild into a separate directory, or omit `--output-dir` to replace the subset
files in `data/drkg/`:

```bash
python transfer_embedding.py --output-dir results/rebuilt_transfer_features
```

`--data-root PATH` selects an alternative directory containing `ms/` and
`drkg/`. Verification compares tensor values and dtypes, not serialized file
hashes, because serialization may differ between PyTorch versions. This script
only prepares features and mappings; the preprocessing performed during transfer
training remains in `transfer_utils.py`.

## Encoder reconstruction limits

`embedding.py` is a historical feature-extraction script, not a complete command
for regenerating the supplied matrices. It references an unbundled
`gene_seq.txt`, does not save the final matrices or assemble the full drug
feature concatenation, and does not supply trained gene-compression checkpoints.
Its second ChemBERTa checkpoint is `DeepChem/ChemBERTa-77M-MLM`. Use the
precomputed matrices for the released experiments; the exact upstream encoder
reconstruction is not supplied by this release.
