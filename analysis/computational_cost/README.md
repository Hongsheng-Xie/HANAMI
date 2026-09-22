# Computational-cost logging and archived MS/DRKG records

## Reproduce the archived summaries without training

From the repository root, using Python 3.9 or newer (standard library only):

```sh
python analysis/computational_cost/rebuild_recorded_tables.py
python -m unittest discover -s analysis/computational_cost -p "test_*.py"
```

This produces `results/computational_cost/recorded_summaries.{json,md}` and
`results/computational_cost/supplementary_table2_drkg.md`. The JSON contains seed
IDs, numbers of complete seeds, individual seed totals, arithmetic means and
sample standard deviations when available. No training or GPU is required for
these commands.

### Relationship to the manuscript tables

- **Supplementary Table 2 (DRKG): reproduced.** All four recorded metrics match
  the table at its reported precision. Source
  `data/computational_cost/drkg_seed1_full_20260912_190355` contains the complete
  seven-task run for every method at seed 1. It is not a ten-seed average.
- **Supplementary Table 1 (MS): provenance reconciliation remains open.** The
  available original ten-seed archive includes timing stalls. The later
  `ms_ten_seeds_verified_20260914` archive is incomplete (34 complete model-seed
  jobs), and the separate HANAMI seed-30 attempt completed only two tasks.
  No saved final selection/replacement manifest was found that reproduces the
  manuscript's 16.06-minute HANAMI aggregate and all other Table 1 values.
  Accordingly, these files do not claim to reconstruct a reviewed nine-seed
  HANAMI/ten-seed-baseline table. The raw ten-seed summary is not a replacement
  for the manuscript table.
- The older `cost_timing_verification_20260911` audit explicitly replaces one
  stalled TriMoGCL task with its complete repeat and reproduces the historical
  **seed-1** values 12.96 minutes and 57.14 ms. This explains those numbers, but
  does not establish them as ten-seed means. The original and replacement tasks
  are both retained in `timing_audit.json`.

The rebuilding script keeps every run separate. It makes no new timing-quality
exclusions and does not select faster runs, fill missing seeds, or infer that an
absence of automatically flagged calls proves a normal run. A historical folder
name containing `verified` is not an assertion that its entire planned run was
completed or independently reviewed.

### Archive contents and provenance

Nine historical record directories are included under `data/computational_cost`.
They preserve event-level measurements, completed-task summaries, configuration,
source and feature hashes, environment details, success markers and timing flags.
Smoke tests, ten-epoch pilots, console logs, checkpoints and model predictions
are not part of this cost archive. Partial follow-up attempts are retained for
audit and do not contribute incomplete jobs to averages.

`import_records.py` created these publication copies by JSON reserialization and
replacement of personal absolute path roots with `<HANAMI_ROOT>`,
`<BASELINE_ROOT>`, `<WORK_ROOT>` and `<USER_ROOT>`. All numeric measurements and
source/data hashes are unchanged. Each `archive_inventory.json` records original
and published SHA256 values. The rebuild command verifies every published hash
before aggregation. Placeholder paths in historical provenance are descriptive,
not runtime input paths.

The recorded DRKG environment was Python 3.13.5, PyTorch 2.7.0+cu126, CUDA 12.6,
PyG 2.6.1 and scikit-learn 1.6.1 on an NVIDIA GeForce RTX 4060 Laptop GPU, with 24
logical CPU processors and 31.71 GiB system RAM. Per-job provenance preserves
the actual environment and hashes; timing is hardware- and workload-dependent.

## Run new measurements

`benchmark_ms.py` adds measurement hooks to the existing five model entry points at runtime. It does not edit their architectures, loss functions, graph construction, optimizers, or feature values. `run_suite.py` executes them serially on the same machine and writes `measurements.json`, `summary.md`, per-model logs, and live `status.json`.

### Measurement scope

- One fixed seed (1) is the initial cost benchmark. All seven binary motif tasks are the default. This is not a ten-seed performance experiment or a replacement for Figures 2–5.
- Neural classifiers retain 150 epochs and their entry-point defaults. RF retains 100 trees, fixed forest seed 42, and its original CPU parallelism. Its identical repeated fits are reduced to one fit per task; an RF fit is not a neural-network epoch.
- HANAMI uses the repository's `dise_All.pth`, `drug_All.pth`, and `gene_All.pth`. Baselines use their existing MS `*_feat.pth` files. The feature widths and hashes are recorded. This measures the implemented pipelines, not an equal-feature encoder ablation.
- Node2Vec's native PyG embedding training is measured separately and must be included when reporting N2V-MLP training cost. RF feature extraction is separately timed. Offline creation of pretrained input features is excluded, not assumed free.
- Training time is the sum of synchronized training calls. Evaluation time includes validation, any validation-triggered test calls, and metric calculation. Task wall time additionally includes model construction and graph preparation. Data loading and split generation are logged separately.
- Inference uses the final trained model, one untimed warm-up and 20 timed repetitions on the entire held-out test set. It includes model encoding, classification, softmax and transfer of predictions to CPU, but excludes feature preparation and metric calculation. Test size is recorded; compare throughput as well as latency. RF uses `predict_proba` on prepared test features.
- Peak process RAM is sampled every 50 ms and may miss short spikes. Peak allocated and reserved GPU memory are PyTorch measurements, not total device usage. Task memory includes resident data and preprocessing tensors. RF data stays on CPU because the original NumPy feature extraction cannot accept CUDA tensors; its forest and feature computations are otherwise unchanged.
- Loaded feature tensors are detached from serialization-time autograd history; their values are unchanged and they remain fixed model inputs.

The original entry points do not implement identical graph handling. In particular, HANAMI builds a masked adjacency but continues using `data.train_graph`; the baseline graph models use the reconstructed adjacency. The logger preserves this behavior and records edge counts. Do not describe these measurements as a strictly identical-graph speed comparison or modify the graph protocol just to obtain timings.

### Ten-seed version

`run_ten_seed_suite.py` runs seeds **1, 10, 20, 30, 40, 50, 60, 70, 80, 90**, all five methods and all seven tasks: 50 model-seed jobs and 350 model-task runs. It starts a fresh worker for each model-seed job and never runs training jobs concurrently. Sample split hashes are checked across methods, and source/data hashes are checked across seeds.

```powershell
python analysis/computational_cost/run_ten_seed_suite.py --baseline-root "PATH/TO/TriMoGCL-main" --output "results/computational_cost/ms_ten_seeds_20260911"
```

`status.json` is updated every ten seconds with the active seed, method, task, epoch progress, and completion counts. `summary.md` reports mean and sample SD across completed seeds. Training and inference times are first summed across the seven tasks within each seed; memory is first reduced to the maximum across tasks within each seed. A partial summary is not a final ten-seed result.

Timing calls longer than both 30 seconds and ten times the typical epoch duration are flagged for review. Their measurements are retained, not silently discarded, truncated, or substituted. Such runs end with `complete_pending_timing_review` rather than an unqualified completion. Keep the computer awake and plugged in and avoid other substantial CPU/GPU workloads. The script does not change power settings.

Use `--resume` only with the same configuration. Fully successful jobs are skipped; an incomplete attempt is retained and requires review before retrying. Existing single-seed results and manuscript text are not overwritten.

### DRKG computational-cost benchmark

The same instrumentation also accepts `--dataset drkg`. A full single-seed run
uses all seven tasks and all five existing implementations, with unchanged model
and training settings. HANAMI loads `data/drkg/{dise,drug,gene}_All.pth`;
the baselines load their existing DRKG `*_feat.pth` inputs. Input hashes,
prepared node/edge counts, per-task graph sizes and matched split hashes are logged.

```powershell
python analysis/computational_cost/run_ten_seed_suite.py --baseline-root "PATH/TO/TriMoGCL-main" --dataset drkg --seeds 1 --method-order HANAMI TriMoGCL RF TriNet N2V-MLP --output "results/computational_cost/drkg_seed1_NEW_RUN"
```

`cost_table.md` reports summed training time and maximum allocated GPU memory
across seven tasks. Preparation, evaluation and offline input-feature generation
remain excluded from training time, as in the MS benchmark. This is a practical
cost measurement on the processed DRKG benchmark, not a graph-size scaling
experiment or proof of superior scalability. A separate two-epoch smoke test
must not be included in reported full-run costs.

Use a fresh output directory for each new run. Resuming requires the same
worker hash and configuration; completed earlier MS outputs are retained.

Use a CUDA-enabled Python environment with PyTorch, PyG, a native Node2Vec backend, NumPy, pandas, scikit-learn, NetworkX and psutil. Pass the local TriMoGCL source directory explicitly.

The external baseline directory must contain `main_tri_binary.py`,
`main_tri_binary_TriNet.py`, `main_tri_binary_N2V_MLP.py` and
`main_tri_binary_RF.py`, their imported modules, and `data/ms` and `data/drkg`
inputs. Baseline gene, drug and disease `*_feat.pth` files must be present. The
recorded hashes, rather than a moving branch name, identify the implementations
used in the archived measurements. The archived timing values are not guaranteed
for modified source or feature files. Reproducing the archived tables requires
none of these external training dependencies.

```powershell
python analysis/computational_cost/run_suite.py --baseline-root "PATH/TO/TriMoGCL-main" --output "results/computational_cost/ms_seed1" --seed 1
```

For a two-epoch instrumentation check, add `--epochs 2 --task first` and use a separate output directory. Do not report smoke-test times as full-training costs. Each run needs a fresh output directory; keep prior measurements intact. Source/data hashes, environment information, and settings accompany every model run. No manuscript or existing figure files are changed by this benchmark.
