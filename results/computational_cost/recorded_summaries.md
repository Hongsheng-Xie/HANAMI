# Reconstructed computational-cost records

Each archive is reported separately. Timing flags and incomplete runs are retained; no new exclusions or substitutions are made.

Training includes Node2Vec fitting but excludes offline input-feature generation, preparation, and evaluation. Within each seed, times are summed across seven tasks and memory is the maximum across tasks. Means below average complete model-seed jobs. RF fitting/inference is CPU-only.

## cost_timing_verification_20260911

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|
| RF | 1 | 3.84 | 258.65 | 3.91 | CPU only | 0 |
| TriNet | 1 | 0.43 | 5.38 | 1.28 | 0.37 | 0 |
| N2V-MLP | 1 | 3.87 | 16.84 | 1.69 | 2.53 | 0 |
| TriMoGCL | 1 | 12.96 | 57.14 | 1.65 | 1.47 | 0 |
| HANAMI | 1 | 16.98 | 71.64 | 1.71 | 1.19 | 0 |

## drkg_seed1_full_20260912_190355

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|
| RF | 1 | 570.38 | 1766.22 | 15.90 | CPU only | 0 |
| TriNet | 1 | 0.82 | 16.68 | 1.98 | 1.81 | 0 |
| N2V-MLP | 1 | 9.40 | 49.06 | 2.39 | 3.56 | 0 |
| TriMoGCL | 1 | 85.66 | 157.80 | 3.59 | 7.21 | 0 |
| HANAMI | 1 | 265.93 | 389.18 | 4.69 | 4.66 | 0 |

## ms_hanami_seed30_clean_20260914

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|

## ms_rerun_30_60_70_80_90_20260914

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|
| RF | 1 | 4.66 | 262.98 | 3.67 | CPU only | 0 |
| TriNet | 1 | 0.41 | 4.91 | 1.28 | 0.37 | 0 |
| N2V-MLP | 1 | 3.74 | 15.02 | 1.67 | 2.53 | 0 |
| TriMoGCL | 1 | 66.43 | 225.75 | 1.66 | 1.47 | 0 |

## ms_seed10_fresh_20260912_112757

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|
| RF | 1 | 3.59 | 202.02 | 3.68 | CPU only | 0 |
| TriNet | 1 | 0.32 | 3.82 | 1.29 | 0.37 | 0 |
| N2V-MLP | 1 | 3.55 | 14.26 | 1.71 | 2.53 | 0 |
| TriMoGCL | 1 | 10.74 | 37.34 | 1.65 | 1.47 | 0 |
| HANAMI | 1 | 14.71 | 69.59 | 1.69 | 1.19 | 0 |

## ms_seed1_full_20260911

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|
| RF | 1 | 3.84 | 258.65 | 3.91 | CPU only | 0 |
| TriNet | 1 | 0.43 | 5.38 | 1.28 | 0.37 | 0 |
| N2V-MLP | 1 | 3.87 | 16.84 | 1.69 | 2.53 | 0 |
| TriMoGCL | 1 | 23.37 | 55.12 | 1.65 | 1.47 | 0 |
| HANAMI | 1 | 16.98 | 71.64 | 1.71 | 1.19 | 0 |

## ms_seed20_fresh_20260912_152011

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|
| RF | 1 | 4.06 | 221.01 | 3.96 | CPU only | 0 |
| TriNet | 1 | 0.43 | 4.69 | 1.28 | 0.37 | 0 |
| N2V-MLP | 1 | 3.66 | 14.29 | 1.67 | 2.53 | 0 |
| TriMoGCL | 1 | 12.26 | 46.91 | 1.66 | 1.47 | 0 |
| HANAMI | 1 | 119.62 | 602.98 | 1.69 | 1.19 | 0 |

## ms_ten_seeds_20260911

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|
| RF | 10 | 4.25 | 240.61 | 3.85 | CPU only | 0 |
| TriNet | 10 | 0.47 | 5.69 | 1.29 | 0.37 | 0 |
| N2V-MLP | 10 | 3.77 | 14.72 | 1.69 | 2.53 | 0 |
| TriMoGCL | 10 | 36.53 | 99.52 | 1.66 | 1.47 | 1 |
| HANAMI | 10 | 37.35 | 172.92 | 1.69 | 1.19 | 1 |

## ms_ten_seeds_verified_20260914

| Method | Complete seeds | Training min | Inference ms | Peak RAM GiB | Peak GPU GiB | Flagged calls |
|---|---:|---:|---:|---:|---:|---:|
| RF | 7 | 4.30 | 236.16 | 3.79 | CPU only | 0 |
| TriNet | 7 | 0.43 | 5.53 | 1.29 | 0.37 | 0 |
| N2V-MLP | 7 | 3.71 | 14.86 | 1.68 | 2.53 | 0 |
| TriMoGCL | 7 | 27.96 | 103.64 | 1.66 | 1.47 | 0 |
| HANAMI | 6 | 17.59 | 74.07 | 1.70 | 1.19 | 0 |
