# Scalable Vector Search Benchmarking
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/ScalableVectorSearchBenchmarking/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/ScalableVectorSearchBenchmarking)
![GitHub License](https://img.shields.io/github/license/IntelLabs/ScalableVectorSearchBenchmarking)
![python-support](https://img.shields.io/badge/Python-3.12-3?logo=python)

Scalable Vector Search Benchmarking enables the benchmarking or evaluation of the [Scalable Vector Search](https://github.com/intel/ScalableVectorSearch) library.

## Installation

Requires Python >= 3.12.

```sh
python -m pip install \
    git+https://github.com/IntelLabs/ScalableVectorSearchBenchmarking
```

## Usage

### Building an index

```sh
python -m svsbench.build \
    --vecs_file /path/to/vectors.fvecs \
    --svs_type leanvec4x8 --leanvec_dims -4 \
    --proportion_vectors_init 0.5 --batch_size 10000
```

### Computing the ground truth

For the query vectors used in performance measurements:
```sh
python -m svsbench.generate_ground_truth \
    --vecs_file vectors.fvecs \
    --query_file query_vectors.fvecs
```

For the query vectors used in the calibration of search parameters:
```sh
python -m svsbench.generate_ground_truth \
    --vecs_file vectors.fvecs \
    --query_file calibration_query_vectors.fvecs
```

### Searching

Calibrating the search parameters for a given recall and then searching:
```sh
python -m svsbench.search \
    --idx_dir /path/to/index_dir_from_build \
    --svs_type leanvec4x8 \
    --query_file /path/to/query_vectors.fvecs \
    --ground_truth_file /path/to/ground_truth.ivecs \
    -k 5 \
    --recall 0.95 \
    --calibration_query_file /path/to/calibration_query_vectors.fvecs \
    --calibration_ground_truth_file /path/to/calibration_ground_truth.ivecs
```

Searching using specified search parameters:
```sh
python -m svsbench.search \
    --idx_dir /path/to/index_dir_from_build \
    --svs_type leanvec4x8 \
    --query_file /path/to/query_vectors.fvecs \
    --ground_truth_file /path/to/ground_truth.ivecs \
    -k 5 \
    --batch_size 1 \
    --search_window_size 14 \
    --search_buffer_capacity 34 \
    --prefetch_lookahead 1 \
    --prefetch_step 0 \
    --batch_size 10000 \
    --search_window_size 15 \
    --search_buffer_capacity 36 \
    --prefetch_lookahead 10 \
    --prefetch_step 4
```
