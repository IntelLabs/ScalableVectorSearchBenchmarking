# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Benchmark concurrent search and update on a DynamicVamana index.

Phase 1 (baseline): n_workers threads search with batch_size=1.
Phase 2 (concurrent): (n_workers - n_update) search + n_update update.
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import svs
from tqdm import tqdm

from . import consts, utils
from .build import build_dynamic
from .loader import create_loader
from .search import STR_TO_CALIBRATE_SEARCH_BUFFER

logger = logging.getLogger(__file__)


# --- Workers ---

def _search_worker(index, query, count, barrier, results_out):
    """Search all queries one at a time (batch_size=1)."""
    query_size = query.shape[0]
    barrier.wait()
    total_time = 0
    result_batches = []
    batch_times = []
    errors = 0
    for q_idx in range(query_size):
        start = time.perf_counter()
        try:
            result, _ = index.search(query[q_idx : q_idx + 1], count)
        except Exception:
            errors += 1
            continue
        elapsed = time.perf_counter() - start
        total_time += elapsed
        batch_times.append(elapsed)
        result_batches.append(result)
    results_out["results"] = (
        np.concatenate(result_batches)
        if result_batches
        else np.empty((0, count), np.int32)
    )
    successful = query_size - errors
    results_out["qps"] = successful / total_time if total_time > 0 else 0
    results_out["p95"] = (
        np.percentile(batch_times, 95) if batch_times else 0
    )
    results_out["errors"] = errors


def _update_worker(
    index, update_vectors, update_ids, batch_size, num_vectors_delete,
    barrier, results_out, seed,
):
    """Add batches from update pool, delete, consolidate — like build.py."""
    try:
        rng = np.random.default_rng(seed)
        num_update = len(update_ids)
        num_batches = int(np.ceil(num_update / batch_size))
        add_times = []
        delete_times = []
        barrier.wait()
        for batch_idx in range(num_batches):
            init_batch = batch_idx * batch_size
            end_batch = min(init_batch + batch_size, num_update)
            batch_ids = update_ids[init_batch:end_batch]

            start = time.perf_counter()
            for v_idx in range(init_batch, end_batch):
                index.add(
                    update_vectors[v_idx : v_idx + 1],
                    update_ids[v_idx : v_idx + 1],
                )
            add_times.append(time.perf_counter() - start)

            # Delete from previous batch before adding new one
            if num_vectors_delete > 0:
                n_del = min(num_vectors_delete, len(batch_ids))
                ids_to_delete = batch_ids[:n_del]
                start = time.perf_counter()
                index.delete(ids_to_delete)
                index.consolidate()
                delete_times.append(time.perf_counter() - start)

        results_out["add_times"] = add_times
        results_out["delete_times"] = delete_times
        results_out["num_batches"] = len(add_times)
    except Exception as e:
        results_out["error"] = e


# --- Orchestration helpers ---

def _run_search_only(label, index, query, count, n_threads):
    """Run n_threads search workers, return results list."""
    barrier = threading.Barrier(n_threads)
    results = [{} for _ in range(n_threads)]
    threads = [
        threading.Thread(
            target=_search_worker,
            args=(index, query, count, barrier, results[i]),
            name=f"{label}-search-{i}",
        )
        for i in range(n_threads)
    ]
    logger.info({"phase_start": label, "n_threads": n_threads})
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_time = time.perf_counter() - wall_start
    logger.info({"phase_end": label, "wall_time": wall_time})
    return results, wall_time


def _run_concurrent(
    label, index, query, count,
    update_vectors, update_ids, batch_size, num_vectors_delete,
    n_search, n_update, seed,
):
    """Run search + update workers concurrently, both run their full workload."""
    n_total = n_search + n_update
    barrier = threading.Barrier(n_total)
    search_results = [{} for _ in range(n_search)]
    update_results = [{} for _ in range(n_update)]

    # Partition update pool across n_update threads
    pool_size = len(update_ids)
    chunk = pool_size // n_update

    search_threads = [
        threading.Thread(
            target=_search_worker,
            args=(index, query, count, barrier, search_results[i]),
            name=f"{label}-search-{i}",
        )
        for i in range(n_search)
    ]
    update_threads = []
    for i in range(n_update):
        start_idx = i * chunk
        end_idx = pool_size if i == n_update - 1 else start_idx + chunk
        t = threading.Thread(
            target=_update_worker,
            args=(
                index,
                update_vectors[start_idx:end_idx],
                update_ids[start_idx:end_idx],
                batch_size, num_vectors_delete,
                barrier, update_results[i], seed + i,
            ),
            name=f"{label}-update-{i}",
        )
        update_threads.append(t)

    all_threads = search_threads + update_threads
    logger.info({
        "phase_start": label, "n_search": n_search, "n_update": n_update,
    })
    wall_start = time.perf_counter()
    for t in all_threads:
        t.start()
    for t in all_threads:
        t.join()
    wall_time = time.perf_counter() - wall_start
    logger.info({"phase_end": label, "wall_time": wall_time})
    return search_results, update_results, wall_time


def _log_search(label, results_list, ground_truth, count):
    qps_list = [r["qps"] for r in results_list if "qps" in r]
    p95_list = [r["p95"] for r in results_list if "p95" in r]
    total_errors = sum(r.get("errors", 0) for r in results_list)
    if total_errors > 0:
        logger.info({f"{label}_search_errors": total_errors})
    if qps_list:
        qps_arr = np.array(qps_list)
        p95_arr = np.array(p95_list)
        logger.info({
            f"{label}_search_results": {
                "qps_per_thread": qps_list,
                "qps_mean": float(np.mean(qps_arr)),
                "qps_rsd": float(
                    np.std(qps_arr, ddof=min(1, len(qps_arr) - 1))
                    / np.mean(qps_arr)
                ),
                "p95_mean": float(np.mean(p95_arr)),
                "p95_rsd": float(
                    np.std(p95_arr, ddof=min(1, len(p95_arr) - 1))
                    / np.mean(p95_arr)
                ),
            }
        })
    if ground_truth is not None:
        last = results_list[-1].get("results")
        if last is not None and len(last) > 0:
            logger.info({
                f"{label}_recall": svs.k_recall_at(
                    ground_truth, last, count, count
                )
            })


def _log_update(label, results_list):
    for r in results_list:
        if "error" in r:
            logger.error({f"{label}_error": str(r["error"])})
    all_add = []
    all_del = []
    for r in results_list:
        all_add.extend(r.get("add_times", []))
        all_del.extend(r.get("delete_times", []))
    if all_add:
        a = np.array(all_add)
        logger.info({f"{label}_add_time": {
            "mean": float(np.mean(a)),
            "rsd": float(np.std(a, ddof=min(1, len(a)-1)) / np.mean(a)),
            "n_cycles": len(a),
        }})
    if all_del:
        d = np.array(all_del)
        logger.info({f"{label}_delete_time": {
            "mean": float(np.mean(d)),
            "rsd": float(np.std(d, ddof=min(1, len(d)-1)) / np.mean(d)),
            "n_cycles": len(d),
        }})


# --- Main benchmark ---

def concurrent_benchmark(
    *,
    # build.py params
    vecs_path: Path,
    svs_type: str,
    distance: svs.DistanceType,
    idx_dir: Path | None = None,
    num_vectors: int | None = None,
    graph_max_degree: int = 64,
    window_size: int = 200,
    prune_to: int | None = None,
    max_candidate_pool_size: int = 750,
    alpha: float | None = None,
    max_threads: int = 1,
    max_threads_init: int | None = None,
    batch_size: int = 10000,
    num_vectors_delete: int = 0,
    num_vectors_init: int | None = None,
    proportion_vectors_init: float | None = None,
    shuffle: bool = False,
    seed: int = 42,
    tmp_dir: Path = Path("/dev/shm"),
    leanvec_dims: int | None = None,
    # search.py params
    query_path: Path,
    ground_truth_path: Path | None = None,
    query_type: svs.DataType = svs.DataType.float32,
    count: int = 10,
    recall: float = 0.9,
    leanvec_alignment: int = 32,
    lvq_strategy: svs.LVQStrategy | None = None,
    train_prefetchers: bool = True,
    search_buffer_optimization: svs.VamanaSearchBufferOptimization = svs.VamanaSearchBufferOptimization.All,
    # concurrent params
    n_workers: int = 1,
    n_update: int = 0,
) -> None:
    logger.info({"concurrent_args": locals()})
    logger.info(utils.read_system_config())

    if n_update > n_workers:
        raise ValueError("n_update must be <= n_workers")
    n_search = n_workers - n_update

    # --- Step 1: Load data, split in half ---
    vectors = svs.read_vecs(str(vecs_path))
    if num_vectors is not None:
        vectors = vectors[:num_vectors]
    total = vectors.shape[0]
    half = total // 2
    query = svs.read_vecs(str(query_path))

    print(f"max_threads = {max_threads}")

    # --- Step 2: Build dynamic index with first half only ---
    index, name, ingest_time, delete_time = build_dynamic(
        vecs_path=vecs_path,
        svs_type=svs_type,
        distance=distance,
        idx_dir=idx_dir,
        num_vectors=half,
        graph_max_degree=graph_max_degree,
        window_size=window_size,
        prune_to=prune_to,
        max_candidate_pool_size=max_candidate_pool_size,
        alpha=alpha,
        max_threads=max_threads,
        max_threads_init=max_threads_init,
        batch_size=batch_size,
        num_vectors_delete=0,
        shuffle=shuffle,
        seed=seed,
        tmp_dir=tmp_dir,
        leanvec_dims=leanvec_dims,
    )
    logger.info({"build_complete": {"index_size": len(index.all_ids())}})

    # Second half is the update pool
    update_vectors = vectors[half:]
    update_ids = np.arange(half, total, dtype=np.uint64)

    # --- Step 3: Calibrate search (like search.py) ---
    ground_truth = (
        svs.read_vecs(str(ground_truth_path))
        if ground_truth_path is not None
        else None
    )

    index.num_threads = min(max_threads, batch_size)
    if ground_truth is not None:
        calibration_parameters = svs.VamanaCalibrationParameters()
        calibration_parameters.search_buffer_optimization = (
            search_buffer_optimization
        )
        calibration_parameters.train_prefetchers = train_prefetchers
        index.experimental_calibrate(
            query, ground_truth, count, recall, calibration_parameters,
        )
    else:
        index.search(query, count)
    logger.info({
        "calibration_results": {
            "search_window_size": index.search_parameters.buffer_config.search_window_size,
            "search_buffer_capacity": index.search_parameters.buffer_config.search_buffer_capacity,
            "prefetch_lookahead": index.search_parameters.prefetch_lookahead,
            "prefetch_step": index.search_parameters.prefetch_step,
        }
    })

    # --- Step 4: Multithreaded benchmark ---
    # Phase 1: Baseline — all n_workers search
    index.num_threads = 1
    search_results, _ = _run_search_only(
        "baseline", index, query, count, n_workers,
    )
    _log_search("baseline", search_results, ground_truth, count)

    # Phase 2: Concurrent — n_search search + n_update update
    if n_update > 0:
        search_results, update_results, _ = _run_concurrent(
            "concurrent", index, query, count,
            update_vectors, update_ids, batch_size, num_vectors_delete,
            n_search, n_update, seed,
        )
        _log_search("concurrent", search_results, None, count)
        _log_update("concurrent", update_results)


# --- CLI ---

def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_arguments(parser)
    # build.py args
    parser.add_argument(
        "--vecs_file", help="Vectors *vecs file", type=Path, required=True
    )
    parser.add_argument(
        "--batch_size", help="Batch size", default=10000, type=int
    )
    parser.add_argument("--idx_dir", help="Index dir", type=Path)
    parser.add_argument("--num_vectors", help="Number of vectors", type=int)
    parser.add_argument("--graph_max_degree", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--prune_to", type=int)
    parser.add_argument("--max_candidate_pool_size", type=int, default=750)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--distance",
        choices=tuple(consts.STR_TO_DISTANCE.keys()),
        default="mip",
    )
    parser.add_argument(
        "--max_threads_init", type=int,
        help="Maximum threads for building the initial graph",
    )
    parser.add_argument(
        "--num_vectors_delete", type=int, default=0,
        help="Vectors to delete per update cycle",
    )
    parser.add_argument("--num_vectors_init", type=int)
    parser.add_argument("--proportion_vectors_init", type=float)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--tmp_dir", type=Path, default="/dev/shm",
    )
    parser.add_argument("--leanvec_dims", type=int)
    # search.py args
    parser.add_argument(
        "--query_file", type=Path, required=True,
    )
    parser.add_argument("--ground_truth_file", type=Path)
    parser.add_argument(
        "--query_type", choices=consts.STR_TO_DATA_TYPE.keys(),
        default="float32",
    )
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--recall", type=float, default=0.9)
    parser.add_argument("--leanvec_alignment", type=int, default=32)
    parser.add_argument(
        "--lvq_strategy",
        choices=tuple(consts.STR_TO_LVQ_STRATEGY.keys()),
        default="auto",
    )
    parser.add_argument("--no_calibrate_prefetchers", action="store_true")
    parser.add_argument(
        "--calibrate_search_buffer",
        choices=STR_TO_CALIBRATE_SEARCH_BUFFER.keys(),
        default="all",
    )
    # concurrent args
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="Total concurrent worker threads",
    )
    parser.add_argument(
        "--n_update", type=int, default=0,
        help="Of n_workers, how many do updates in phase 2",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _read_args(argv)
    log_file = utils.configure_logger(
        logger, args.log_dir if args.log_dir is not None else args.out_dir
    )
    print("Logging to", log_file, sep="\n")
    logger.info({"argv": argv if argv else sys.argv})
    utils.check_uncommitted_and_log_version(logger, args.uncommitted)
    args.out_dir.mkdir(exist_ok=True)
    concurrent_benchmark(
        vecs_path=args.vecs_file,
        svs_type=args.svs_type,
        distance=consts.STR_TO_DISTANCE[args.distance],
        idx_dir=args.idx_dir,
        num_vectors=args.num_vectors,
        graph_max_degree=args.graph_max_degree,
        window_size=args.window_size,
        prune_to=args.prune_to,
        max_candidate_pool_size=args.max_candidate_pool_size,
        alpha=args.alpha,
        max_threads=args.max_threads,
        max_threads_init=args.max_threads_init,
        batch_size=args.batch_size,
        num_vectors_delete=args.num_vectors_delete,
        num_vectors_init=args.num_vectors_init,
        proportion_vectors_init=args.proportion_vectors_init,
        shuffle=args.shuffle,
        seed=args.seed,
        tmp_dir=args.tmp_dir,
        leanvec_dims=args.leanvec_dims,
        query_path=args.query_file,
        ground_truth_path=args.ground_truth_file,
        query_type=consts.STR_TO_DATA_TYPE[args.query_type],
        count=args.k,
        recall=args.recall,
        leanvec_alignment=args.leanvec_alignment,
        lvq_strategy=consts.STR_TO_LVQ_STRATEGY.get(args.lvq_strategy, None),
        train_prefetchers=not args.no_calibrate_prefetchers,
        search_buffer_optimization=STR_TO_CALIBRATE_SEARCH_BUFFER[
            args.calibrate_search_buffer
        ],
        n_workers=args.n_workers,
        n_update=args.n_update,
    )


if __name__ == "__main__":
    main()
