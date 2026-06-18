# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Benchmark mixed workload: each thread does search, add, delete in random order.

Each thread processes a stream of operations (search/add/delete/consolidate)
drawn randomly according to --add_ratio and --delete_ratio, with --n_consolidate
consolidate calls inserted at random positions. Per-operation timings are
collected and reported as QPS for search/add/delete and mean time for consolidate.
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
from .search import STR_TO_CALIBRATE_SEARCH_BUFFER

logger = logging.getLogger(__file__)

# Operation types
OP_SEARCH = 0
OP_ADD = 1
OP_DELETE = 2
OP_CONSOLIDATE = 3


def _mixed_worker(
    index, query, count, vectors, add_ids, delete_ids,
    ops, barrier, results_out,
):
    """Execute a mixed stream of search/add/delete/consolidate operations."""
    query_size = query.shape[0]

    search_times = []
    add_times = []
    delete_times = []
    consolidate_times = []
    errors = 0
    error_details = []

    q_idx = 0
    add_idx = 0
    del_idx = 0

    barrier.wait()
    wall_start = time.perf_counter()

    for op in ops:
        if op == OP_SEARCH:
            qi = q_idx % query_size
            q_idx += 1
            try:
                start = time.perf_counter()
                index.search(query[qi : qi + 1], count)
                search_times.append(time.perf_counter() - start)
            except Exception as e:
                errors += 1
                if len(error_details) < 5:
                    error_details.append(f"search: {e}")

        elif op == OP_ADD:
            if add_idx >= len(add_ids):
                continue
            vid = add_idx
            add_idx += 1
            try:
                start = time.perf_counter()
                index.add(
                    vectors[vid : vid + 1],
                    add_ids[vid : vid + 1],
                )
                add_times.append(time.perf_counter() - start)
            except Exception as e:
                errors += 1
                if len(error_details) < 5:
                    error_details.append(f"add id={add_ids[vid]}: {e}")

        elif op == OP_DELETE:
            if del_idx >= len(delete_ids):
                continue
            del_id = delete_ids[del_idx]
            del_idx += 1
            try:
                start = time.perf_counter()
                index.delete(np.array([del_id], dtype=np.uint64))
                delete_times.append(time.perf_counter() - start)
            except Exception as e:
                errors += 1
                if len(error_details) < 5:
                    error_details.append(f"delete id={del_id}: {e}")

        elif op == OP_CONSOLIDATE:
            try:
                start = time.perf_counter()
                index.consolidate()
                consolidate_times.append(time.perf_counter() - start)
            except Exception as e:
                errors += 1
                if len(error_details) < 5:
                    error_details.append(f"consolidate: {e}")
            break  # Consolidate is slow; stop this thread to keep benchmark concurrent.

    wall_time = (
        sum(search_times) + sum(add_times)
        + sum(delete_times) + sum(consolidate_times)
    )

    results_out["search_times"] = search_times
    results_out["add_times"] = add_times
    results_out["delete_times"] = delete_times
    results_out["consolidate_times"] = consolidate_times
    results_out["wall_time"] = wall_time
    results_out["errors"] = errors
    results_out["error_details"] = error_details


def _log_mixed_results(label, results_list):
    total_errors = 0

    def _per_thread_qps(results_list, op_key):
        """Compute per-thread QPS, then mean and RSD across threads."""
        per_thread_qps_op = []
        per_thread_qps_wall = []
        total_count = 0
        for r in results_list:
            times = r.get(op_key, [])
            wall = r.get("wall_time", 0)
            n = len(times)
            total_count += n
            if not times:
                continue
            op_time = sum(times)
            per_thread_qps_op.append(n / op_time if op_time > 0 else 0)
            per_thread_qps_wall.append(n / wall if wall > 0 else 0)
        return per_thread_qps_op, per_thread_qps_wall, total_count

    def _stats(op_key, name):
        qps_op, qps_wall, count = _per_thread_qps(results_list, op_key)
        if not qps_op:
            return
        qps_op_arr = np.array(qps_op)
        qps_wall_arr = np.array(qps_wall)
        # Collect all times for latency stats
        all_times = []
        for r in results_list:
            all_times.extend(r.get(op_key, []))
        a = np.array(all_times)
        logger.info({
            f"{label}_{name}": {
                "count": count,
                "qps_op_time_mean": float(np.mean(qps_op_arr)),
                "qps_op_time_rsd": float(
                    np.std(qps_op_arr, ddof=min(1, len(qps_op_arr) - 1))
                    / np.mean(qps_op_arr)
                ),
                "qps_wall_time_mean": float(np.mean(qps_wall_arr)),
                "qps_wall_time_rsd": float(
                    np.std(qps_wall_arr, ddof=min(1, len(qps_wall_arr) - 1))
                    / np.mean(qps_wall_arr)
                ),
                "mean_ms": float(np.mean(a) * 1000),
                "p95_ms": float(np.percentile(a, 95) * 1000),
                "p99_ms": float(np.percentile(a, 99) * 1000),
            }
        })

    _stats("search_times", "search")
    _stats("add_times", "add")
    _stats("delete_times", "delete")

    all_consolidate = []
    for r in results_list:
        all_consolidate.extend(r.get("consolidate_times", []))
    if all_consolidate:
        a = np.array(all_consolidate)
        logger.info({
            f"{label}_consolidate": {
                "count": len(a),
                "mean_ms": float(np.mean(a) * 1000),
                "p95_ms": float(np.percentile(a, 95) * 1000),
            }
        })

    total_errors = sum(r.get("errors", 0) for r in results_list)
    if total_errors > 0:
        all_details = []
        for r in results_list:
            all_details.extend(r.get("error_details", []))
        logger.info({
            f"{label}_errors": total_errors,
            f"{label}_error_samples": all_details[:20],
        })


def mixed_benchmark(
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
    # mixed params
    n_workers: int = 1,
    add_ratio: float = 0.1,
    delete_ratio: float = 0.1,
    n_consolidate: int = 2,
) -> None:
    logger.info({"mixed_args": locals()})
    logger.info(utils.read_system_config())

    search_ratio = 1.0 - add_ratio - delete_ratio
    if search_ratio < 0:
        raise ValueError("add_ratio + delete_ratio must be <= 1.0")

    # --- Step 1: Load data, split in half ---
    vectors = svs.read_vecs(str(vecs_path))
    if num_vectors is not None:
        vectors = vectors[:num_vectors]
    total = vectors.shape[0]
    half = total // 2
    query = svs.read_vecs(str(query_path))

    # --- Step 2: Build dynamic index with first half ---
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

    update_vectors = vectors[half:]
    update_ids_base = half

    # --- Step 3: Calibrate search ---
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

    # --- Step 4: Generate operation streams and run ---
    index.num_threads = 1

    # Each thread processes all queries — same as search.py/concurrent.py.
    ops_per_thread = query.shape[0]
    # Divide update vectors evenly across threads (limits max adds).
    adds_per_thread = len(update_vectors) // n_workers

    op_weights = np.array([search_ratio, add_ratio, delete_ratio])

    logger.info({
        "workload_per_thread": {
            "ops_per_thread": ops_per_thread,
            "max_adds_per_thread": adds_per_thread,
            "expected_adds": int(ops_per_thread * add_ratio),
            "expected_deletes": int(ops_per_thread * delete_ratio),
            "expected_searches": int(ops_per_thread * search_ratio),
        }
    })

    barrier = threading.Barrier(n_workers)
    results = [{} for _ in range(n_workers)]
    threads = []

    # Distribute n_consolidate calls across random threads
    rng_main = np.random.default_rng(seed)
    consolidate_per_thread = [0] * n_workers
    for _ in range(n_consolidate):
        consolidate_per_thread[rng_main.integers(n_workers)] += 1

    for i in range(n_workers):
        thread_rng = np.random.default_rng(seed + i)
        # Generate search/add/delete ops
        ops = thread_rng.choice(
            [OP_SEARCH, OP_ADD, OP_DELETE],
            size=ops_per_thread,
            p=op_weights,
        )
        # Insert this thread's share of consolidate ops at random positions
        n_cons = consolidate_per_thread[i]
        if n_cons > 0:
            insert_positions = thread_rng.choice(
                len(ops) + 1, size=n_cons, replace=False,
            )
            insert_positions.sort()
            ops = np.insert(ops, insert_positions, OP_CONSOLIDATE)

        # Each thread gets its own slice of update vectors for adds.
        add_start = i * adds_per_thread
        add_end = add_start + adds_per_thread
        thread_vectors = update_vectors[add_start:add_end]
        thread_ids = np.arange(
            update_ids_base + add_start,
            update_ids_base + add_end,
            dtype=np.uint64,
        )

        # Pre-sample delete targets from the initial index.
        num_deletes = int(np.sum(ops == OP_DELETE))
        thread_delete_ids = thread_rng.choice(
            half, size=num_deletes, replace=True,
        ).astype(np.uint64)

        t = threading.Thread(
            target=_mixed_worker,
            args=(
                index, query, count, thread_vectors, thread_ids,
                thread_delete_ids, ops, barrier, results[i],
            ),
            name=f"mixed-{i}",
        )
        threads.append(t)

    logger.info({
        "mixed_config": {
            "n_workers": n_workers,
            "ops_per_thread": ops_per_thread,
            "n_consolidate": n_consolidate,
            "add_ratio": add_ratio,
            "delete_ratio": delete_ratio,
            "search_ratio": search_ratio,
        }
    })
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_time = time.perf_counter() - wall_start

    logger.info({"mixed_wall_time": wall_time})
    _log_mixed_results("mixed", results)


# --- CLI ---

def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    utils.add_common_arguments(parser)
    parser.add_argument("--vecs_file", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--idx_dir", type=Path)
    parser.add_argument("--num_vectors", type=int)
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
    parser.add_argument("--max_threads_init", type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--tmp_dir", type=Path, default="/dev/shm")
    parser.add_argument("--leanvec_dims", type=int)
    parser.add_argument("--query_file", type=Path, required=True)
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
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="Total concurrent worker threads",
    )
    parser.add_argument(
        "--add_ratio", type=float, default=0.1,
        help="Fraction of operations that are adds (0.0-1.0)",
    )
    parser.add_argument(
        "--delete_ratio", type=float, default=0.1,
        help="Fraction of operations that are deletes (0.0-1.0)",
    )
    parser.add_argument(
        "--n_consolidate", type=int, default=2,
        help="Number of consolidate calls per thread",
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
    mixed_benchmark(
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
        add_ratio=args.add_ratio,
        delete_ratio=args.delete_ratio,
        n_consolidate=args.n_consolidate,
    )


if __name__ == "__main__":
    main()
