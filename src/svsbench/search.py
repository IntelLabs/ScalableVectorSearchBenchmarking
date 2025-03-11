# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Search benchmark."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import svs
from tqdm import tqdm

from . import consts, utils
from .loader import create_loader

STR_TO_STRATEGY: Final[dict[str, svs.LVQStrategy]] = {
    "auto": svs.LVQStrategy.Auto,
    "sequential": svs.LVQStrategy.Sequential,
    "turbo": svs.LVQStrategy.Turbo,
}


logger = logging.getLogger(__file__)


def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description=__file__.__doc__)
    utils.add_common_arguments(parser)
    parser.add_argument(
        "--batch_size", help="Batch size", type=int, action="append"
    )
    parser.add_argument(
        "--query_type",
        help="Query type",
        choices=consts.STR_TO_DATA_TYPE.keys(),
        default="float32",
        type=consts.STR_TO_DATA_TYPE.get,
    )
    parser.add_argument("--idx_dir", help="Index dir", type=Path)
    parser.add_argument("--data_dir", help="Data dir", type=Path)
    parser.add_argument("--vecs_file", help="Vectors *vecs file", type=Path)
    parser.add_argument("--query_file", help="Query *vecs file", type=Path)
    parser.add_argument(
        "--calibration_query_file",
        help="Calibration query *vecs file",
        type=Path,
    )
    parser.add_argument(
        "--ground_truth_file", help="Ground truth ivecs file", type=Path
    )
    parser.add_argument(
        "--strategy",
        help="LVQ strategy",
        choices=tuple(STR_TO_STRATEGY.keys()),
        default="auto",
        type=STR_TO_STRATEGY.get,
    )
    parser.add_argument(
        "--leanvec_dims", help="LeanVec dimensionality", default=-4, type=int
    )
    parser.add_argument(
        "--leanvec_alignment", help="LeanVec alignment", default=32, type=int
    )
    parser.add_argument(
        "--search_window_size",
        help="Search window size",
        type=int,
        action="append",
    )
    parser.add_argument(
        "--search_buffer_capacity",
        help="Search buffer capacity",
        type=int,
        action="append",
    )
    parser.add_argument(
        "--prefetch_lookahead",
        help="Prefetch lookahead",
        type=int,
        action="append",
    )
    parser.add_argument(
        "--prefetch_step", help="Prefetch step", type=int, action="append"
    )
    parser.add_argument(
        "--recall",
        help="Target recall for calibration",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "-k", help="Number of neighbors to return", default=10, type=int
    )
    parser.add_argument(
        "--idx_not_compressed",
        help="The index is not compressed",
        action="store_true",
    )
    parser.add_argument(
        "--num_rep", help="Number of search repetitions", default=5, type=int
    )
    parser.add_argument(
        "--static", help="Index is static", action="store_true"
    )
    parser.add_argument(
        "--distance",
        choices=tuple(consts.STR_TO_DISTANCE.keys()),
        default="mip",
        type=consts.STR_TO_DISTANCE.get,
    )
    parser.add_argument(
        "--load_from_static",
        action="store_true",
        help="Load from static index",
    )
    return parser.parse_args(argv)


def search(
    *,
    svs_type: str,
    idx_dir: Path,
    vecs_path: Path = None,
    data_dir: Path = None,
    query_path: Path | None = None,
    ground_truth_path: Path | None = None,
    batch_sizes: list[int] = [10000],
    count=10,
    max_threads=255,
    search_window_sizes: list[int] | None = None,
    recall: float = 0.9,
    compress: bool = True,
    leanvec_dims: int | None = -4,
    leanvec_alignment: int | None = 32,
    num_rep: int = 5,
    search_buffer_capacities: list[int] | None = None,
    prefetch_lookaheads: list[int] | None = None,
    prefetch_steps: list[int] | None = None,
    static: bool = False,
    distance: svs.DistanceType,
    query_type: svs.DataType = svs.DataType.float32,
    calibration_query_path: Path | None = None,
    load_from_static: bool = False,
):
    logger.info({"search_args": locals()})
    logger.info(utils.read_system_config())
    if query_path is None:
        query_path = idx_dir.parent / "query.fvecs"
    if ground_truth_path is None:
        if vecs_path is None:
            raise ValueError("Could not find ground truth")
        ground_truth_path = utils.ground_truth_path(
            vecs_path, query_path, distance, None, False
        )
    if data_dir is None:
        data_dir = idx_dir / "data"
    if not ground_truth_path.is_file():
        raise ValueError(
            "Ground truth path does not point to a file", ground_truth_path
        )
    if not query_path.is_file():
        raise ValueError("Query path does not point to a file", query_path)

    loader = create_loader(
        svs_type,
        vecs_path=vecs_path,
        data_dir=data_dir,
        compress=compress,
        leanvec_dims=leanvec_dims,
        leanvec_alignment=leanvec_alignment,
    )

    if static:
        index_class = svs.Vamana
        extra_kwargs = {}
    else:
        index_class = svs.DynamicVamana
        extra_kwargs = {"debug_load_from_static": load_from_static}
    index = index_class(
        str(idx_dir / "config"),
        str(idx_dir / "graph"),
        loader,
        query_type=query_type,
        distance=distance,
        num_threads=max_threads,
        enforce_dims=True,
        **extra_kwargs,
    )
    logger.info({"backend_string": index.experimental_backend_string})

    query = svs.read_vecs(str(query_path))
    ground_truth = svs.read_vecs(str(ground_truth_path))

    for batch_size_idx, batch_size in enumerate(batch_sizes):
        index.num_threads = min(max_threads, batch_size)
        if search_window_sizes is None:
            calibration_query = (
                svs.read_vecs(str(calibration_query_path))
                if calibration_query_path is not None
                else query
            )
            index.experimental_calibrate(
                calibration_query, ground_truth, count, recall
            )
            logger.info(
                {
                    "calibration_results": {
                        "search_window_size": index.search_parameters.buffer_config.search_window_size,
                        "search_buffer_capacity": index.search_parameters.buffer_config.search_buffer_capacity,
                        "prefetch_lookahead": index.search_parameters.prefetch_lookahead,
                        "prefetch_step": index.search_parameters.prefetch_step,
                        "calibration_parameters": {
                            "recall": recall,
                            "count": count,
                            "num_threads": index.num_threads,
                        },
                    }
                }
            )
        else:
            buffer_config_kwargs = (
                {}
                if search_buffer_capacities is None
                else {
                    "search_buffer_capacity": search_buffer_capacities[
                        batch_size_idx
                    ]
                }
            )
            buffer_config = svs.SearchBufferConfig(
                search_window_size=search_window_sizes[batch_size_idx],
                **buffer_config_kwargs,
            )
            search_params_kwargs = {}
            if prefetch_lookaheads is not None:
                search_params_kwargs["prefetch_lookahead"] = (
                    prefetch_lookaheads[batch_size_idx]
                )
            if prefetch_steps is not None:
                search_params_kwargs["prefetch_step"] = prefetch_steps[
                    batch_size_idx
                ]
            search_params = svs.VamanaSearchParameters(
                buffer_config, **search_params_kwargs
            )
            index.search_parameters = search_params
            # Warm-up search instead of calibration
            index.search(query, count)

        logger.info({"free_huge_pages": utils.read_free_huge_pages()})

        query_size = query.shape[0]
        num_batches = int(np.ceil(query_size / float(batch_size)))

        qps = []
        p95s = []
        for _ in tqdm(range(num_rep)):
            total_time = 0
            results = np.empty((0, count), np.int32)
            batch_times = []
            for batch_idx in tqdm(range(num_batches)):
                init_batch = batch_idx * batch_size
                end_batch = min(init_batch + batch_size, query_size)

                start = time.perf_counter()
                result, _ = index.search(query[init_batch:end_batch], count)
                batch_time = time.perf_counter() - start
                total_time += batch_time
                batch_times.append(batch_time)
                results = np.append(results, result, axis=0)

            qps.append(query_size / total_time)
            p95s.append(np.percentile(batch_times, 95))
        recall = svs.k_recall_at(ground_truth, results, count, count)
        logger.info(
            {
                "search_results": {
                    "qps": qps,
                    "p95": p95s,
                    "search_parameters": {
                        "search_window_size": index.search_parameters.buffer_config.search_window_size,
                        "search_buffer_capacity": index.search_parameters.buffer_config.search_buffer_capacity,
                        "prefetch_lookahead": index.search_parameters.prefetch_lookahead,
                        "prefetch_step": index.search_parameters.prefetch_step,
                    },
                    "batch_size": batch_size,
                    "recall": recall,
                },
            }
        )


def main(argv: str | None = None) -> None:
    args = _read_args(argv)
    if args.batch_size is None:
        # https://github.com/python/cpython/issues/60603
        args.batch_size = [10000]
    log_file = utils.configure_logger(
        logger, args.log_dir if args.log_dir is not None else args.out_dir
    )
    print("Logging to", log_file, sep="\n")
    logger.info({"argv": sys.argv})
    search(
        idx_dir=args.idx_dir,
        vecs_path=args.vecs_file,
        data_dir=args.data_dir,
        query_path=args.query_file,
        ground_truth_path=args.ground_truth_file,
        svs_type=args.svs_type,
        batch_sizes=args.batch_size,
        max_threads=args.max_threads,
        search_window_sizes=args.search_window_size,
        recall=args.recall,
        count=args.k,
        compress=args.idx_not_compressed,
        leanvec_dims=args.leanvec_dims,
        leanvec_alignment=args.leanvec_alignment,
        search_buffer_capacities=args.search_buffer_capacity,
        prefetch_lookaheads=args.prefetch_lookahead,
        prefetch_steps=args.prefetch_step,
        num_rep=args.num_rep,
        static=args.static,
        distance=args.distance,
        query_type=args.query_type,
        calibration_query_path=args.calibration_query_file,
        load_from_static=args.load_from_static,
    )


if __name__ == "__main__":
    main()
