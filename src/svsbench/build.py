# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Benchmark SVS dynamic index addition."""

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import svs
from tqdm import tqdm

from . import consts
from .loader import create_loader
from . import utils

logger = logging.getLogger(__file__)


def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description=__file__.__doc__)
    utils.add_common_arguments(parser)
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
        "--max_threads_init",
        help="Maximum number of threads for the first operation",
        default=max(len(os.sched_getaffinity(0)) - 1, 1),
        type=int,
    )
    parser.add_argument(
        "--num_vectors_delete",
        help="Number of vectors to delete",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_vectors_init",
        help="Number of vectors to add initially",
        type=int,
    )
    parser.add_argument(
        "--proportion_vectors_init",
        help="Proportion of vectors to add initially",
        type=float,
    )
    parser.add_argument(
        "--max_threads_ignore_batch",
        help="Do not cap max threads based on batch size",
        action="store_true",
    )
    parser.add_argument(
        "--shuffle", help="Shuffle order of vectors", action="store_true"
    )
    parser.add_argument(
        "--static", help="Index is static", action="store_true"
    )
    parser.add_argument(
        "--convert_vecs",
        help="Convert data type of vecs file to the SVS type",
        action="store_true",
    )
    parser.add_argument(
        "--tmp_dir", help="Temporary dir", type=Path, default="/dev/shm"
    )
    parser.add_argument(
        "--leanvec_dims", help="LeanVec dimensionality", type=int
    )
    return parser.parse_args(argv)


def main(argv: str | None = None) -> None:
    args = _read_args(argv)
    log_file = utils.configure_logger(
        logger, args.log_dir if args.log_dir is not None else args.out_dir
    )
    print("Logging to", log_file, sep="\n")
    logger.info({"argv": argv if argv else sys.argv})
    args.out_dir.mkdir(exist_ok=True)
    if args.static:
        index, name = build_static(
            vecs_path=args.vecs_file,
            svs_type=args.svs_type,
            distance=consts.STR_TO_DISTANCE[args.distance],
            graph_max_degree=args.graph_max_degree,
            window_size=args.window_size,
            prune_to=args.prune_to,
            max_candidate_pool_size=args.max_candidate_pool_size,
            alpha=args.alpha,
            max_threads=args.max_threads,
            leanvec_dims=args.leanvec_dims,
        )
    else:
        index, name, ingest_time, delete_time = build_dynamic(
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
            max_threads_ignore_batch=args.max_threads_ignore_batch,
            shuffle=args.shuffle,
            seed=args.seed,
            convert_vecs=args.convert_vecs,
            tmp_dir=args.tmp_dir,
            leanvec_dims=args.leanvec_dims,
        )
        np.save(args.out_dir / (name + ".ingest.npy"), ingest_time)
        if args.num_vectors_delete > 0:
            np.save(args.out_dir / (name + ".delete"), delete_time)
    save(index, args.out_dir, name)


def build_dynamic(
    *,
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
    max_threads_init: int = 1,
    batch_size: int = 10000,
    num_vectors_delete: int = 0,
    num_vectors_init: int | None = None,
    proportion_vectors_init: float | None = None,
    max_threads_ignore_batch: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    convert_vecs: bool = False,
    tmp_dir: Path = Path("/dev/shm"),
    leanvec_dims: int | None = None,
) -> tuple[svs.DynamicVamana, str]:
    """Build SVS index."""
    logger.info({"build_args": locals()})
    logger.info(utils.read_system_config())

    if (vecs_type := consts.SUFFIX_TO_SVS_TYPE.get(vecs_path.suffix)) is None:
        raise ValueError("Unknown suffix: " + vecs_path.suffix)
    if svs_type.startswith("float") and vecs_type != svs_type:
        if not convert_vecs:
            raise ValueError(
                f"Expected svs_type={vecs_type} for {vecs_path=!s}"
                f" based on suffix but got {svs_type=!s}."
                f" You can also use convert_vecs."
            )
        conversion_necessary = True
    else:
        conversion_necessary = False
    vectors = svs.read_vecs(str(vecs_path))
    if conversion_necessary:
        vectors = vectors.astype(consts.SVS_TYPE_TO_DTYPE[svs_type])

    if num_vectors is None:
        num_vectors = vectors.shape[0]
    elif num_vectors > vectors.shape[0]:
        raise ValueError(
            f"{num_vectors=} is greater than the number of vectors"
            f" in {vecs_path=!s}, {vectors.shape[0]}"
        )

    if num_vectors_init is None:
        if proportion_vectors_init is None:
            num_vectors_init = batch_size
        else:
            num_vectors_init = int(num_vectors * proportion_vectors_init)
    if not max_threads_ignore_batch:
        max_threads = min(max_threads, batch_size)
        max_threads_init = min(max_threads_init, num_vectors_init)

    vectors = vectors[:num_vectors]
    if shuffle:
        vectors = vectors[np.random.default_rng(seed).permutation(num_vectors)]

    if prune_to is None:
        prune_to = graph_max_degree - 4
    window_size = window_size
    alpha = consts.DISTANCE_TO_ALPHA[distance] if alpha is None else alpha

    num_batches = int(np.ceil((num_vectors - num_vectors_init) / batch_size))
    ingest_time = np.zeros(num_batches + 1)
    delete_time = np.zeros(num_batches + 1) if num_vectors_delete > 0 else None
    vector_ids = np.array(np.arange(vectors.shape[0]), dtype=np.uint64)

    if idx_dir is None:
        parameters = svs.VamanaBuildParameters(
            graph_max_degree=graph_max_degree,
            window_size=window_size,
            prune_to=prune_to,
            alpha=alpha,
            max_candidate_pool_size=max_candidate_pool_size,
        )

        if svs_type.startswith(("float32", "leanvec", "lvq")):
            start = time.perf_counter()
            index = svs.DynamicVamana.build(
                parameters,
                vectors[:num_vectors_init],
                vector_ids[:num_vectors_init],
                distance,
                num_threads=max_threads_init,
            )
            index_build_time = time.perf_counter() - start
        else:
            start = time.perf_counter()
            index = svs.Vamana.build(
                parameters,
                vectors[:num_vectors_init],
                distance,
                num_threads=max_threads_init,
            )
            index_build_time = time.perf_counter() - start
        logger.info({"index_build_time": index_build_time})
        ingest_time[0] = index_build_time
        if svs_type != "float32":
            with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_idx_dir_str:
                tmp_idx_dir = Path(tmp_idx_dir_str)
                index.save(
                    str(tmp_idx_dir / "config"),
                    str(tmp_idx_dir / "graph"),
                    str(tmp_idx_dir / "data"),
                )
                loader = create_loader(
                    svs_type,
                    data_dir=tmp_idx_dir / "data",
                    compress=not svs_type.startswith("float"),
                    leanvec_dims=leanvec_dims,
                )
                index = svs.DynamicVamana(
                    str(tmp_idx_dir / "config"),
                    str(tmp_idx_dir / "graph"),
                    loader,
                    distance=distance,
                    num_threads=max_threads_init,
                    debug_load_from_static=svs_type.startswith("float"),
                )
    else:
        loader = create_loader(svs_type, data_dir=idx_dir / "data")
        index = svs.DynamicVamana(
            str(idx_dir / "config"),
            svs.GraphLoader(str(idx_dir / "graph")),
            loader,
            distance,
            num_threads=max_threads_init,
        )

    rng_delete = np.random.default_rng(seed)
    logger.info(
        {"experimental_backend_string": index.experimental_backend_string}
    )
    index.num_threads = max_threads
    for batch_idx in tqdm(range(num_batches)):
        init_batch = batch_idx * batch_size + num_vectors_init
        end_batch = min(init_batch + batch_size, num_vectors)
        start = time.perf_counter()
        index.add(
            vectors[init_batch:end_batch], vector_ids[init_batch:end_batch]
        )
        batch_time = time.perf_counter() - start
        ingest_time[batch_idx + 1] = batch_time
        if num_vectors_delete > 0:
            ids_to_delete = rng_delete.choice(
                index.all_ids(), size=num_vectors_delete, replace=False
            )
            start = time.perf_counter()
            index.delete(ids_to_delete)
            delete_time[batch_idx + 1] = time.perf_counter() - start

    name = "__".join(
        (
            "svs",
            "vecs_file=" + vecs_path.name,
            "distance=" + consts.DISTANCE_TO_STR[distance],
            "num_vectors=" + str(num_vectors),
            "graph_max_degree=" + str(graph_max_degree),
            "window_size=" + str(window_size),
            "num_vectors_init=" + str(num_vectors_init),
            "batch_size=" + str(batch_size),
            "svs_type=" + svs_type,
            "shuffle=" + (str(seed) if shuffle else "False"),
            "idx_dir=" + str(idx_dir is not None),
        )
    )
    return index, name, ingest_time, delete_time


def build_static(
    *,
    vecs_path: Path,
    svs_type: str,
    distance: svs.DistanceType,
    graph_max_degree: int = 64,
    window_size: int = 200,
    prune_to: int | None = None,
    max_candidate_pool_size: int = 750,
    alpha: float | None = None,
    max_threads: int = 1,
    leanvec_dims: int | None = None,
) -> tuple[svs.Vamana, str]:
    logger.info({"build_args": locals()})
    logger.info(utils.read_system_config())
    if prune_to is None:
        prune_to = graph_max_degree - 4
    alpha = consts.DISTANCE_TO_ALPHA[distance] if alpha is None else alpha
    parameters = svs.VamanaBuildParameters(
        graph_max_degree=graph_max_degree,
        window_size=window_size,
        prune_to=prune_to,
        alpha=alpha,
        max_candidate_pool_size=max_candidate_pool_size,
    )
    start = time.perf_counter()
    index = svs.Vamana.build(
        parameters,
        create_loader(
            svs_type, vecs_path=vecs_path, leanvec_dims=leanvec_dims
        ),
        distance,
        num_threads=max_threads,
    )
    index_build_time = time.perf_counter() - start
    logger.info(
        {"experimental_backend_string": index.experimental_backend_string}
    )
    logger.info({"index_build_time": index_build_time})
    return index, "__".join(
        (
            "svs",
            "vecs_file=" + vecs_path.name,
            "distance=" + consts.DISTANCE_TO_STR[distance],
            "graph_max_degree=" + str(graph_max_degree),
            "window_size=" + str(window_size),
            "svs_type=" + svs_type,
        )
    )


def save(
    index: svs.Vamana | svs.DynamicVamana,
    out_dir: Path = Path("out"),
    name: str = "index",
) -> None:
    idx_dir = out_dir / name
    idx_dir.mkdir(exist_ok=True)
    index.save(
        str(idx_dir / "config"),
        str(idx_dir / "graph"),
        str(idx_dir / "data"),
    )
    logger.info({"index_saved": idx_dir})


if __name__ == "__main__":
    main()
