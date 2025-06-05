# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generate ground truth ivecs file."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import svs

from . import consts
from . import utils

logger = logging.getLogger(__file__)


def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description=__file__.__doc__)
    utils.add_common_arguments(parser)
    parser.add_argument("--vecs_file", help="Vectors *vecs file", type=Path)
    parser.add_argument("--query_file", help="Query vectors file", type=Path)
    parser.add_argument("--out_file", help="Output file", type=Path)
    parser.add_argument(
        "--query_out_file",
        help="Output file for query vectors generated when num_queries given",
        type=Path,
    )
    parser.add_argument(
        "--distance",
        help="Distance",
        choices=tuple(consts.STR_TO_DISTANCE.keys()),
        default="mip",
    )
    parser.add_argument(
        "-k", help="Number of neighbors", type=int, default=100
    )
    parser.add_argument("--num_vectors", help="Number of vectors", type=int)
    parser.add_argument(
        "--num_query_vectors",
        help="Number of query vectors."
        " If given, query vectors will be shuffled."
        " If more than in the query file, the query vectors will be shuffled"
        " and repeated as needed.",
        type=int,
    )
    parser.add_argument(
        "--shuffle", help="Shuffle order of vectors", action="store_true"
    )
    return parser.parse_args(argv)


def main(argv: str | None = None) -> None:
    args = _read_args(argv)
    log_file = utils.configure_logger(
        logger, args.log_dir if args.log_dir is not None else args.out_dir
    )
    print("Logging to", log_file, sep="\n")
    logger.info({"argv": argv if argv else sys.argv})
    generate_ground_truth(
        vecs_path=args.vecs_file,
        query_file=args.query_file,
        distance=consts.STR_TO_DISTANCE[args.distance],
        num_vectors=args.num_vectors,
        k=args.k,
        num_threads=args.max_threads,
        out_file=args.out_file,
        query_out_path=args.query_out_file,
        shuffle=args.shuffle,
        seed=args.seed,
        num_query_vectors=args.num_query_vectors,
    )


def generate_ground_truth(
    *,
    vecs_path: Path,
    query_file: Path,
    distance: svs.DistanceType,
    num_vectors: int | None = None,
    k: int = 100,
    num_threads: int = 1,
    out_file: Path | None = None,
    query_out_path: Path | None = None,
    shuffle: bool = False,
    seed: int = 42,
    num_query_vectors: int | None = None,
) -> None:
    if out_file is not None and out_file.suffix != ".ivecs":
        raise SystemExit("Error: --out_file must end in .ivecs")
    if (
        query_out_path is not None
        and query_out_path.suffix != query_file.suffix
    ):
        raise SystemExit(
            "Error: --query_out_path must have the same suffix as --query_file"
        )
    queries = svs.read_vecs(str(query_file))
    vectors = svs.read_vecs(str(vecs_path))
    # If num_vectors is None or larger than the number of vectors,
    # slicing will return the whole array.
    vectors = vectors[:num_vectors]
    if shuffle:
        np.random.default_rng(seed).shuffle(vectors)
    index = svs.Flat(vectors, distance=distance, num_threads=num_threads)
    idxs, _ = index.search(queries, k)
    if num_query_vectors is not None:
        queries_all = np.empty_like(
            queries, shape=(num_query_vectors, queries.shape[1])
        )
        ground_truth_all = np.empty_like(
            idxs, shape=(num_query_vectors, idxs.shape[1])
        )
        rng = np.random.default_rng(seed)
        cursor = 0
        while cursor < num_query_vectors:
            permutation = rng.permutation(len(queries))
            batch_size = min(num_query_vectors - cursor, len(queries))
            queries_all[cursor : cursor + batch_size] = queries[
                permutation[:batch_size]
            ]
            ground_truth_all[cursor : cursor + batch_size] = idxs[
                permutation[:batch_size]
            ]
            cursor += batch_size
        if query_out_path is None:
            query_out_path = (
                query_file.parent
                / f"{query_file.stem}-{num_query_vectors}_{seed}"
                f"{query_file.suffix}"
            )
        svs.write_vecs(queries_all, str(query_out_path))
        queries_path = query_out_path
    else:
        queries_path = query_file
        ground_truth_all = idxs
    if out_file is None:
        out_file = utils.ground_truth_path(
            vecs_path,
            queries_path,
            distance,
            num_vectors,
            seed if shuffle else None,
        )
    svs.write_vecs(ground_truth_all.astype(np.uint32), str(out_file))
    logger.info({"ground_truth_saved": out_file})


if __name__ == "__main__":
    main()
