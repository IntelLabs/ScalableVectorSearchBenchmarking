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
        "--distance",
        help="Distance",
        choices=tuple(consts.STR_TO_DISTANCE.keys()),
        type=consts.STR_TO_DISTANCE.get,
        default="mip",
    )
    parser.add_argument(
        "-k", help="Number of neighbors", type=int, default=100
    )
    parser.add_argument("--num_vectors", help="Number of vectors", type=int)
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
        distance=args.distance,
        num_vectors=args.num_vectors,
        k=args.k,
        num_threads=args.max_threads,
        out_file=args.out_file,
        shuffle=args.shuffle,
        seed=args.seed,
    )


def generate_ground_truth(
    *,
    vecs_path: Path,
    query_file: Path,
    distance: svs.DistanceType,
    num_vectors: int | None,
    k: int = 100,
    num_threads: int = 1,
    out_file: Path | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> None:
    if out_file is None:
        out_file = utils.ground_truth_path(
            vecs_path, query_file, distance, num_vectors, seed if shuffle else None,
        )
    else:
        if out_file.suffix != ".ivecs":
            raise SystemExit("Error: --out_file must end in .ivecs")
        out_file = str(out_file)
    queries = svs.read_vecs(str(query_file))
    vectors = svs.read_vecs(str(vecs_path))
    if num_vectors is None:
        num_vectors = vectors.shape[0]
    vectors = vectors[:num_vectors]
    if shuffle:
        vectors = vectors[np.random.default_rng(seed).permutation(num_vectors)]
    index = svs.Flat(vectors, distance=distance, num_threads=num_threads)
    idxs, _ = index.search(queries, k)
    svs.write_vecs(idxs.astype(np.uint32), out_file)
    logger.info({"ground_truth_saved": out_file})


if __name__ == "__main__":
    main()
