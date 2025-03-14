import argparse
import os
from pathlib import Path

import hnswlib
import numpy as np

import merge

def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description=__file__.__doc__)
    parser.add_argument("--vecs_file", help="Vectors *vecs file", type=Path)
    parser.add_argument(
        "--out_dir",
        help="Output dir where SVS index dir will be created",
        type=Path,
        default="ebay",
    )
    parser.add_argument(
        "--num_threads",
        help="Number of threads",
        type=int,
        default=max(len(os.sched_getaffinity(0)) - 1, 1),
    )
    parser.add_argument("--distance", choices=("l2", "ip"), default="ip")
    parser.add_argument("--ef_construction", type=int, default=500)
    parser.add_argument("-m", type=int, default=64)
    parser.add_argument("--num_vectors", type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = _read_args()
    print(args)
    build(
        vecs_path=args.vecs_file,
        out_path=args.out_dir,
        num_threads=args.num_threads,
        distance=args.distance,
        ef_construction=args.ef_construction,
        m=args.m,
        num_vectors=args.num_vectors,
    )

def build(*, vecs_path: Path, out_path: Path, num_threads: int, distance: str, ef_construction: int, m: int, num_vectors: int | None) -> None:
    X_db = merge.read_vecs(vecs_path)
    if num_vectors is not None:
        X_db = X_db[:num_vectors]
    else:
        num_vectors = 0

    num_elements = X_db.shape[0]
    dim = X_db.shape[1]
    ids = np.arange(num_elements)

    p = hnswlib.Index(space = distance, dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = ef_construction, M = m)
    p.set_num_threads(num_threads)
    p.add_items(X_db, ids)
    p.save_index(f"{out_path}/{Path(vecs_path).stem}-{num_vectors}_M{m}_{ef_construction}_hnswlib.bin")


if __name__ == "__main__":
    main()
