"""HNSWlib benchmark for search."""
import argparse
import logging
import os
import time
import sys
import itertools
from pathlib import Path
from typing import Final

import hnswlib
import numpy as np
from tqdm import tqdm

import utils

EF_DEFAULT: Final = (175, 200, 225)

logger = logging.getLogger(__file__)


def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description=__file__.__doc__)
    parser.add_argument("--batch_size", help="Batch size", type=int, action="append")
    parser.add_argument("--max_threads", help="Maximum number of threads", default=max(len(os.sched_getaffinity(0)) - 1, 1), type=int)
    parser.add_argument("--idx_file", help="Index file", type=Path)
    parser.add_argument("--ground_truth_file", help="Ground truth file", type=Path)
    parser.add_argument("--query_file", help="Query file", type=Path, default="ebay/queries.fvecs")
    parser.add_argument("-k", help="Number of neighbors to return", default=10, type=int)
    parser.add_argument("--num_rep", help="Number of search repetitions", default=5, type=int)
    parser.add_argument("--ef", help="ef", action="append", type=int)
    parser.add_argument("--distance", help="Distance", choices=("l2", "ip"), default="ip")
    parser.add_argument("--log_dir", help="Log dir", default="logs", type=Path)
    return parser.parse_args(argv)

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

def k_recall_at(gt_idx, result_idx, k: int, at: int):
    if k > at:
        raise ValueError(f'K={k} is higher than @={at}')
    if gt_idx.shape[1] < k:
        raise ValueError(f'Too few ground truth neighbors'
                         f'({gt_idx.shape[1]}) to compute {k}-recall')
    if result_idx.shape[1] < at:
        raise ValueError(f'Too few approximate neighbors'
                         f'({result_idx.shape[1]}) to compute recall@{at}')

    ls_intersection = itertools.starmap(np.intersect1d,
                                        zip(gt_idx[:, :k], result_idx[:, :at]))

    ls_recall = [len(intersect) for intersect in ls_intersection]

    return sum(ls_recall) / (len(ls_recall) * k)

def search(
    *,
    batch_sizes: list[int],
    max_threads: int,
    idx_file: Path,
    ground_truth_file: Path,
    k: int = 10,
    num_rep: int = 5,
    query_file: Path = Path("ebay/queries.fvecs"),
    efs: list[int],
    distance: str = "ip",
) -> None:
    logger.info({"search_args": locals()})
    logger.info(utils.read_system_config())
    ground_truth = ivecs_read(ground_truth_file)
    X_query = fvecs_read(query_file)

    nq = X_query.shape[0]
    dim = X_query.shape[1]

    p = hnswlib.Index(space = distance, dim = dim)
    p.load_index(str(idx_file))

    for batch_size in batch_sizes:
        num_threads = min(max_threads, batch_size)
        print(f"Batch size = {batch_size}")
        print(f"Number of Threads = {num_threads}")

        num_batches = int(np.ceil(nq/float(batch_size)))
        print(f"Number of queries = {nq}, number of batches = {num_batches}")

        p.set_num_threads(num_threads)

        for ef in tqdm(efs):
            p.set_ef(ef)
            qps = []
            p95s = []

            for i in tqdm(range(num_rep + 1)):
                total_time = float(0)
                batch_times = []
                idx = np.empty((0, k), int)

                for bb in tqdm(range(num_batches)):
                    init_batch = bb*batch_size
                    end_batch = min(init_batch + batch_size, nq)

                    start = time.perf_counter()
                    res, _ = p.knn_query(X_query[init_batch:end_batch], k=k)
                    batch_time = time.perf_counter() - start
                    batch_times.append(batch_time)
                    total_time += batch_time
                    idx = np.append(idx, res, axis=0)
                if i > 0:
                    qps.append(nq/total_time)
                    p95s.append(np.percentile(batch_times, 95))

            print("qps: Avg {} Max {} StdDev {}".format(np.mean(qps), max(qps), np.std(qps, ddof=1)))
            print("All qps values:", qps)
            print("Mean p95:", np.mean(p95s))
            print("All p95 values:", p95s)
            rec = k_recall_at(ground_truth, idx, k, k)
            print(f'ef_search = {ef}, recall = {rec}')
            logger.info(
                {
                    "search_results": {
                        "qps": qps,
                        "p95": p95s,
                        "search_parameters": {"ef": ef},
                        "batch_size": batch_size,
                        "recall": rec,
                    },
                }
            )

def main() -> None:
    args = _read_args()
    efs = args.ef if args.ef is not None else EF_DEFAULT
    log_file = utils.configure_logger(logger, args.log_dir)
    print("Logging to", log_file, sep="\n")
    logger.info({"argv": sys.argv})
    search(
        batch_sizes=args.batch_size,
        max_threads=args.max_threads,
        idx_file=args.idx_file,
        query_file=args.query_file,
        ground_truth_file=args.ground_truth_file,
        k=args.k,
        num_rep=args.num_rep,
        efs=efs,
        distance=args.distance,
    )

if __name__ == "__main__":
    main()
