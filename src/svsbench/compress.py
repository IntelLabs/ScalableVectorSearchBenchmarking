# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Compress SVS index."""

import argparse
import logging
import sys
from pathlib import Path

import svs

from . import utils
from .loader import create_loader

logger = logging.getLogger(__file__)


def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description=__file__.__doc__)
    parser.add_argument("--idx_dir", help="Index dir", type=Path)
    parser.add_argument(
        "--no_dynamic", action="store_true", help="Do not use dynamic index"
    )
    parser.add_argument(
        "--load_from_static",
        action="store_true",
        help="Load from static index",
    )
    utils.add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv: str | None = None) -> None:
    args = _read_args(argv)
    log_file = utils.configure_logger(
        logger, args.log_dir if args.log_dir is not None else args.out_dir
    )
    print("Logging to", log_file, sep="\n")
    logger.info({"argv": argv if argv else sys.argv})
    compress(
        idx_dir=args.idx_dir,
        svs_type=args.svs_type,
        out_dir=args.out_dir,
        dynamic=not args.no_dynamic,
        max_threads=args.max_threads,
        load_from_static=args.load_from_static,
    )


def compress(
    *,
    idx_dir: Path,
    svs_type: str,
    out_dir: Path = Path("out"),
    dynamic: bool = True,
    max_threads: int = 1,
    load_from_static: bool = False,
) -> None:
    """Compress SVS index."""
    logger.info({"build_args": locals()})
    logger.info(utils.read_system_config())
    out_dir.mkdir(exist_ok=True)
    save_dir = out_dir / (idx_dir.name + f"__compressed={svs_type}")
    save_dir.mkdir(exist_ok=True)
    loader = create_loader(svs_type, data_dir=idx_dir / "data", compress=True)
    if dynamic:
        index = svs.DynamicVamana(
            str(idx_dir / "config"),
            str(idx_dir / "graph"),
            loader,
            num_threads=max_threads,
            debug_load_from_static=load_from_static,
        )
    else:
        index = svs.Vamana(
            str(idx_dir / "config"),
            str(idx_dir / "graph"),
            loader,
            num_threads=max_threads,
        )
    logger.info(
        {"experimental_backend_string": index.experimental_backend_string}
    )
    index.save(
        str(save_dir / "config"),
        str(save_dir / "graph"),
        str(save_dir / "data"),
    )
    logger.info({"save_dir": save_dir})


if __name__ == "__main__":
    main()
