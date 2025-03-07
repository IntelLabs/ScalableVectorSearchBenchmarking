# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Merge multiple *vecs files into one."""
import argparse
import struct
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from tqdm import tqdm


SUFFIX_TO_DTYPE: Final = {
    ".fvecs": np.float32,
    ".hvecs": np.float16,
    ".bvecs": np.uint8,
    ".ivecs": np.uint32,
}


SUFFIX_TO_PADDING: Final = {
    ".fvecs": 1,
    ".hvecs": 2,
    ".bvecs": 4,
    ".ivecs": 1,
}


def _read_dim(fname: Path) -> int:
    """Read vector dimension from *vecs."""
    with open(fname, "rb") as file:
        dim = struct.unpack("i", file.read(4))[0]
    return dim


def read_vecs(fname: Path) -> npt.NDArray:
    """Create NumPy memory maps."""
    dim = _read_dim(fname)
    padding = SUFFIX_TO_PADDING[fname.suffix]
    array = np.memmap(fname, dtype=SUFFIX_TO_DTYPE[fname.suffix], mode="r")
    return array.reshape(-1, dim + padding)[:, padding:]


def write(inputs: list[Path], output: Path, num_vectors: int | None) -> None:
    """Write merged array."""
    dim = _read_dim(inputs[0])
    dim_bytes = dim.to_bytes(4, "little")
    with open(output, "wb") as file:
        for fname in tqdm(inputs):
            array = read_vecs(fname)
            if num_vectors is not None:
                array = array[:num_vectors]
            for vector in tqdm(array):
                file.write(dim_bytes)
                file.write(vector.tobytes())


def _read_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read command line arguments."""
    parser = argparse.ArgumentParser(description=__file__.__doc__)
    parser.add_argument(
            "i", help="Input file names", action="extend", type=Path, nargs="+"
    )
    parser.add_argument("-o", help="Output file name", type=Path)
    parser.add_argument("--num_vectors", type=int)
    return parser.parse_args(argv)


def main():
    args = _read_args()
    write(args.i, args.o, args.num_vectors)


if __name__ == "__main__":
    main()
