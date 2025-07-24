# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generate LeanVec-OOD matrices."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import svs
import typer

from . import consts, merge


def main(
    vecs_file: Path,
    train_query_file: Path,
    max_vectors: int = consts.DEFAULT_LEANVEC_TRAIN_MAX_VECTORS,
    leanvec_dims: int = consts.DEFAULT_LEANVEC_DIMS,
    out_dir: Path = Path(),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (data_matrix, query_matrix), (leanvec_dims_effective, _) = (
        generate_leanvec_matrices(
            vecs_file, train_query_file, max_vectors, leanvec_dims
        )
    )
    data_matrix_path, query_matrix_path = save_leanvec_matrices(
        vecs_file,
        train_query_file,
        max_vectors,
        leanvec_dims_effective,
        data_matrix,
        query_matrix,
        out_dir,
    )
    print("Saved LeanVec matrices:", data_matrix_path, query_matrix_path)


def generate_leanvec_matrices(
    vecs_file: Path,
    train_query_file: Path,
    max_vectors: int = consts.DEFAULT_LEANVEC_TRAIN_MAX_VECTORS,
    leanvec_dims: int | None = None,
) -> tuple[tuple[npt.NDArray, npt.NDArray], tuple[int, int]]:
    """Generate LeanVec matrices from base vectors and query vectors."""
    base_vectors = merge.read_vecs(vecs_file, max_vectors)
    query_vectors = merge.read_vecs(train_query_file)
    dim = base_vectors.shape[1]
    if leanvec_dims is None:
        leanvec_dims = consts.DEFAULT_LEANVEC_DIMS
    if leanvec_dims < 0:
        leanvec_dims = dim // -leanvec_dims
    return svs.compute_leanvec_matrices(
        base_vectors, query_vectors, leanvec_dims
    ), (leanvec_dims, max_vectors)


def save_leanvec_matrices(
    vecs_file: Path,
    train_query_file: Path,
    max_vectors: int,
    leanvec_dims: int,
    data_matrix: npt.NDArray,
    query_matrix: npt.NDArray,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Save LeanVec matrices to files."""
    name_components = [
        vecs_file.name,
        train_query_file.name,
        str(leanvec_dims),
    ]
    if max_vectors > 0:
        name_components.append(str(max_vectors))
    base_name = "__".join(name_components)
    data_matrix_path = out_dir / (base_name + ".data.npy")
    query_matrix_path = out_dir / (base_name + ".query.npy")
    np.save(data_matrix_path, data_matrix)
    np.save(query_matrix_path, query_matrix)
    return data_matrix_path, query_matrix_path


if __name__ == "__main__":
    # https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    typer.run(main)
