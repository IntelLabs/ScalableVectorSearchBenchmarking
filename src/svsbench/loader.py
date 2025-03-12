# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""SVS loader."""

from pathlib import Path

import svs

from . import consts


def create_loader(
    svs_type: str,
    *,
    vecs_path: Path = None,
    data_dir: Path = None,
    compress: bool = False,
    leanvec_dims: int = -4,
    leanvec_alignment: int = 32,
) -> svs.VectorDataLoader | svs.LVQLoader | svs.LeanVecLoader:
    """Create loader."""
    unkown_msg = f"Unknown {svs_type=}"
    if svs_type not in consts.SVS_TYPES:
        raise ValueError(unkown_msg)
    if not svs_type.startswith(("leanvec", "lvq")) and compress:
        raise ValueError(f"Compression is not supported for {svs_type=}")
    if vecs_path is None:
        if data_dir is None:
            raise ValueError("Either vecs_path or data_dir must be not None")
        if not svs_type.startswith(("leanvec", "lvq")) or compress:
            loader_or_str = svs.VectorDataLoader(str(data_dir))
        else:
            loader_or_str = str(data_dir)
    else:
        if data_dir is not None:
            raise ValueError(
                "vecs_path and data_dir cannot be used at the same time"
            )
        if (
            vecs_type := consts.SUFFIX_TO_SVS_TYPE.get(vecs_path.suffix)
        ) is None:
            raise ValueError("Unknown suffix: " + vecs_path.suffix)
        loader_or_str = svs.VectorDataLoader(
            str(vecs_path), consts.SUFFIX_TO_DATA_TYPE[vecs_path.suffix]
        )
    if svs_type.startswith("lvq"):
        if svs_type == "lvq4x4":
            primary = 4
            residual = 4
            strategy = svs.LVQStrategy.Turbo
        elif svs_type == "lvq4x8":
            primary = 4
            residual = 8
            strategy = svs.LVQStrategy.Turbo
        elif svs_type == "lvq8":
            primary = 8
            residual = 0
            strategy = svs.LVQStrategy.Sequential
        else:
            raise ValueError(unkown_msg)
        padding = 32
        if vecs_path is not None or compress:
            loader = svs.LVQLoader(
                loader_or_str,
                primary=primary,
                residual=residual,
                padding=padding,
                strategy=strategy,
            )
        else:
            loader = svs.LVQLoader(
                loader_or_str, padding=padding, strategy=strategy
            )
    elif svs_type.startswith("leanvec"):
        if svs_type == "leanvec4x4":
            primary = svs.LeanVecKind.lvq4
            secondary = svs.LeanVecKind.lvq4
        elif svs_type == "leanvec4x8":
            primary = svs.LeanVecKind.lvq4
            secondary = svs.LeanVecKind.lvq8
        elif svs_type == "leanvec8x8":
            primary = svs.LeanVecKind.lvq8
            secondary = svs.LeanVecKind.lvq8
        else:
            raise ValueError(unkown_msg)
        if vecs_path is not None or compress:
            if leanvec_dims < 0:
                leanvec_dims = loader_or_str.dims // -leanvec_dims
            loader = svs.LeanVecLoader(
                loader_or_str,
                leanvec_dims=leanvec_dims,
                primary_kind=primary,
                secondary_kind=secondary,
                alignment=leanvec_alignment,
            )
        else:
            loader = svs.LeanVecLoader(
                loader_or_str, alignment=leanvec_alignment
            )
    else:
        if vecs_path is not None and vecs_type != svs_type:
            raise ValueError(
                f"Expected svs_type={vecs_type} for {vecs_path=!s}"
                f" based on suffix but got {svs_type=!s}"
            )
        loader = loader_or_str
    return loader
