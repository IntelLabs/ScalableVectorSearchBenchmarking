# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import svs

from svsbench.consts import SUFFIX_TO_SVS_TYPE, SVS_TYPES
from svsbench.loader import create_loader


@pytest.mark.parametrize("svs_type", SVS_TYPES)
def test_create_loader(index_dir_with_svs_type_and_dynamic, svs_type):
    index_dir, index_svs_type, index_dynamic = (
        index_dir_with_svs_type_and_dynamic
    )
    compress = False
    if index_svs_type.startswith(("leanvec", "lvq")):
        if svs_type != index_svs_type:
            pytest.skip("Not supported")
    if not svs_type.startswith(("leanvec", "lvq")):
        if svs_type != index_svs_type:
            pytest.skip("Not supported")
    if svs_type != index_svs_type:
        if index_svs_type.startswith("int"):
            pytest.skip("Not supported")
        compress = True
    data_dir = index_dir / "data"
    loader = create_loader(svs_type, data_dir=data_dir, compress=compress)
    if svs_type.startswith("lvq"):
        assert isinstance(loader, svs.LVQLoader)
    elif svs_type.startswith("leanvec"):
        assert isinstance(loader, svs.LeanVecLoader)
    else:
        assert isinstance(loader, svs.VectorDataLoader)


def test_create_loader_compression_not_supported():
    with pytest.raises(ValueError, match="Compression is not supported"):
        create_loader("float32", data_dir=Path(), compress=True)


def test_create_loader_no_vecs_path_no_data_dir():
    with pytest.raises(
        ValueError, match="Either vecs_path or data_dir must be not None"
    ):
        create_loader("float32")


def test_create_loader_with_vecs_path(tmp_vecs):
    svs_type = SUFFIX_TO_SVS_TYPE[tmp_vecs.suffix]
    loader = create_loader(svs_type, vecs_path=tmp_vecs)
    assert isinstance(loader, svs.VectorDataLoader)


def test_create_loader_with_mismatched_vecs_path(tmp_fvecs):
    with pytest.raises(ValueError, match="Expected svs_type="):
        create_loader("float16", vecs_path=tmp_fvecs)


def test_create_loader_with_invalid_vecs_path():
    with pytest.raises(ValueError, match="Unknown suffix"):
        create_loader("float32", vecs_path=Path("data.invalid"))


def test_create_loader_invalid_svs_type():
    with pytest.raises(ValueError, match="Unknown svs_type="):
        create_loader("invalid_type")
