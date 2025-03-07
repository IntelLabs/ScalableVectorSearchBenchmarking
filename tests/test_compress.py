# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest

import svsbench.compress


@pytest.mark.parametrize("dynamic", (True, False))
@pytest.mark.parametrize(
    "svs_type",
    [
        svs_type
        for svs_type in svsbench.consts.SVS_TYPES
        if svs_type.startswith(("leanvec", "lvq"))
    ],
)
def test_compress(
    svs_type, dynamic, index_dir_with_svs_type_and_dynamic, tmp_path
):
    index_dir, index_svs_type, index_dynamic = (
        index_dir_with_svs_type_and_dynamic
    )
    if index_svs_type.startswith(("leanvec", "lvq", "int")) or (
        index_dynamic and not dynamic
    ):
        pytest.skip("Not supported")

    svsbench.compress.compress(
        idx_dir=index_dir,
        svs_type=svs_type,
        out_dir=tmp_path,
        dynamic=dynamic,
        max_threads=1,
        load_from_static=not index_dynamic,
    )
