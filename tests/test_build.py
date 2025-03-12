# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
import svs

import svsbench.build
from svsbench.consts import SUFFIX_TO_SVS_TYPE


@pytest.mark.parametrize("svs_type", svsbench.consts.SVS_TYPES)
def test_build_static(svs_type, tmp_vecs):
    if (
        svs_type.startswith("float")
        and svs_type != SUFFIX_TO_SVS_TYPE[tmp_vecs.suffix]
    ):
        with pytest.raises(ValueError, match="Expected svs_type"):
            svsbench.build.build_static(
                vecs_path=tmp_vecs,
                svs_type=svs_type,
                distance=svs.DistanceType.L2,
            )
    else:
        svsbench.build.build_static(
            vecs_path=tmp_vecs,
            svs_type=svs_type,
            distance=svs.DistanceType.L2,
        )

@pytest.mark.parametrize("svs_type", svsbench.consts.SVS_TYPES)
def test_build_dynamic(svs_type, tmp_vecs):
    if SUFFIX_TO_SVS_TYPE[tmp_vecs.suffix] == "float16":
        pytest.xfail("https://github.com/intel/ScalableVectorSearch/issues/93")
    if (
        svs_type.startswith("float")
        and svs_type != SUFFIX_TO_SVS_TYPE[tmp_vecs.suffix]
    ):
        with pytest.raises(ValueError, match="Expected svs_type"):
            svsbench.build.build_dynamic(
                vecs_path=tmp_vecs,
                svs_type=svs_type,
                distance=svs.DistanceType.L2,
            )
    svsbench.build.build_dynamic(
        vecs_path=tmp_vecs,
        svs_type=svs_type,
        distance=svs.DistanceType.L2,
        convert_vecs=True,
    )
