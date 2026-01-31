# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
import svs

import svsbench.build
from svsbench.consts import SUFFIX_TO_SVS_TYPE
from svsbench.generate_leanvec_matrices import (
    generate_leanvec_matrices,
    save_leanvec_matrices,
)


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


def test_main_with_train_query(tmp_path, tmp_vecs, query_path):
    if SUFFIX_TO_SVS_TYPE[tmp_vecs.suffix] == "float16":
        pytest.xfail("https://github.com/intel/ScalableVectorSearch/issues/93")
    svsbench.build.main(
        [
            "--vecs_file",
            str(tmp_vecs),
            "--svs_type",
            "leanvec4x8",
            "--train_query_file",
            str(query_path),
            "--out_dir",
            str(tmp_path),
            "--uncommitted",
        ]
    )


def test_main_with_matrices(tmp_path, tmp_vecs, query_path):
    if SUFFIX_TO_SVS_TYPE[tmp_vecs.suffix] == "float16":
        pytest.xfail("https://github.com/intel/ScalableVectorSearch/issues/93")
    (data_matrix, query_matrix), (leanvec_dims_effective, max_vectors_effective) = (
        generate_leanvec_matrices(
            tmp_vecs,
            query_path,
        )
    )
    data_matrix_path, query_matrix_path = save_leanvec_matrices(
        tmp_vecs,
        query_path,
        max_vectors_effective,
        leanvec_dims_effective,
        data_matrix,
        query_matrix,
        tmp_path,
    )
    svsbench.build.main(
        [
            "--vecs_file",
            str(tmp_vecs),
            "--svs_type",
            "leanvec4x8",
            "--data_matrix_file",
            str(data_matrix_path),
            "--query_matrix_file",
            str(query_matrix_path),
            "--out_dir",
            str(tmp_path),
            "--uncommitted",
        ]
    )
