# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
import svs

from svsbench.build import build_dynamic, save
from svsbench.consts import SVS_TYPES
from svsbench.generate_ground_truth import generate_ground_truth
from svsbench.search import search


@pytest.mark.parametrize("static", (True, False))
@pytest.mark.parametrize("svs_type", SVS_TYPES)
def test_search(
    static,
    svs_type,
    index_dir_with_svs_type_and_dynamic,
    ground_truth_path,
    query_path,
):
    index_dir, index_svs_type, index_dynamic = (
        index_dir_with_svs_type_and_dynamic
    )
    if index_dynamic and static:
        pytest.xfail("Not implemented")
    compress = False
    if index_svs_type.startswith(("leanvec", "lvq")):
        if svs_type != index_svs_type:
            pytest.skip("Not supported")
    if not svs_type.startswith(("leanvec", "lvq")):
        if svs_type != index_svs_type:
            pytest.skip("Not supported")
    if svs_type != index_svs_type:
        compress = True
    _, _, recalls = search(
        idx_dir=index_dir,
        svs_type=svs_type,
        distance=svs.DistanceType.L2,
        compress=compress,
        ground_truth_path=ground_truth_path,
        query_path=query_path,
        static=static,
        load_from_static=not index_dynamic,
    )
    # Search parameters are calibrated to recall 0.9
    assert recalls[0] > 0.8


def test_search_with_separate_data_dir():
    pytest.xfail("TODO: Implement")


@pytest.mark.parametrize("tmp_vecs", [".fvecs"], indirect=True)
def test_search_with_shuffle(tmp_vecs, query_path, tmp_path):
    seed = 123
    svs_type = "float32"
    distance = svs.DistanceType.L2

    ground_truth_path = tmp_path / "ground_truth.ivecs"
    generate_ground_truth(
        vecs_path=tmp_vecs,
        query_file=query_path,
        distance=distance,
        out_file=ground_truth_path,
    )
    build_result = build_dynamic(
        vecs_path=tmp_vecs,
        svs_type=svs_type,
        distance=distance,
        shuffle=True,
        seed=seed,
    )
    idx_dir = save(build_result[0], tmp_path)
    _, _, recalls = search(
        idx_dir=idx_dir,
        svs_type=svs_type,
        distance=distance,
        ground_truth_path=ground_truth_path,
        query_path=query_path,
        shuffle=True,
        seed=seed,
    )
    # Search parameters are calibrated to recall 0.9
    assert recalls[0] > 0.8
