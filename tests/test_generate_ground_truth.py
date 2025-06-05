# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import functools

import conftest
import pytest
import svs

from svsbench.generate_ground_truth import generate_ground_truth, main


def test_generate_ground_truth_no_shuffle(
    tmp_vecs, query_path, distance, num_threads, tmp_path_factory
):
    if tmp_vecs.suffix == ".hvecs":
        pytest.xfail("Not implemented")
    out_file = tmp_path_factory.mktemp("output") / "ground_truth.ivecs"
    k = 10
    generate_ground_truth(
        vecs_path=tmp_vecs,
        query_file=query_path,
        distance=distance,
        num_vectors=None,
        k=k,
        num_threads=num_threads,
        out_file=out_file,
        query_out_path=None,
        shuffle=False,
        seed=42,
    )
    assert out_file.is_file()
    gt = svs.read_vecs(str(out_file))
    assert gt.shape == (conftest.NUM_RANDOM_QUERY_VECTORS, k), (
        "Expected (num_queries, k) shape"
    )


def test_generate_ground_truth_shuffle(
    tmp_vecs, query_path, distance, num_threads, tmp_path_factory
):
    if tmp_vecs.suffix == ".hvecs":
        pytest.xfail("Not implemented")
    out_file = (
        tmp_path_factory.mktemp("output") / "ground_truth_shuffled.ivecs"
    )
    k = 5
    generate_ground_truth(
        vecs_path=tmp_vecs,
        query_file=query_path,
        distance=distance,
        num_vectors=500,
        k=k,
        num_threads=num_threads,
        out_file=out_file,
        shuffle=True,
        seed=2,
    )
    assert out_file.is_file()
    gt = svs.read_vecs(str(out_file))
    assert gt.shape == (conftest.NUM_RANDOM_QUERY_VECTORS, k)


def test_generate_ground_truth_num_query_vectors(
    tmp_vecs, query_path, distance, num_threads, tmp_path_factory
):
    k = 7
    if tmp_vecs.suffix == ".hvecs":
        pytest.xfail("Not implemented")
    out_dir = tmp_path_factory.mktemp("output")
    out_file = out_dir / "ground_truth_subqueries.ivecs"
    query_out_path = out_dir / "queries_out.fvecs"
    generate_ground_truth_partial = functools.partial(
        generate_ground_truth,
        vecs_path=tmp_vecs,
        query_file=query_path,
        distance=distance,
        k=k,
        num_threads=num_threads,
        out_file=out_file,
        query_out_path=query_out_path,
        shuffle=True,
    )
    for num_query_vectors in [20, 200]:
        generate_ground_truth_partial(num_query_vectors=num_query_vectors)
        gt = svs.read_vecs(str(out_file))
        new_queries = svs.read_vecs(str(query_out_path))
        assert gt.shape == (num_query_vectors, k)
        assert new_queries.shape == (
            num_query_vectors,
            conftest.RANDOM_VECTORS_SHAPE[1],
        )


def test_generate_ground_truth_main(
    tmp_vecs, query_path, num_threads, tmp_path_factory
):
    if tmp_vecs.suffix == ".hvecs":
        pytest.xfail("Not implemented")
    out_dir = tmp_path_factory.mktemp("cli")
    out_file = out_dir / "gt.ivecs"
    query_out_path = out_dir / "queries_out.fvecs"
    k = 8
    num_query_vectors = 150

    argv = [
        "--vecs_file",
        str(tmp_vecs),
        "--query_file",
        str(query_path),
        "--out_file",
        str(out_file),
        "--distance",
        "mip",
        "-k",
        str(k),
        "--max_threads",
        str(num_threads),
        "--seed",
        "42",
        "--num_query_vectors",
        str(num_query_vectors),
        "--query_out_file",
        str(query_out_path),
    ]

    main(argv)

    gt = svs.read_vecs(str(out_file))
    new_queries = svs.read_vecs(str(query_out_path))
    assert gt.shape == (num_query_vectors, k)
    assert new_queries.shape == (
        num_query_vectors,
        conftest.RANDOM_VECTORS_SHAPE[1],
    )
