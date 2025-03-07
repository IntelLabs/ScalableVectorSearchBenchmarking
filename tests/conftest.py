import os
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import svs

import svsbench.consts

SVS_TYPE_TO_DTYPE: Final = {
    "float16": np.float16,
    "float32": np.float32,
}

INT_TO_LEANVEC_KIND: Final = {
    4: svs.LeanVecKind.lvq4,
    8: svs.LeanVecKind.lvq8,
}


def random_array(dtype: np.dtype) -> np.ndarray:
    rng = np.random.default_rng(42)
    if np.dtype(dtype).kind == "i":
        iinfo = np.iinfo(dtype)
        return rng.integers(iinfo.min, iinfo.max, (1000, 100), dtype=dtype)
    else:
        return rng.random((1000, 100)).astype(dtype)


@pytest.fixture(scope="session")
def tmp_fvecs(tmp_path_factory):
    fvecs_path = tmp_path_factory.mktemp("fvecs") / "random.fvecs"
    svs.write_vecs(random_array(np.float32), str(fvecs_path))
    return fvecs_path


@pytest.fixture(
    scope="session", params=svsbench.consts.SUFFIX_TO_SVS_TYPE.keys()
)
def tmp_vecs(request, tmp_path_factory):
    suffix = request.param
    vecs_path = tmp_path_factory.mktemp("vecs") / ("random" + suffix)
    svs.write_vecs(
        random_array(
            SVS_TYPE_TO_DTYPE[svsbench.consts.SUFFIX_TO_SVS_TYPE[suffix]]
        ),
        str(vecs_path),
    )
    return vecs_path


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(
            (svs_type, dynamic), marks=pytest.mark.xfail(raises=RuntimeError)
        )
        if svs_type in ("lvq8x8",) and dynamic
        else (svs_type, dynamic)
        for svs_type in svsbench.consts.SVS_TYPES
        for dynamic in (True, False)
    ],
    ids=lambda x: str(x),
)
def index_dir_with_svs_type_and_dynamic(request, tmp_path_factory):
    svs_type, dynamic = request.param
    if svs_type.startswith(("leanvec", "lvq")):
        data_type_initial = "float32"
    else:
        data_type_initial = svs_type
    parameters = svs.VamanaBuildParameters(
        graph_max_degree=16,
        window_size=10,
    )
    array = random_array(SVS_TYPE_TO_DTYPE[data_type_initial])
    if dynamic:
        index_class = svs.DynamicVamana
        index_initial = svs.DynamicVamana.build(
            parameters,
            array,
            np.arange(len(array), dtype=np.uint64),
            svs.DistanceType.L2,
            num_threads=1,
        )
    else:
        index_class = svs.Vamana
        index_initial = svs.Vamana.build(
            parameters, array, svs.DistanceType.L2
        )
    index_path_initial = tmp_path_factory.mktemp("index")
    index_initial.save(
        str(index_path_initial / "config"),
        str(index_path_initial / "graph"),
        str(index_path_initial / "data"),
    )
    if svs_type.startswith(("leanvec", "lvq")):
        if svs_type.startswith("leanvec"):
            primary, secondary = [
                INT_TO_LEANVEC_KIND[int(kind)]
                for kind in svs_type[len("leanvec") :].split("x")
            ]
            loader = svs.LeanVecLoader(
                svs.VectorDataLoader(str(index_path_initial / "data")),
                leanvec_dims=array.shape[1] // 4,
                primary_kind=primary,
                secondary_kind=secondary,
            )
        elif svs_type.startswith("lvq"):
            if svs_type == "lvq8":
                primary = 8
                residual = 0
            elif svs_type == "lvq4x4":
                primary = 4
                residual = 4
            elif svs_type == "lvq4x8":
                primary = 4
                residual = 8
            else:
                raise ValueError("Unknown svs_type=" + svs_type)
            loader = svs.LVQLoader(
                svs.VectorDataLoader(str(index_path_initial / "data")),
                primary=primary,
                residual=residual,
            )
        index = index_class(
            str(index_path_initial / "config"),
            str(index_path_initial / "graph"),
            loader,
        )
        index_path = tmp_path_factory.mktemp("index")
        index.save(
            str(index_path / "config"),
            str(index_path / "graph"),
            str(index_path / "data"),
        )
    else:
        index_path = index_path_initial
    np.save(str(index_path / "data.npy"), array)
    return index_path, svs_type, dynamic


@pytest.fixture(scope="session")
def query_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("query") / "query.fvecs"
    svs.write_vecs(
        np.random.default_rng(42).random((100, 100)).astype(np.float32), path
    )
    return path


@pytest.fixture(scope="session")
def distance() -> svs.DistanceType:
    return svs.DistanceType.L2


@pytest.fixture(scope="session")
def num_threads() -> int:
    return max(len(os.sched_getaffinity(0)) - 1, 1)


@pytest.fixture(scope="session")
def ground_truth_path(
    index_dir_with_svs_type_and_dynamic,
    query_path,
    distance,
    num_threads,
    tmp_path_factory,
) -> Path:
    index_dir, index_svs_type, index_dynamic = (
        index_dir_with_svs_type_and_dynamic
    )
    vectors = np.load(index_dir / "data.npy")
    index = svs.Flat(vectors, distance=distance, num_threads=num_threads)
    idxs, _ = index.search(svs.read_vecs(str(query_path)), 100)
    ground_truth_path = (
        tmp_path_factory.mktemp("ground_truth")
        / f"ground_truth_{index_svs_type}.ivecs"
    )
    svs.write_vecs(idxs.astype(np.uint32), ground_truth_path)
    return ground_truth_path
