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
        pytest.skip("Not supported")
    svsbench.build.build_static(
        vecs_path=tmp_vecs,
        svs_type=svs_type,
        distance=svs.DistanceType.L2,
    )

@pytest.mark.parametrize("svs_type", svsbench.consts.SVS_TYPES)
def test_build_dynamic(svs_type, tmp_vecs):
    if (
        svs_type.startswith("float")
        and svs_type != SUFFIX_TO_SVS_TYPE[tmp_vecs.suffix]
    ):
        pytest.skip("Not supported")
    if SUFFIX_TO_SVS_TYPE[tmp_vecs.suffix] == "float16":
        pytest.xfail("https://github.com/intel/ScalableVectorSearch/issues/93")
    svsbench.build.build_dynamic(
        vecs_path=tmp_vecs,
        svs_type=svs_type,
        distance=svs.DistanceType.L2,
    )
