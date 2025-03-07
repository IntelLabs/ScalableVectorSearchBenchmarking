import pytest
import svs

from svsbench.consts import SVS_TYPES
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
    search(
        idx_dir=index_dir,
        svs_type=svs_type,
        distance=svs.DistanceType.L2,
        compress=compress,
        ground_truth_path=ground_truth_path,
        query_path=query_path,
        static=static,
        load_from_static=not index_dynamic,
    )


def test_search_with_separate_data_dir():
    pytest.xfail("TODO: Implement")
