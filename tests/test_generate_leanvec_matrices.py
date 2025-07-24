# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import svsbench.generate_leanvec_matrices


def test_main(tmp_path, tmp_vecs, query_path):
    svsbench.generate_leanvec_matrices.main(
        tmp_vecs,
        query_path,
        out_dir=tmp_path,
    )
