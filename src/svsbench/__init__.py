# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Scalable Vector Search Benchmarking."""

from importlib.metadata import version

if __spec__ is None:
    raise RuntimeError("Running __init__.py directly is not supported.")
__version__ = version(__spec__.parent)
