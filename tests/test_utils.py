import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import svs

import svsbench.consts
from svsbench.utils import (
    JSONFormatter,
    add_common_arguments,
    configure_logger,
    ground_truth_path,
    log_to_dict,
    logs_to_dicts,
    read_free_huge_pages,
    read_system_config,
)


def test_json_formatter():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    msg, created = json.loads(formatted)
    assert msg == "test message"
    assert isinstance(datetime.fromisoformat(created), datetime)


def test_configure_logger(tmp_path):
    logger = logging.getLogger("test_logger")
    log_dir = tmp_path / "logs"
    log_file = configure_logger(logger, log_dir)
    assert log_file.exists()
    assert log_file.parent == log_dir


def test_read_free_huge_pages():
    with (
        patch(
            "pathlib.Path.iterdir",
            return_value=[Path("node0"), Path("notnode")],
        ),
        patch("pathlib.Path.read_text", return_value="10"),
    ):
        huge_pages = read_free_huge_pages()
        assert huge_pages == [10]


def test_read_system_config():
    with (
        patch("pathlib.Path.read_text", return_value="1"),
        patch("svsbench.utils.read_free_huge_pages", return_value=[10]),
    ):
        config = read_system_config()
        assert config["system_config"]["no_turbo"] == 1
        assert config["system_config"]["governor"] == "1"
        assert config["system_config"]["free_huge_pages"] == [10]


def test_log_to_dict(tmp_path):
    pytest.skip("Not implemented yet")
    log_path = tmp_path / "log.json"
    log_data = json.dumps(
        (
            {
                "search_results": {
                    "qps": [1],
                    "p95": [2],
                    "batch_size": 32,
                    "recall": 0.9,
                }
            },
            datetime.now().isoformat(),
        )
    )
    log_path.write_text(log_data)
    result = log_to_dict(log_path)
    assert result["qps"] == [1]
    assert result["p95"] == [2]
    assert result["batch_size"] == [32]
    assert result["recall"] == [0.9]


def test_logs_to_dicts(tmp_path):
    pytest.skip("Not implemented yet")
    log_path = tmp_path / "log.txt"
    log_data = "+ tee log.json\nLogging to log.json"
    log_path.write_text(log_data)
    log_json_path = tmp_path / "log.json"
    log_json_data = json.dumps(
        (
            {
                "search_results": {
                    "qps": [1],
                    "p95": [2],
                    "batch_size": 32,
                    "recall": 0.9,
                }
            },
            datetime.now().isoformat(),
        )
    )
    log_json_path.write_text(log_json_data)
    result = logs_to_dicts([log_path])
    assert result[0]["qps"] == [1]
    assert result[0]["p95"] == [2]
    assert result[0]["batch_size"] == [32]
    assert result[0]["recall"] == [0.9]


def test_add_common_arguments():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args([])
    assert args.log_dir is None
    assert isinstance(args.max_threads, int)
    assert isinstance(args.out_dir, Path)
    assert isinstance(args.seed, int)
    assert args.svs_type in svsbench.consts.SVS_TYPES


def test_ground_truth_path():
    vecs_path = Path("vecs.fvecs")
    query_file = Path("query.fvecs")
    distance = svs.DistanceType.L2
    num_vectors = 100
    seed = 42
    result = ground_truth_path(
        vecs_path, query_file, distance, num_vectors, seed
    )
    assert isinstance(result, Path)
