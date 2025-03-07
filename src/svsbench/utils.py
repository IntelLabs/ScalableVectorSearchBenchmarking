import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from warnings import warn

import svs

from . import consts


class JSONFormatter(logging.Formatter):
    """Formatter that dumps msg and created as JSON."""

    def format(self, record):
        return json.dumps(
            (
                record.msg,
                datetime.fromtimestamp(record.created)
                .astimezone()
                .isoformat(),
            ),
            default=str,
        )


def configure_logger(
    logger: logging.Logger, log_dir: Path, stderr: bool = True
) -> Path:
    """Configure logger."""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (
        Path(logger.name).name
        + "-"
        + datetime.now().astimezone().isoformat()
        + ".log"
    )
    handler = logging.FileHandler(log_file)
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if stderr:
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.setLevel(logging.INFO)
    return log_file


def read_free_huge_pages() -> list[int]:
    """Read the number of free huge pages from the kernel."""
    huge_page_sizes = []
    for node in Path("/sys/devices/system/node/").iterdir():
        if not node.name.startswith("node"):
            continue
        huge_page_sizes.append(
            int(
                (
                    node / "hugepages/hugepages-1048576kB/free_hugepages"
                ).read_text()
            )
        )
    return huge_page_sizes


def read_system_config() -> dict[str, Any]:
    """Read system information."""
    return {
        "system_config": {
            "no_turbo": int(
                Path(
                    "/sys/devices/system/cpu/intel_pstate/no_turbo"
                ).read_text()
            ),
            "governor": Path(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            )
            .read_text()
            .strip(),
            "free_huge_pages": read_free_huge_pages(),
        }
    }


def log_to_dict(log_path: Path) -> dict[str, Any]:
    """Create a dict from a log."""
    no_turbo_all = []
    huge_pages_all = []
    timestamp_all = []
    repetition_all = []
    qps_all = []
    p95_all = []
    batch_size_all = []
    data_type_all = []
    count_all = []
    max_threads_all = []
    recall_all = []
    library_all = []
    index_all = []
    for line in log_path.read_text().splitlines():
        message, timestamp_str = json.loads(line)
        timestamp = datetime.fromisoformat(timestamp_str)
        message_key, message_value = next(iter(message.items()))
        match message_key:
            case "argv":
                pass
            case "system_config":
                no_turbo = bool(message_value["no_turbo"])
                huge_pages = any(
                    bool(num) for num in message_value["free_huge_pages"]
                )
            case "search_args":
                if "efs" in message_value:
                    library = "hnswlib"
                    data_type = "float32"
                    count = message_value["k"]
                    index = message_value["idx_file"]
                else:
                    library = "svs"
                    data_type = message_value["data_type"]
                    count = message_value["count"]
                    index = message_value["idx_dir"]
                max_threads = message_value["max_threads"]
            case "search_results":
                for repetition, (qps, p95) in enumerate(
                    zip(
                        message_value["qps"], message_value["p95"], strict=True
                    )
                ):
                    no_turbo_all.append(no_turbo)
                    huge_pages_all.append(huge_pages)
                    timestamp_all.append(timestamp)
                    qps_all.append(qps)
                    p95_all.append(p95)
                    repetition_all.append(repetition)
                    batch_size_all.append(message_value["batch_size"])
                    data_type_all.append(data_type)
                    count_all.append(count)
                    max_threads_all.append(max_threads)
                    recall_all.append(message_value["recall"])
                    library_all.append(library)
                    index_all.append(index)
            case "calibration_results":
                pass
            case _:
                warn(f"Ignoring unknown log message: {message}")
    return {
        "no_turbo": no_turbo_all,
        "huge_pages": huge_pages_all,
        "timestamp": timestamp_all,
        "repetitions": repetition_all,
        "qps": qps_all,
        "p95": p95_all,
        "batch_size": batch_size_all,
        "data_type": data_type_all,
        "count": count_all,
        "max_threads": max_threads_all,
        "recall": recall_all,
        "library": library_all,
        "index": index_all,
    }


def logs_to_dicts(
    log_paths: list[Path], huge_pages: bool | None = None
) -> list[dict[str, Any]]:
    """Find log files referenced in logs and create dicts."""
    results = []
    for log_path in log_paths:
        for line in log_path.read_text().splitlines():
            if line.startswith("+ tee"):
                out_path = Path(line.split()[2])
                found_log = False
                for out_line in out_path.read_text().splitlines():
                    if not found_log:
                        if out_line.startswith("Logging to"):
                            found_log = True
                    else:
                        results_dict = log_to_dict(Path(out_line))
                        if huge_pages is not None:
                            results_dict["huge_pages"] = [huge_pages] * len(
                                results_dict["huge_pages"]
                            )
                        results.append(results_dict)
                        break
    return results


def add_common_arguments(parser):
    """Add common command line arguments to a parser."""
    parser.add_argument("--log_dir", help="Log dir", type=Path)
    parser.add_argument(
        "--max_threads",
        help="Maximum number of threads",
        default=max(len(os.sched_getaffinity(0)) - 1, 1),
        type=int,
    )
    parser.add_argument(
        "--out_dir", help="Output dir", type=Path, default="out"
    )
    parser.add_argument("--seed", help="Random seed", default=42, type=int)
    parser.add_argument(
        "--svs_type",
        help="SVS type",
        choices=consts.SVS_TYPES,
        default="float32",
    )


def ground_truth_path(
    vecs_path: Path,
    queries_file: Path,
    distance: svs.DistanceType,
    num_vectors: int | None,
    seed: int | None = None,
) -> Path:
    num_vectors_part = (
        f"_first{num_vectors}" if num_vectors is not None else ""
    )
    return vecs_path.parent / "_".join(
        (
            "gt",
            vecs_path.name + num_vectors_part,
            queries_file.name,
            consts.DISTANCE_TO_STR[distance],
            f"{seed if seed is not None else False}.ivecs",
        )
    )
