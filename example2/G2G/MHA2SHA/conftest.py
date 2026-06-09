# -----------------------------------------------------------------------------
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2022-24 Qualcomm Technologies, Inc. All rights reserved.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
# -----------------------------------------------------------------------------
from typing import List
import os
import glob

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--filtered-configs",
        action="store",
        help="Run only the models specified in comma separated format",
    )
    parser.addoption(
        "--save-artifacts",
        action="store_true",
        default=False,
        help="By default, the test artifacts are saved to temp directory, this will save them in 'test-artifacts'",
    )


@pytest.fixture
def save_artifacts(request):
    return request.config.getoption("--save-artifacts")


def pytest_collection_modifyitems(config, items):
    # If the user is at the root dir, no need to run all tests. Just full flow.
    if "/".join(os.path.normpath(os.getcwd()).split("/")[-3:]) == "mha2sha/python/test":
        selected_items = []
        deselected_items = []
        for item in items:
            if "test_full_run_onnxruntime.py" in item.nodeid:
                selected_items.append(item)
            else:
                deselected_items.append(item)
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items


def pytest_generate_tests(metafunc):
    if "test_full_run_onnxruntime" == metafunc.definition.module.__name__:
        config_files = glob.glob(os.path.abspath("./configs/*.json"))
        if filtered_configs := metafunc.config.option.filtered_configs:
            config_files = get_selected_config_files(filtered_configs, config_files)
        metafunc.parametrize("config_file", config_files)


def get_selected_config_files(filtered_config_files: str, config_files: List[str]):
    filters = [f.strip().lower() for f in filtered_config_files.split(",")]
    base_config_filenames = [os.path.basename(c).split(".")[0] for c in config_files]
    selected_config_files = [
        c
        for c, b in zip(config_files, base_config_filenames)
        if any(b.startswith(f) for f in filters)
    ]

    if not selected_config_files:
        raise ValueError(
            f"No models found for the given filter(s): {filtered_config_files}\n"
            f"Available filters are: {', '.join(base_config_filenames)}"
        )
    return selected_config_files
