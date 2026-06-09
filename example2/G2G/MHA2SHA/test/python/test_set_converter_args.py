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
"""Tests Setting Converter Args"""

import pytest

from mha2sha.utils.arg_parser import (
    converter_arg_parser,
    ExplicitlySetArgumentParser,
    filter_args,
    flagify,
)
from mha2sha.utils.base_llm_configs import base_llms_to_flags


@pytest.fixture(autouse=True)
def clean_up_explicitly_set_args():
    # Clean the set args each test
    ExplicitlySetArgumentParser.set_args = set()
    yield


class TestSetConverterArgs:

    def test_python_args(self):
        before_args = base_llms_to_flags["llama2"]
        after_args = filter_args(**before_args)

        for before_arg, before_val in before_args.items():
            assert (
                before_val == after_args[before_arg]
            ), f"Arg shouldn't change after filtering. For key {before_arg} got: {before_val} instead of {after_args[before_arg]}"

    def test_llama2_python_args_explicit_overwrite(self):
        before_args = base_llms_to_flags["llama2"] | {"handle_rope_ops": False}
        after_args = filter_args(**before_args)
        assert not after_args[
            "handle_rope_ops"
        ], "'handle_rope_ops' should be changed to False"

    def test_llama2_base_llm_only(self):
        before_args = {"base_llm": "llama2"}
        after_args = filter_args(**before_args)

        llama2_flags = base_llms_to_flags["llama2"]

        for l_arg, l_val in llama2_flags.items():
            assert (
                after_args[l_arg] == l_val
            ), f"Arg '{l_arg}' should be overwritten from default value"

    def test_sys_args(self):
        parser = converter_arg_parser()
        sys_args = ["--exported-model-path", ""]
        for arg, val in base_llms_to_flags["llama2"].items():
            sys_args.extend([flagify(arg).replace("'", "")])

        before_args = vars(parser.parse_args(sys_args))
        for not_needed_arg in ["model_name", "sha_export_path", "exported_model_path"]:
            before_args.pop(not_needed_arg)
        after_args = filter_args(**before_args)

        for before_arg, before_val in before_args.items():
            assert (
                before_val == after_args[before_arg]
            ), f"Arg shouldn't change after filtering. For key {before_arg} got: {before_val} instead of {after_args[before_arg]}"

    def test_llama2_sys_args(self):
        parser = converter_arg_parser()
        sys_args = ["--exported-model-path", "", "--base-llm", "llama2"]
        before_args = vars(parser.parse_args(sys_args))

        for not_needed_arg in ["model_name", "sha_export_path", "exported_model_path"]:
            before_args.pop(not_needed_arg)
        after_args = filter_args(**before_args)

        for llama2_arg, llama2_val in base_llms_to_flags["llama2"].items():
            assert (
                llama2_val == after_args[llama2_arg]
            ), f"Arg shouldn't change after filtering. For key {llama2_arg} got: {llama2_val} instead of {after_args[llama2_arg]}"

    def test_llama2_sys_args_explicit_overwrite(self):
        parser = converter_arg_parser()
        sys_args = [
            "--exported-model-path",
            "",
            "--base-llm",
            "llama2",
            "--gqa-model",  # explicitly set!
        ]
        before_args = vars(parser.parse_args(sys_args))

        for not_needed_arg in ["model_name", "sha_export_path", "exported_model_path"]:
            before_args.pop(not_needed_arg)
        after_args = filter_args(**before_args)

        for llama2_arg, llama2_val in base_llms_to_flags["llama2"].items():
            check = llama2_val == after_args[llama2_arg]
            if llama2_arg == "gqa_model":
                check = not check
            assert (
                check
            ), f"Arg shouldn't change after filtering. For key {llama2_arg} got: {llama2_val} instead of {after_args[llama2_arg]}"
