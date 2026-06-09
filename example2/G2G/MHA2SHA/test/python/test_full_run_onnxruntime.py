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
"""Tests small LLM subgraphs with ONNXRuntime"""

import json
import os

from onnx.onnx_pb import ModelProto
from mha2sha.converter import MHA2SHAConverter


class TestFullRunONNXRuntime:

    def test_full_run(self, tmpdir, save_artifacts: bool, config_file: str):
        with open(config_file) as config_json:
            config = json.load(config_json)

        save_dir = os.path.abspath(
            os.path.join(".", f"test-artifacts/{config['model_name']}")
        )
        config["sha_export_path"] = save_dir if save_artifacts else tmpdir
        config["model_or_path"] = os.path.abspath(config["model_or_path"])
        config["exported_model_encoding_path"] = os.path.abspath(
            config["exported_model_encoding_path"]
        )
        converter = MHA2SHAConverter(
            config.pop("model_name"),
            config.pop("sha_export_path"),
            config.pop("model_or_path"),
            **config
        )

        mha2sha_model, verification_status = converter.convert()
        assert isinstance(mha2sha_model, ModelProto), f"Got: {type(mha2sha_model)}"
        assert verification_status, "Failed verification"
