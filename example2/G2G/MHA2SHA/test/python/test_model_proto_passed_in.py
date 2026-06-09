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
"""Tests small LLM subgraphs with ONNXRuntime - ModelProto passed in"""

import json
import os

import onnx
from mha2sha.converter import MHA2SHAConverter


class TestModelProtoPassedIn:

    def test_model_proto_converter(self, tmpdir):

        test_dir = os.path.dirname(__file__)
        config_file = os.path.join(test_dir, "configs/ss-llama-mha-kv.json")

        with open(config_file) as config_json:
            config = json.load(config_json)

        config["sha_export_path"] = tmpdir
        config["model_or_path"] = os.path.join(test_dir, config["model_or_path"])
        config["exported_model_encoding_path"] = os.path.join(
            test_dir,
            config["exported_model_encoding_path"]
        )

        config["model_or_path"] = onnx.load(config["model_or_path"])
        converter = MHA2SHAConverter(
            config.pop("model_name"),
            config.pop("sha_export_path"),
            config.pop("model_or_path"),
            config.pop("log_level"),
            **config
        )

        converter.convert()
