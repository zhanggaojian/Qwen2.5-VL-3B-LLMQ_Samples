import os
import sys
import traceback
import subprocess
import concurrent.futures
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import time
from pathlib import Path
go_parallel = True

workfolder = os.getcwd()

LLAMA_MODELS = "/data/leozhen/quant_experiments/Qwen253bos"
QNN_SDK_ROOT = "/opt/qcom/aistack/qairt/2.36.0.250627-auto-qnx"

assert os.path.exists(QNN_SDK_ROOT) == True,"QNN_SDK_ROOT path does not exist"
assert os.path.exists(LLAMA_MODELS) == True,"LLAMA_MODELS path does not exist"
os.environ['QNN_SDK_ROOT'] = QNN_SDK_ROOT
sys.path.append(workfolder+'/../G2G')
sys.path.append(workfolder+'/../G2G/split_onnx_utils')
sys.path.append(workfolder+'/../../')
from utilities.nsptargets import NspTargets
from utilities.profiler import event_marker

nsp_target = NspTargets.Android.GEN4

CL = 2048 # 2048
ARNs = [1, 128]
EXPORT_AR = 1073
EXPORT_CONTEXT_LENGTH = 2048
onnx_name = f"qwen25llm"
# onnx_name = f"ConvertedModel_AR{EXPORT_AR}"
num_splits = 1

splits = range(1, num_splits+1)
arn_list = [ arn for arn in ARNs for i in splits ]
split_idxs = [i for arn in ARNs for i in splits]
print('All task list:', [f"ar{arn}-{n}" for arn,n in zip(arn_list,split_idxs)])

os.makedirs(f"{workfolder}/assets/models_ar_n", exist_ok=True)

import change_hardcoding
def gen_ar(arn):
    try:
        change_hardcoding.execute(
                f"{LLAMA_MODELS}",
                f"{workfolder}/assets/models_ar_n/ar{arn}-cl{CL}",
                [f" {EXPORT_AR},{arn}",f" -{EXPORT_AR},-1",f" {EXPORT_CONTEXT_LENGTH},{CL}",f" {EXPORT_CONTEXT_LENGTH-EXPORT_AR},{CL-arn}"]
                )
    except Exception as e:
        logger.error(traceback.format_exc())
        print(e)
        exit(0)

# gen_ar(1)
# assert 0
with event_marker(f'prepare-export'):
    with concurrent.futures.ProcessPoolExecutor(max_workers = len(ARNs) if go_parallel else 1) as executor:
        results = executor.map(gen_ar, ARNs)

print(f"Prepare AR128 AR1 export done.")
# ## Preprocess ONNX
# Prior to utilizing the QNN tool chain to compile and generate the context binary for LLaMA we need to split the model and generate the following artifacts
# - ONNX file for each split of the model
# - input vectors for each split
# - golden output vectors for each split

import os
import utils

qnn_env = os.environ.copy()
qnn_env["QNN_SDK_ROOT"] = QNN_SDK_ROOT
qnn_env["PYTHONPATH"] = QNN_SDK_ROOT + "/benchmarks/QNN/:" + QNN_SDK_ROOT + "/lib/python"
qnn_env["PATH"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang:" + qnn_env["PATH"]
qnn_env["LD_LIBRARY_PATH"] = QNN_SDK_ROOT + "/lib/x86_64-linux-clang"
qnn_env["HEXAGON_TOOLS_DIR"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang"
qnn_env["NUM_LAYERS_PER_SPLIT"] = "28"
qnn_env["LLM"] = "1"
qnn_env["split_embedding"] = "0"
qnn_env["split_lmhead"] = "0"
os.environ = qnn_env


# ### Split Onnx export
#
# This step splits a model into multiple parts based on the number of splits specified.
#
# Expected execution time: ~< 40 minutes


def thread_split(arn):
    try:
        name = f"ar{arn}-cl{CL}"
        model_export = f"{workfolder}/assets/models_ar_n"
        model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
        os.makedirs(model_artifact, exist_ok = True)

        # create symlink to export
        symlink_src = os.path.join(model_artifact, 'src')
        symlink_path = Path(symlink_src)
        if symlink_path.is_symlink():
            os.unlink(symlink_src)
        os.symlink(src = os.path.join(model_export, name), dst = symlink_src)

        os.makedirs(f"{model_artifact}/split_onnx", exist_ok = True)
        TEST_VECTOR_PICKLE_TYPE = "pkl"
        print(f"Starting {onnx_name}.onnx")
        utils.split_onnx(onnxfile = f"{model_artifact}/src/onnx/{onnx_name}.onnx", modelname = name,
                        pickle_filedir = os.path.join(model_export, f"ar{arn}-cl{CL}/test_vectors"),
                        num_splits = num_splits, output_dir = model_artifact, split_embedding = False,
                        encoding_file = f"{model_artifact}/src/onnx/{onnx_name}.encodings",using_qairt_workflow = True
                        )
        print(f"Ending {onnx_name}.onnx")
    except Exception as e:
        logger.error(traceback.format_exc())
        print(e)
        exit(0)

with event_marker(f'split-onnx'):
    with concurrent.futures.ProcessPoolExecutor(max_workers = len(ARNs) if go_parallel else 1) as executor:
        results = executor.map(thread_split, ARNs)

print(f"All onnx model splitted.")


# ### Convert attention layers from MHA to SHA
# The `mha2sha-onnx-converter` tool converts a model from MHA representation to its equivalent SHA representation.
# The encoding files generated from the AIMET workflow are provided as an input to this step via the `--exported-model-encoding-path` option.
# This step generates a new `.onnx` file that represents the model in SHA format.
# Expected execution time: ~60 minutes

mha2sha_root = workfolder+"/../G2G/MHA2SHA"
g2g_env = os.environ.copy()
g2g_env["PYTHONPATH"] = os.pathsep.join([g2g_env.get("PYTHONPATH", ""), os.path.join(mha2sha_root, "src/python")])
g2g_env["PATH"] = os.pathsep.join([g2g_env.get("PATH", ""), os.path.join(mha2sha_root, "bin")])
print(f"MHA2SHA tool root set to: {mha2sha_root}")

def thread_g2g(arn,split):
    try:
        # if split == 1:
        #     print("As first split only include embedding layer, so let's skip first split")
        #     return
        model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
        split_work_dir = os.path.join(model_artifact,f"{split}_of_{num_splits}")
        name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
        os.makedirs(split_work_dir, exist_ok = True)
        sha_folder = f"{split_work_dir}/sha_output/"
        os.makedirs(sha_folder, exist_ok = True)
        name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
        print(f"mha2sha-onnx-converter {name} running...")
        #it will reported error like this np.allclose() shape mismatch, no sure the reason know, but it do not affect genrated files
        #keep going
        args=["mha2sha-onnx-converter",
                            "--sha-export-path", sha_folder,
                            "--model-name", name,
                            "--exported-model-encoding-path", f"{model_artifact}/src/onnx/{onnx_name}.encodings",
                            "--exported-model-path", f"{model_artifact}/split_onnx/{name}.onnx",
                            # "--base-llm", "llama3",
                            "--llm-model",
                            "--handle-rope-ops",
                            "--handle-past-key-value",
                            "--mha-conv",
                            "--gqa-model",
                            "--nchw-aligned",
                            "--log-level","verbose"]
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=g2g_env)
        output, error = proc.communicate()
        print(output.decode(),error.decode())
        print(f"mha2sha-onnx-converter {name} done.")
    except Exception as e:
        logger.error(traceback.format_exc())
        print(e)
        exit(0)

with event_marker(f'mha2sha'):
    with concurrent.futures.ProcessPoolExecutor(max_workers = len(arn_list) if go_parallel else 1) as executor:
        results = executor.map(thread_g2g, arn_list, split_idxs)

print(f"All mha2sha convert done.")

# ## Convert the model from ONNX representation to QNN DLC representation
# The Qualcomm AI Engine Direct SDK `qairt-converter` tool converts a model from ONNX representation to its equivalent QNN DLC representation.
# The encoding files generated from the AIMET workflow are provided as an input to this step via the `–quantization_overrides model.encodings` option.
# This step generates a `.dlc` file that represents the model as a series of QNN API calls.
# Expected execution time: ~< 60 minutes

def thread_convert(arn,split):
    try:
        model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
        split_work_dir = os.path.join(model_artifact,f"{split}_of_{num_splits}")
        name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
        os.makedirs(split_work_dir, exist_ok = True)
        out_dir = os.path.join(split_work_dir, "converted_model")
        os.makedirs(out_dir, exist_ok = True)

        # create symlink to export
        for src in [f"input_list_{name}.txt",f"test_inputs_{name}"]:
            symlink_input = os.path.join(split_work_dir, src)
            symlink_path = Path(symlink_input)
            if symlink_path.is_symlink():
                os.unlink(symlink_input)
            os.symlink(src = os.path.join(model_artifact, src), dst = symlink_input)
        input_onnx=f"{split_work_dir}/sha_output/{name}.onnx"
        quantization_overrides= f"{split_work_dir}/sha_output/{name}.encodings"
        # if split != 1:
        #     input_onnx=f"{split_work_dir}/sha_output/{name}.onnx"
        #     quantization_overrides= f"{split_work_dir}/sha_output/{name}.encodings"
        # else:
            # mha2sha not applied to fisrt split
            # input_onnx = f"{model_artifact}/split_onnx/{name}.onnx"
            # quantization_overrides = f"{model_artifact}/src/onnx/{onnx_name}.encodings"
        args = [QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qairt-converter",
                        "--input_network", input_onnx,
                        "--quantization_overrides", quantization_overrides,
                        "-o", f'{out_dir}/{name}.dlc'
                        ]
        options = utils.get_input_layout(input_onnx, using_qairt_workflow = True)
        for entry in options:
            args+=entry

        proc = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE, env = qnn_env)
        output, error = proc.communicate()
        print(output.decode(), error.decode())
        print(f"qairt-converter {name} done!")
    except Exception as e:
        logger.error(traceback.format_exc())
        print(e)
        exit(0)

with event_marker(f'convert-onnx'):
    with concurrent.futures.ProcessPoolExecutor(max_workers = len(split_idxs) if go_parallel else 1) as executor:
        results = executor.map(thread_convert, arn_list, split_idxs)

print(f"All qairt-converter done.")

# ##  Quantized QNN DLC model
# The  Qualcomm AI Engine Direct SDK `qairt-quantizer` compiles the model `.dlc` and input`.raw` files into a `model.quantized.dlc` file.
# The inputs to this stage are the input raw files &  `model.dlc` generated in the previous step.
# Expected execution time: ~< 25 minutes


def thread_genlib(arn,split):
    try:
        model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
        split_work_dir = os.path.join(model_artifact,f"{split}_of_{num_splits}")
        name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
        os.chdir(split_work_dir)
        out_dir = os.path.join(split_work_dir,"compiled_model")
        os.makedirs( os.path.join(split_work_dir,"compiled_model"), exist_ok = True)

        float_dlc_file = os.path.join(split_work_dir, "converted_model", f'{name}.dlc')
        quantized_dlc_file = os.path.join(out_dir, f'{name}_quantized.dlc')
        ip_list_file = os.path.join(model_artifact, f'input_list_{name}.txt')

        proc = subprocess.Popen([QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qairt-quantizer",
                                "--input_dlc", float_dlc_file,
                                "--input_list", ip_list_file,
                                "--output_dlc", quantized_dlc_file,
                                "--act_bitwidth", "16",
                                "--bias_bitwidth", "32",
                                "--keep_weights_quantized"
                                ],stdout = subprocess.PIPE, stderr = subprocess.PIPE, env = qnn_env)
        """
        qairt-quantizer \
        --input_dlc /local/qwen2.5-VL-3B-os/example2/host_linux/assets/artifacts/ar1-cl2048/1_of_1/converted_model/ar1-cl2048_1_of_1.dlc \
        --input_list /local/qwen2.5-VL-3B-os/example2/host_linux/assets/artifacts/ar1-cl2048/input_list_ar1-cl2048_1_of_1.txt \
        --output_dlc /local/qwen2.5-VL-3B-os/example2/host_linux/assets/artifacts/ar1-cl2048/1_of_1/compiled_model/ar1-cl2048_1_of_1_quantized.dlc \
        --act_bitwidth 16 \
        --bias_bitwidth 32 \
        --keep_weights_quantized
        """
        output, error = proc.communicate()
        print(output.decode(), error.decode())
        print(f"qairt-quantizer {name} done!")
        os.chdir(workfolder)
    except Exception as e:
        print(e)
        exit(0)

with event_marker(f'qairt-quantizer'):
    with concurrent.futures.ProcessPoolExecutor(max_workers = len(split_idxs) if go_parallel else 1) as executor:
        results = executor.map(thread_genlib, arn_list, split_idxs)

print(f"All qairt-quantizer done.")

"""
qnn-context-binary-generator \
--backend libQnnHtp.so \
--model libQnnModelDlc.so \
--input_output_tensor_mem_type memhandle \
--output_dir /local/qwen2.5-VL-3B-os/example2/host_linux/assets \
--config_file /local/qwen2.5-VL-3B-os/example2/host_linux/htp_backend_ext_config_ar1.json \
--binary_file q25llm_weight_sharing_model_1_of_1.serialized \
--dlc_path /local/qwen2.5-VL-3B-os/example2/host_linux/assets/artifacts/ar1-cl2048/1_of_1/compiled_model/ar1-cl2048_1_of_1_quantized.dlc \
--log_level=debug

qnn-context-binary-generator \
--backend libQnnHtp.so \
--model libQnnModelDlc.so \
--input_output_tensor_mem_type memhandle \
--output_dir /local/qwen2.5-VL-3B-os/example2/host_linux/assets \
--config_file /local/qwen2.5-VL-3B-os/example2/host_linux/htp_backend_ext_config.json \
--binary_file q25llm_weight_sharing_model_1_of_1.serialized \
--dlc_path /local/qwen2.5-VL-3B-os/example2/host_linux/assets/artifacts/ar128-cl2048/1_of_1/compiled_model/ar128-cl2048_1_of_1_quantized.dlc,/local/qwen2.5-VL-3B-os/example2/host_linux/assets/artifacts/ar1-cl2048/1_of_1/compiled_model/ar1-cl2048_1_of_1_quantized.dlc \
--log_level=debug

"""


# ## QNN HTP weight sharing context binary
# The  Qualcomm AI Engine Direct SDK `qnn-context-binary-generator` tool creates a QNN context binary applicable to the QNN HTP backend.
# This binary can be deployed to run on a Snapdragon 8 Gen4 device that runs Android.
# This step requires the ar128 and ar1 quantized DLCs from the previous step and the `libQnnHtp.so` library, available in the Qualcomm AI Engine Direct SDK.
# Provide additional options that pertain to the QNN HTP backend by passing the `libQnnHtpBackendExtensions.so` library that implements extensions for the QNN HTP backend.
# The library is available in the Qualcomm AI Engine Direct SDK.
#
# ### Define Htp Perf Setting


# import os
# import json
#
# def make_config_file(index, folder, src_graphs, soc_id=72, dsp_arch="v81"):
#     htp_config_json = os.path.join(folder, f"HtpConfigFile_API_{index}.json")
#     perf_config_json = os.path.join(folder, f"PerfSetting_API_{index}.conf")
#
#     soc_id = int(soc_id)
#     with open(htp_config_json, 'w') as f:
#         config = {
#             "backend_extensions": {
#                 "shared_library_path": "libQnnHtpNetRunExtensions.so",
#                 "config_file_path": f"{perf_config_json}"
#             }
#         }
#
#         json.dump(config, f, indent=4)
#
#     with open(perf_config_json,'w') as f:
#         config = {
#             "graphs": [{
#                 "O": 3.0,
#                 "vtcm_mb": 8,
#                 "graph_names": src_graphs,
#                 "fp16_relaxed_precision": 0
#             }],
#             "devices": [
#                 {
#                     "soc_id": soc_id,
#                     "dsp_arch": dsp_arch,
#                     "cores": [
#                         {
#                             "perf_profile": "burst",
#                             "rpc_control_latency": 100
#                         }
#                     ],
#                     "pd_session": "unsigned"
#                 }
#             ],
#             "context": {
#                     "weight_sharing_enabled": len(src_graphs) > 1
#             },
#             "memory": {
#                     "mem_type": "shared_buffer"
#             }
#         }
#         json.dump(config, f, indent = 4)


# ### Compile context binary
# Expected execution time: ~60 minutes


# import subprocess

# soc_id = nsp_target.soc_id
# dsp_arch = nsp_target.dsp_arch
# soc_id = 39
# dsp_arch = 'v68'
#
# def thread_gen_ws_cb(i):
#     try:
#         ar128_src = f"{workfolder}/assets/artifacts/ar128-cl{CL}/"
#         ar1_src = f"{workfolder}/assets/artifacts/ar1-cl{CL}/"
#         output_dir = f"{workfolder}/assets/artifacts/ar128-ar1-cl{CL}_conf_files/"
#         ctx_output_dir = f"{workfolder}/assets/artifacts/ar128-ar1-cl{CL}/"
#         os.makedirs(output_dir, exist_ok = True)
#         os.makedirs(ctx_output_dir, exist_ok = True)
#
#         src1_split_folder = os.path.join(ar128_src, f"{i}_of_{num_splits}", "compiled_model")
#         src2_split_folder = os.path.join(ar1_src, f"{i}_of_{num_splits}", "compiled_model")
#
#         src1_graph_name = f"ar128-cl{CL}_{i}_of_{num_splits}"
#         src1_q_dlc = os.path.join(src1_split_folder, f"{src1_graph_name}_quantized.dlc")
#         src2_graph_name = f"ar1-cl{CL}_{i}_of_{num_splits}"
#         src2_q_dlc = os.path.join(src2_split_folder, f"{src2_graph_name}_quantized.dlc")
#
#         graph_list = [src1_graph_name, src2_graph_name]
#         make_config_file(i, output_dir, graph_list, soc_id, dsp_arch)
#
#         cmd = ["qnn-context-binary-generator",
#                 "--log_level=verbose",
#                 "--backend","libQnnHtp.so",
#                 "--model", "libQnnModelDlc.so",
#                 "--input_output_tensor_mem_type", "memhandle",
#                 "--output_dir", ctx_output_dir,
#                 "--config_file",f"{output_dir}/HtpConfigFile_API_{i}.json",
#                 "--binary_file", f"weight_sharing_model_{i}_of_{num_splits}.serialized",
#                 "--dlc_path", f"{src1_q_dlc},{src2_q_dlc}"]
#         proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=qnn_env)
#         output, error = proc.communicate()
#         print(output.decode(), error.decode())
#         print(f'#{i} weight sharing model generated')
#     except Exception as e:
#         print(e)
#         exit(0)
#
# with event_marker(f'gen-binary'):
#     with concurrent.futures.ProcessPoolExecutor(max_workers = len(splits) if go_parallel else 1) as executor:
#         results = executor.map(thread_gen_ws_cb, splits)
#
# print(f"All weight shared qnn-context-binary generated.")

# ### Save profiling stats

# from utilities.profiler import EventProfiler
# EventProfiler().report()
# EventProfiler().json_dump(os.path.join(workfolder, 'assets/profiling_stats.json'))
# print("finished.")
# Upon completion of these steps to prepare GQA models for inference, QNN context binaries  are available in `./assets/artifacts`.
# The next step is to execute the prepared models (now represented as serialized context binaries)on a Snapdragon 8 Gen4 Android device using executable utilities available in the Qualcomm AI Engine Direct SDK.
#
#
# Copyright (c) 2024 Qualcomm Technologies, Inc. and/or its subsidiaries.
