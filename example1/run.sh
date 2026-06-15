#!/usr/bin/env bash
# ============================================================
# llm_quant.py 启动脚本
# 自动从 config.yaml 读取 qnn_sdk_root，设置 LD_LIBRARY_PATH 后运行量化脚本
# 用法:
#   ./run.sh                # 用默认 python 运行
#   PYTHON=python3.10 ./run.sh   # 指定 python 解释器
# ============================================================
set -euo pipefail

# 脚本所在目录（与执行目录无关）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
PYTHON="${PYTHON:-python}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[run.sh] 错误: 找不到配置文件 ${CONFIG_FILE}" >&2
    exit 1
fi

# 从 config.yaml 读取 qnn_sdk_root
QNN_SDK_ROOT="$("${PYTHON}" -c "import yaml,sys; print(yaml.safe_load(open('${CONFIG_FILE}', encoding='utf-8'))['environment']['qnn_sdk_root'])")"

if [[ -z "${QNN_SDK_ROOT}" ]]; then
    echo "[run.sh] 错误: 未能从 config.yaml 读取 environment.qnn_sdk_root" >&2
    exit 1
fi

# 从 config.yaml 读取 output_dir，并把临时目录(TMPDIR)指到同一块(大)盘，
# 避免 3B 模型导出 ONNX 时把系统盘(/tmp)写满 -> "No space left on device"
OUTPUT_DIR="$("${PYTHON}" -c "import yaml; print(yaml.safe_load(open('${CONFIG_FILE}', encoding='utf-8'))['environment']['output_dir'])")"
if [[ -n "${OUTPUT_DIR}" ]]; then
    TMPDIR_PATH="${OUTPUT_DIR}/.tmp"
    mkdir -p "${TMPDIR_PATH}"
    export TMPDIR="${TMPDIR_PATH}"
    export TEMP="${TMPDIR_PATH}"
    export TMP="${TMPDIR_PATH}"
    echo "[run.sh] TMPDIR         = ${TMPDIR}"
fi

LIB_CLANG_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang"
if [[ ! -d "${LIB_CLANG_PATH}" ]]; then
    echo "[run.sh] 警告: 目录不存在 ${LIB_CLANG_PATH}，请确认 config.yaml 中的 qnn_sdk_root 是否正确" >&2
fi

# 关键: 在启动 python 之前设置 LD_LIBRARY_PATH，确保动态链接器能找到 libc++.so.1 等原生库
export LD_LIBRARY_PATH="${LIB_CLANG_PATH}:${LD_LIBRARY_PATH:-}"

# 自动在 SDK 里搜索 libc++.so* 所在目录，并加入 LD_LIBRARY_PATH（兼容库在其它子目录的情况）
LIBCXX_FILE="$(find "${QNN_SDK_ROOT}" -name "libc++.so*" -print -quit 2>/dev/null || true)"
if [[ -n "${LIBCXX_FILE}" ]]; then
    LIBCXX_DIR="$(dirname "${LIBCXX_FILE}")"
    export LD_LIBRARY_PATH="${LIBCXX_DIR}:${LD_LIBRARY_PATH}"
    echo "[run.sh] 在 SDK 中找到 libc++: ${LIBCXX_FILE}"
else
    echo "[run.sh] 警告: 在 SDK (${QNN_SDK_ROOT}) 内未找到 libc++.so*" >&2
    echo "[run.sh]        若启动报 'libc++.so.1: cannot open shared object file'，请安装系统库:" >&2
    echo "[run.sh]        apt-get update && apt-get install -y libc++1 libc++abi1" >&2
fi

echo "[run.sh] QNN_SDK_ROOT   = ${QNN_SDK_ROOT}"
echo "[run.sh] LD_LIBRARY_PATH= ${LD_LIBRARY_PATH}"
echo "[run.sh] 启动 llm_quant.py ..."

cd "${SCRIPT_DIR}"
exec "${PYTHON}" "${SCRIPT_DIR}/llm_quant.py" "$@"
