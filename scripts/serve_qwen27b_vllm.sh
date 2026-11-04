#!/usr/bin/env bash
# =============================================================================
# Qwen3.8-27B (compressed-tensors W8A16) + MTP speculative decoding on vLLM
#
# 目标硬件: 单张 CMP 170HX (GA100 / sm_80 / 64GB / 190W / PCIe Gen2 x16)
# 已验证:   vLLM 0.27.1, 262144 ctx, MTP k=3 -> 51.4 tok/s decode (conc1)
#
# 用法:
#   DRY=1 ./serve_qwen27b_vllm.sh     # 只打印展开后的命令, 不启动、不碰任何服务
#   ./serve_qwen27b_vllm.sh           # 真正启动 (前台)
#
# 注意: 这张卡的算力被砍过, 解码受算力/功耗墙限制, 所以降比特省的是内存不是时间;
#      真正的加速来自投机解码 (本配置 MTP k=3, 接受长度 3.02).
#      27B 与 Flash-Next 都满载占卡, 64GB 只容得下一个, 切换必须先停另一个.
# =============================================================================
set -euo pipefail

# ---------------- 路径与可调参数 ----------------
APP_ROOT="${APP_ROOT:-/home/appuser}"
VENV="${VENV:-$APP_ROOT/vllm-env}"
MODEL="${MODEL:-$APP_ROOT/models/qwen38-heretic-w8a16}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B-heretic-ara-W8A16-MTP}"
MAX_LEN="${MAX_LEN:-262144}"
MAX_SEQS="${MAX_SEQS:-4}"
MTP_N="${MTP_N:-3}"
KV="${KV:-bfloat16}"
GPU_UTIL="${GPU_UTIL:-0.95}"
BATCH_TOKENS="${BATCH_TOKENS:-16384}"
PERF_MODE="${PERF_MODE:-throughput}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-NONE}"   # NONE = 压思考链 "!!!!" 重复的稳定性选择, 代价 5-12% 解码
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"   # 不要用 hermes 解析器
SAMPLE_CFG="${SAMPLE_CFG:-{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":1.5,\"repetition_penalty\":1.05}}"

ARGS=(
  serve "$MODEL"
  --host "$HOST" --port "$PORT"
  --served-model-name "$SERVED_NAME"
  --dtype bfloat16
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --kv-cache-dtype "$KV"
  --enable-chunked-prefill
  --max-num-batched-tokens "$BATCH_TOKENS"
  --long-prefill-token-threshold 0
  --performance-mode "$PERF_MODE"
  --mamba-cache-mode align
  --prefix-match-unit 16
  --enable-prefix-caching
  --trust-remote-code
  --reasoning-parser "$REASONING_PARSER"
  --enable-auto-tool-choice
  --tool-call-parser "$TOOL_CALL_PARSER"
  --limit-mm-per-prompt '{"image":1,"video":0}'
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}'
  --override-generation-config "$SAMPLE_CFG"
)

if [[ "$CUDAGRAPH_MODE" == "NONE" ]]; then
  ARGS+=(--compilation-config '{"cudagraph_mode":"NONE"}')
fi
if [[ "${MTP_N}" != "0" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_N}}")
fi

VLLM_BIN="$VENV/bin/vllm"

# ---------------- DRY 模式: 在一切破坏性动作之前退出 ----------------
if [[ "${DRY:-0}" == "1" ]]; then
  echo "# environment (export these yourself when running for real):"
  echo "#   VLLM_USE_FLASHINFER_SAMPLER=0"
  echo "#   VLLM_WORKER_MULTIPROC_METHOD=spawn"
  echo "#   VLLM_ENGINE_READY_TIMEOUT_S=1800"
  echo "#   CUDA_VISIBLE_DEVICES=0"
  printf '%q ' "$VLLM_BIN" "${ARGS[@]}"
  echo
  exit 0
fi

# ---------------- 真跑 ----------------
if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vllm not found at $VLLM_BIN (set VENV=...)" >&2
  exit 1
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 端口被占时不要硬起 (本机 :8080 常被前端网关占用 -> 换 PORT=8085 打点)
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "port ${PORT} already in use - pick another PORT= (e.g. 8085)" >&2
  exit 3
fi

exec "$VLLM_BIN" "${ARGS[@]}"
