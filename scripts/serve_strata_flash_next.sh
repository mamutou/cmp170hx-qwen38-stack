#!/usr/bin/env bash
# =============================================================================
# Qwen3.8-Flash-Next (abliterated) on the Strata engine — 生产主力档
#
# 目标硬件: 单张 CMP 170HX (GA100 / sm_80 / 64GB / 190W)
# 已验证:   Strata 0.2.2 (build 0.1.47), UD-IQ4_XS pack + MTP spec 4, 262K ctx
#           解码 p50 85 tok/s (p90 102.5 / max 135.9, n=5682 次生产请求)
#           专家缓存命中 p50 99.2%
#
# 为什么 176B 能进 64GB:
#   125B 总参 / ~6B 激活 (主分片) + 51B lookup 表 (PLE, 按行查, 走 SSD/内存)
#   + 4B MTP head。判装不装只看"常驻矩阵乘权重 + KV", 不用总参数量。
#
# 用法:
#   DRY=1 ./serve_strata_flash_next.sh    # 只打印展开后的命令
#   ./serve_strata_flash_next.sh          # 真跑 (前台)
#
# 注意:
#   - `--pack` 必须是引擎指纹校验通过的 pack; 任意 GGUF 塞 --model-path 会报
#     "this is not a model size Strata knows"。自打包需要 50-60GB 空闲内存。
#   - 引擎子进程名是 strata-host, 停服务时按实际进程名兜底, 否则显存不释放。
#   - parallel>1 会为 KV 缩小 GPU 专家缓存 (单请求 -9%), 单人长上下文用 1。
# =============================================================================
set -euo pipefail

APP_ROOT="${APP_ROOT:-/home/appuser}"
STRATA_HOME="${STRATA_HOME:-$APP_ROOT/strata22}"
STRATA_BIN="${STRATA_BIN:-$STRATA_HOME/ctl/bin/strata}"
PACK="${PACK:-packs/hui-iq4xs}"
NATIVE_SHARD="${NATIVE_SHARD:-$APP_ROOT/models/hh-fn-v2-udiq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
PLE_SHARD="${PLE_SHARD:-$APP_ROOT/models/hh-fn-v2-udiq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf}"
EXPERT_PROFILE="${EXPERT_PROFILE:-data/expert-profile.bin}"
PORT="${PORT:-18080}"
HOST="${HOST:-127.0.0.1}"
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
SPEC_DEPTH="${SPEC_DEPTH:-4}"          # 本卡实测 n=2/3/4 -> 75.4/79.9/85.3 (代码题); n=5 显存吃紧
KV="${KV:-int8}"
PREFILL="${PREFILL:-16384}"

ARGS=(
  serve
  --pack "$PACK"
  --native "$NATIVE_SHARD"
  --ple-gguf "$PLE_SHARD"
  --expert-profile "$EXPERT_PROFILE"
  --expert-cache auto
  --prefill "$PREFILL"
  --spec "$SPEC_DEPTH"
  --mtp mtp/rt
  --spec-min-p 0.5
  --max-context "$MAX_CONTEXT"
  --kv "$KV"
  --kv-grow
  --resident-experts     # 专家转文件映射: 腾出 ~37GB 内存给共租负载, 解码 -0.7%, 冷启 36->52s
  --adapt-async 1
  --port "$PORT"
)

# ---------------- DRY 模式 ----------------
if [[ "${DRY:-0}" == "1" ]]; then
  echo "# cwd: $STRATA_HOME   (Strata 的 --pack/--mtp 相对路径都以 cwd 为基准)"
  printf '%q ' "$STRATA_BIN" "${ARGS[@]}"
  echo
  exit 0
fi

cd "$STRATA_HOME"
if [[ ! -x "$STRATA_BIN" ]]; then
  echo "strata not found at $STRATA_BIN (set STRATA_BIN=...)" >&2
  exit 1
fi
if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "port ${PORT} already in use" >&2
  exit 3
fi

exec "$STRATA_BIN" "${ARGS[@]}"
