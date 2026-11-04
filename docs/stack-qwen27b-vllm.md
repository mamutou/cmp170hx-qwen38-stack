# vLLM 栈：Qwen3.8-27B 稠密（W8A16 + MTP）

## 1. 权重

| 项 | 值 |
|---|---|
| 基座 | Qwen3.8-27B（heretic-ara 消融版），`Qwen3_5ForConditionalGeneration`（带视觉塔） |
| 量化 | **compressed-tensors pack-quantized W8A16**：权重 int8、group 128、`mode: affine`，视觉塔整体不量化（`ignore` 里 308 条） |
| 体积 | 30GB（含 bf16 embedding/lm_head 与视觉塔） |
| MTP | 权重里带 `model_mtp.safetensors`（15 个张量：`mtp.fc` / `mtp.layers.0` / `pre_norm`），投机不需要额外文件 |
| 上下文 | 262144 |

GA100 是 sm_80：**没有 FP8/FP4 单元**，所以 8bit 只能走 int8（W8A16/W8A8），4bit 走 AWQ/GPTQ/Marlin。W8A16 是这张卡"质量优先"的现实上限档。

## 2. 启动参数逐项理由

```bash
vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8080 --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 --max-model-len 262144 --max-num-seqs 4 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype bfloat16 \
  --enable-chunked-prefill --max-num-batched-tokens 16384 \
  --performance-mode throughput --mamba-cache-mode align --prefix-match-unit 16 \
  --enable-prefix-caching --trust-remote-code \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.05}' \
  --compilation-config '{"cudagraph_mode":"NONE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

| 参数 | 为什么是它 |
|---|---|
| `--reasoning-parser qwen3` | 把 ` thinking` 拆成 `reasoning`，否则思考段混进正文 |
| `--tool-call-parser qwen3_xml` | Qwen 的工具调用是 XML 风格；**不要用 hermes 解析器**，会工具丢失或 `</think>` 后正文为空 |
| `--enable-auto-tool-choice` | OpenAI 客户端 `tool_choice: auto` 才生效 |
| `--override-generation-config` | 官方 thinking 采样（temp 0.6 / top_p 0.95 / top_k 20）+ 抗重复（presence 1.5 / repetition 1.05）；权重自带的 `temperature=1.0` 偏热，思考段容易刷 `!!!!` |
| `--mamba-cache-mode align` + `--prefix-match-unit 16` | 混合 GDN/Mamba 层的缓存对齐；前缀缓存命中率靠它 |
| `--kv-cache-dtype bfloat16` | 该架构的上限；fp8 KV 在部分新式注意力上会被直接拒绝，q8 KV 在深上下文反而更慢，必须实测 |
| `--gpu-memory-utilization 0.95` | 自动 profile 分配 KV。**固定 `--kv-cache-memory-bytes 30G` 在 batch=32768 下会 OOM**（峰值 activation 更大） |
| `--max-num-batched-tokens 16384` | 与 32768 相比 KV 余量更好；预填 chunk 再大只是拿显存换一点点吞吐 |
| `--max-num-seqs 4` | 4 路是这张卡不把单路压垮的上限；单人流场景设 1 更省 KV |
| `--compilation-config cudagraph_mode=NONE` | **为压思考链 `!!!!` 重复保留的稳定性开关**，代价约 5–12% 解码。想开 FULL 必须先用固定题集复验重复率 |
| `--speculative-config mtp / 3` | MTP 深度 3：本卡多次 A/B 下 n=1 更慢、n=3 思考最快；深度必须自己扫 |

## 3. 实测

同场打点（1753 token 提示 / 256 token 生成 / 两轮取二轮）：

| 并发 | 解码/路 | 解码合计 | 预填 | TTFT |
|---|---|---|---|---|
| 1 | **51.4 tok/s** | 51.4 | 1835 tok/s | 0.999 s |
| 4 | 52.2 tok/s | **202.7 tok/s** | 2536 tok/s | 0.714 s |

投机（MTP k=3）：接受长度 **3.02**，逐位置接受率 **0.855 / 0.651 / 0.518**，平均 67.5%。

历史对照（同机、不同配置，仅供趋势）：

| 权重/配置 | 解码 |
|---|---|
| bf16 | 22.6 tok/s |
| W8A8 int8 | 23.6 tok/s |
| **W8A16 + MTP 3** | **51.4 tok/s** |

读法：**并发不提高总吞吐的天花板**在这张卡上成立——4 路合计 202.7 = 每路 52.2 × 4，看着像线性，但那是 4 条短请求摊薄了排队；一旦每条都很长，合计会回落到接近单路（算力饱和，每多一个 token 就多一份算力开销）。

## 4. 运维入口

```bash
systemctl --user status  vllm-heretic.service
systemctl --user restart vllm-heretic.service     # 改脚本后
systemctl --user disable --now vllm-heretic.service   # 与 Flash-Next 二选一
```

⚠️ **端口冲突**：前端网关 `gpu-orch` 自己也监听 `:8080`。两者同时起会有一个 `EADDRINUSE`。规则：网关在跑时，vLLM 档要么停网关，要么换端口（打点时用 8085，别动网关——它有 cron 每 5 分钟拉回来）。

## 5. 还能怎么提（诚实排序）

| 动作 | 预期 | 状态 |
|---|---|---|
| **int4 W4A16（19.6GB）+ MTP 3** | 每步少读 ~10GB → 65–80 tok/s | 待测（权重要下 19.6GB；HF 上现成的 W4A16 / GPTQ-Int4 仓库都保留了 15 个 MTP 张量） |
| MTP 深度扫 3/4 | 第 4 位接受率还有 0.518，可能是正收益 | 待测（零成本） |
| CUDAGRAPH 开 FULL | +5~12% | 待测（需复验 `!!!!` 重复） |
| DFlash 草稿 | **实测 −20~28%，否决** | 已完成，见 [optimization-log.md](optimization-log.md#4-dflash-草稿实测否决2026-10-24) |
| 4 卡 TP | 解码不随卡数线性提升（带宽/算力按卡分），且本机只装 1 张 | 不做 |
| 换引擎（SGLang/Strata） | 见 [optimization-log.md](optimization-log.md#6-引擎选型结论) | 待评估 |
