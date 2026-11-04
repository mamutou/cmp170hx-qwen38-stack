# 测量方法学与原始数字

## 口径总则

1. **同时段 A/B 交替**：同一小时内交替跑两套配置。跨时段比较无效——持续负载下同配置自身就会降 14–43%。
2. **两轮取第二轮**：首轮含 kernel 编译/缓存冷启（Triton JIT、Metal 编译、prefix cache 空），只信第二轮。
3. **只信服务端统计行**：vLLM 看 `Avg generation throughput` 与 `SpecDecoding metrics`；Strata 看 `[strata] done: N tokens in T s (X tok/s) ... expert cache Y% hit`。不要用客户端 chunk 数估 token。
4. **报 tok/s 必须同时报"生成了多少 token"和"提示有多长"**：解码随生成长度下滑，预填随提示长度下滑，只报一个数没有可比性。
5. **TTFT 口径**：预填速率 = `prompt_tokens / TTFT`（要求请求带唯一前缀，否则前缀缓存会把它算成 0.0s）。

## 打点器

`scripts/bench_concurrent.py`：OpenAI 兼容端点的并发打点器，输出每路 TTFT / 预填 tok/s / 解码 tok/s / 合计 tok/s。

```bash
python3 scripts/bench_concurrent.py http://127.0.0.1:8080 --conc 1,4 \
  --prompt-tokens 2048 --gen-tokens 256 --runs 2
```

注意：`base_url` 是**不带 `/v1`** 的根地址（健康检查打 `/health`）。

## 27B（vLLM）原始数字

条件：1753 token 提示 / 256 token 生成 / `enable_thinking: false` / 两轮取二轮 / MTP k=3。

| 并发 | TTFT | 预填 | 解码/路 | 解码合计 |
|---|---|---|---|---|
| 1 | 0.999 s | 1834.7 tok/s | 51.4 | 51.4 |
| 4 | 0.714 s | 2535.6 tok/s | 52.2 | 202.7 |

投机统计（服务端 `SpecDecoding metrics`）：

| 项 | 值 |
|---|---|
| 平均接受长度 | 3.02 |
| 逐位置接受率 | 0.855 / 0.651 / 0.518 |
| 平均草稿接受率 | 67.5% |

## DFlash 对打（同一批提示，同日同时段）

| 配置 | conc1 解码 | conc4 解码/路 | conc4 合计 | 预填 | 接受长度 |
|---|---|---|---|---|---|
| MTP k=3 | 51.4 | 52.2 | 202.7 | 1835 / 2536 | 3.02 |
| DFlash n=7 | 41.3 | 37.5 | 140.0 | 1631 / 446 | 2.38 |
| DFlash n=3 | 37.2 | 35.1 | 132.3 | 1622 / 444 | 2.17 |

## Strata（Flash-Next）原始数字

从服务日志的完成行统计（`done: N tokens in T s (X tok/s) ... expert cache Y% hit`），n=5682：

| 指标 | p10 | p50 | p90 | max |
|---|---|---|---|---|
| 完整答复解码 tok/s | 40.2 | **85.0** | 102.5 | 135.9 |
| 思考段解码 tok/s（n=53196） | 31.2 | 80.1 | 91.3 | 130.5 |

| 指标 | 值 |
|---|---|
| 专家缓存命中率 | p50 99.2%，最低 83.5% |
| 答复长度 | p50 300 tok / p90 2232 tok / max 150K tok |
| 预填（8K 提示，独立打点） | 约 290–320 tok/s |

**这些数字的边界**：来自真实生产日志而非受控 benchmark，长度分布很宽（p90 2232 tok），所以 p50 是"日常对话/agent 短答"的典型值，**不是**长文生成的保证值。前缀缓存命中时的 200K 级提示（日志 `reading the prompt: 207961 of 207962 tokens, 1 s`）不算预填性能。

## 硬件侧锚点

| 指标 | 持续 | 说明 |
|---|---|---|
| FP16 | 126 TF | cuBLAS 8192³ |
| FP32 | 10.9 TF | 解锁后（未解锁约 0.395 TF） |
| FP64 | 11.12 TF | DGEMM 4096³ |
| D2D | 1605 GB/s | 对照 A100 80GB PCIe 1935 |

## 功率/时钟采样

```bash
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,temperature.gpu \
  --format=csv,noheader
```

解码期间典型读数（生产负载）：SM 贴住本批次档位、util 90–100%、功耗钳在 190W、51–53℃。用 `激活参数 × bpw/8 ÷ 实测 tok/s` 反推的实际带宽只有理论值 ~15–19%，**这就是"提功率/超显存都换不到解码"的判据**。
