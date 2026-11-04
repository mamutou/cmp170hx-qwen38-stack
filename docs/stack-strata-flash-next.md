# Strata 栈：Qwen3.8-Flash-Next（生产主力）

## 1. 为什么这个模型能塞进 64GB

Flash-Next 标称 176.9B，但**不是 176B 全矩阵乘**：

| 组成 | 参数量 | 放哪 | 是否占显存 |
|---|---|---|---|
| 骨干（稠密+注意力+激活专家） | 125B 总 / **~6B 激活** | 主分片 → 显存（专家溢出到内存） | 部分 |
| lookup 表（PLE / `per_layer_token_embd`） | 51B | 第二分片 → SSD/内存，按行查表 | **否** |
| MTP draft head | 4B | 随引擎 | 少量 |

判"装不装得下"只看**常驻矩阵乘权重 + KV**，绝不用总参数量。KV 也按 hybrid 结构实算：只有少数层是全注意力，262K 上下文 bf16/int8 KV 约 6–7GB 量级，不是稠密模型的公式值。

## 2. 版本与模型资产

| 项 | 值 |
|---|---|
| 引擎 | Strata **0.2.2**（build `0.1.47`） |
| 权重 | `hh-fn-v2-udiq4xs/`：UD-IQ4_XS 主分片（3 段）+ PLE 分片 |
| pack | `packs/hui-iq4xs`（`--pack` 自打包，引擎指纹校验通过后使用） |
| 备选档 | UD-IQ4_XS / GSQ-RCO IQ3_S / Q4_K_XL 三套都备过，最终跑 IQ4_XS |
| tokenizer | `packs/hui-iq3s/tokenizer` |

Strata 的量化档位与定位（官方口径）：Q2_0 37.6GB / IQ2_XS 39.2GB / IQ3_XXS 47GB / **IQ3_S 54.8GB（官方称与完整模型公开测试持平）** / UD-IQ4_XS 及以上 ~80GB 起（需低内存流式）。在 64GB 卡上，Strata 的价值是**速度**不是比特数；升到 IQ4_XS 要靠 `--resident-experts` 让专家转文件映射腾出内存。

## 3. profile 逐项解释

```json
{
  "args": [
    "--pack", "packs/hui-iq4xs",
    "--native", "<UD-IQ4_XS 主分片 1>",
    "--ple-gguf", "<UD-IQ4_XS 分片 2（PLE）>",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "16384",
    "--spec", "4",
    "--mtp", "mtp/rt",
    "--spec-min-p", "0.5",
    "--max-context", "262144",
    "--kv", "int8",
    "--kv-grow",
    "--resident-experts",
    "--adapt-async", "1"
  ],
  "parallel": 2,
  "port": 18080
}
```

| 参数 | 作用 | 为什么这么设 |
|---|---|---|
| `--pack` / `--native` / `--ple-gguf` | 自打包格式 + 主分片 + PLE 分片 | Strata 是托管体验，`--model-path` 要塞任意 GGUF 会撞 fingerprint 白名单；`--pack` 是官方给消融权重开的路 |
| `--expert-profile` | 专家命中剖面，喂给 LRU 缓存 | 用真实负载统计出的热点专家表，比 `auto` 冷启命中率高 |
| `--expert-cache auto` | 显存里放多少专家 | 显存大者多放；实测命中率 p50 **99.2%**，即基本不落 PCIe |
| `--prefill 16384` | 预填 chunk | 与 27B 档保持同口径；再大只是峰值显存换一点点吞吐 |
| `--spec 4` | 投机深度 | 本卡实测 n=2/3/4 → 代码 75.4 / 79.9 / **85.3**，n=5 显存吃紧；深度必须自己扫，不存在通用最优 |
| `--mtp mtp/rt` | 用 MTP head 做草稿 | 主线构建若不带 MTP 图会**静默忽略** head，必须用带图的构建并显式指定 |
| `--spec-min-p 0.5` | 草稿最低接受概率 | 低于阈值不再往下猜，省算力 |
| `--kv int8` + `--kv-grow` | KV 量化 + 按需增长 | 262K 上下文下省显存；`--kv-grow` 避免一上来按最大上下文预分配 |
| `--resident-experts` | 专家转文件映射（页缓存可回收） | 让同一台机器还能跑别的服务：可用内存 0→37GB，解码只 −0.7%，代价是冷启动 36→52s |
| `--adapt-async 1` | 自适应异步 | 与扫参叠加时注意三次重复噪声 ±8–10%，判据要用两轮取中位 |
| `parallel: 2` | 并行槽位 | 该引擎会为并行腾 KV 而**缩小 GPU 专家缓存**，单请求 −9%；单人长上下文用 1 |

⚠️ **配置漂移**：`model_name` 里写的是 `IQ3_S-MTP`，但 `--pack`/`--native` 指向 IQ4_XS。对外报"这个服务是什么量化档"时必须读 args，不能读名字——我们就是因为名字先信了口径，后来才发现实际在跑 IQ4_XS。

## 4. 服务化

```ini
# ~/.config/systemd/user/strata-flash.service
[Service]
Type=simple
WorkingDirectory=/home/appuser/strata22
Environment=XDG_RUNTIME_DIR=/run/user/1000
Environment=PATH=/home/appuser/strata22/ctl/python/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/appuser/strata22/ctl/bin/strata serve --profile /home/appuser/strata22/profile-flash-strata.json --port 18080
Restart=always
RestartSec=20
TimeoutStartSec=600
TimeoutStopSec=120
KillMode=control-group
StandardOutput=append:/home/appuser/strata22/logs/strata-flash.service.log
```

引擎子进程名叫 `strata-host`（不是 unit 名），所以网关/脚本里的"停服务"必须按**实际进程名**兜底，否则显存不释放。

## 5. 实测（本机，单卡 190W，2026-10 日志统计）

| 指标 | 值 | 样本 |
|---|---|---|
| 完整答复解码 | **p50 85 tok/s**，p90 102.5，max 135.9，p10 40.2 | n=5682 次完成 |
| 思考段解码 | p50 80.1 tok/s，p90 91.3 | n=53196 段 |
| 专家缓存命中 | **p50 99.2%**，最低 83.5% | 同上 |
| 预填（8K 提示） | 约 290–320 tok/s | 独立打点 |
| 答复长度 | p50 300 tok、p90 2232 tok、max 150K tok | 同上 |

读法：**解码速度随生成长度下滑**（短答案 110+ tok/s，长答案落到 40–80），因为专家缓存与 KV 都在增长；对外只报一个"tok/s"是误导，要连同"生成多少 token"一起报。前缀缓存命中时 200K 级提示的预填会近乎瞬时（日志里 `reading the prompt: 207961 of 207962 tokens, 1 s`），别把这种情况当成真实预填速度。

## 6. 一句话结论

在这张算力被削的卡上，Flash-Next + Strata 是当前**单位成本最高的输出能力**：专家缓存几乎全命中、MTP 投机深度 4、262K 上下文还能跑。它不适合与 27B 同时常驻（64GB 只够一个满载模型），切换必须停一个再起一个。
