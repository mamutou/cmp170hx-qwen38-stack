# CMP 170HX 上的 Qwen3.8 双栈部署手册

单张 **NVIDIA CMP 170HX（GA100，解锁到 64GB）** 上同时维护两套生产推理栈的完整记录：

- **Strata 0.2.2** 跑 **Qwen3.8-Flash-Next**（176B 总参 / ~6B 激活 + 51B 查表 + MTP）——当前生产主力，走专家缓存 + 投机解码。
- **vLLM 0.27.1** 跑 **Qwen3.8-27B 稠密**（compressed-tensors W8A16，int8 g128，带 MTP 投机）——备用/对照档。

仓库里给的是：硬件与解锁前提、两条栈每一项参数为什么这么设、全部**实测**数字、以及我们试过并**失败**的路线（附数字与机制），避免后来者重跑一遍。

> 全部数字来自同一台机器的实测（单卡 / 190W / PCIe Gen2 x16 / 1695-1728MHz 频率档），不引用厂商标称或他人跑分。每条数字的测量口径写在 [docs/benchmarks.md](docs/benchmarks.md)。

## 实测速览

| 栈 | 模型 / 量化 | 解码（单流） | 解码（4 路合计） | 预填 | 备注 |
|---|---|---|---|---|---|
| Strata 0.2.2 | Flash-Next abliterated / UD-IQ4_XS（专家走内存+SSD 查表） | **p50 85 tok/s**（p90 102.5 / max 135.9，n=5682） | 受并行槽位限制 | 8K ≈ 290–320 tok/s | 专家缓存命中率 p50 **99.2%** |
| vLLM 0.27.1 | Qwen3.8-27B 稠密 / W8A16 int8 g128（30GB） | **51.4 tok/s** | 52.2/路，合计 202.7 | 1835 tok/s（1753 tok 提示，TTFT 1.0s） | MTP k=3，接受长度 **3.02** |

一个反直觉但可复用的结论：**这张卡的解码被"算力/功耗墙"而非带宽摁住**。125B-A6B 级 MoE 在 1.4TB/s 的卡上实测只有带宽公式值的 ~40%，所以降比特换不到速度，能动的只有"每步多出 token"（投机接受率）和"更省算力的内核"（换引擎/换格式）。

## 硬件与前提

| 项 | 值 |
|---|---|
| GPU | 1 × NVIDIA CMP 170HX（GA100，sm_80），解锁后 64GB HBM2e |
| 频率档 | 1695 SM / 1728 mem（同型号存在 1410/1458 的另一批，判法见 hardware 文档） |
| 功率墙 | 190W（硬件上限 300W；被动散热，250W 会撞热降频） |
| PCIe | Gen2 x16（出厂 x4，需硬件补焊 + early-retrain 服务） |
| 算力 | FP16 126 TF 持续 / 165.8 TF 短时；FP32 10.9 / 12.7 TF；FP64 11.12 TF；D2D 1605 GB/s |
| 平台 | Ubuntu 24.04.4 / kernel 6.8.0-139 / 驱动 610.43.02（社区补丁模块） |
| 主机 | i9-12900（16C/24T）、62GB DDR4、2TB NVMe |

解锁、驱动白名单、PCIe Gen2、功率墙持久化的完整步骤见 [docs/hardware.md](docs/hardware.md)。**注意：解锁依赖打过补丁的 nvidia 模块，内核升级会直接让整机失去 GPU**——这是本仓库最重要的一条坑，见 [docs/pitfalls.md](docs/pitfalls.md#1-内核升级猝死)。

## 目录

```
docs/
  hardware.md                    卡、解锁、频率档、PCIe、功率/散热、驱动冻结
  stack-strata-flash-next.md     Strata 栈：pack 结构、专家缓存、MTP 投机、profile 逐项解释
  stack-qwen27b-vllm.md          vLLM 栈：W8A16 权重、MTP k=3、解析器/采样参数逐项理由
  optimization-log.md            优化时间线（含所有失败路线与数字）
  pitfalls.md                    运维坑清单（按踩过的次数排序）
  benchmarks.md                  测量方法学与原始数字口径
scripts/
  bench_concurrent.py            并发/预填/解码打点器（OpenAI 兼容端点）
  serve_qwen27b_vllm.sh          27B 档启动脚本（含 DRY=1 空跑模式）
  serve_strata_flash_next.sh     Strata 档启动脚本（含 DRY=1 空跑模式）
```

## 最短上手路径

```bash
# 1. 先确认卡是全容量、Gen2、功率墙已落
nvidia-smi --query-gpu=memory.total,clocks.max.sm,pcie.link.gen.current,power.limit --format=csv

# 2. 27B（vLLM）：打印展开后的启动命令，不真的起服务
DRY=1 ./scripts/serve_qwen27b_vllm.sh

# 3. Strata（Flash-Next）：同上
DRY=1 ./scripts/serve_strata_flash_next.sh

# 4. 打点（服务起来后）
python3 scripts/bench_concurrent.py http://127.0.0.1:8080 --conc 1,4
```

## 隐私

仓库内容已做脱敏：不含内网地址、用户名、主机名、GPU UUID/序列号、MAC、SSH 信息与任何密钥；脚本里的路径统一写作 `/home/appuser`，主机写作 `<GPU_HOST>`。模型权重与量化仓库均引用 Hugging Face 公开仓库 ID，不含私有链接。

## License

MIT，见 [LICENSE](LICENSE)。文中引用的第三方项目（Strata / vLLM / cmpunlocker / 模型权重）各自遵循其原始许可。
