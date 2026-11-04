# 硬件与解锁前提

## 1. 卡的基本盘

| 项 | 值 | 说明 |
|---|---|---|
| 型号 | NVIDIA CMP 170HX（GA100，`10de:20c2`） | 矿卡核心，70 SM = **4480 CUDA 核**（GA100 每 SM 64 个 FP32 核，不是消费级 Ampere 的 128，用 `SM×128` 算会高估一倍） |
| 显存 | 出厂 8GB → 解锁 **65536 MiB** | 解锁后须过全容量哈希 + 长时间满载压测才算可用 |
| 频率档 | **1695 SM / 1728 mem** | 同型号存在另一批 1410/1458（算力 -15~20%、带宽 1605→1355 GB/s），但真实 LLM 推理差距只有 3–10% |
| 功率 | 190W 墙（硬件允许 300W） | 被动散热；250W 稳态 84–85℃ 触发热降频，190W 稳态 79–81℃ 无节流且稳态频率几乎相同 |
| PCIe | Gen2 x16 | 出厂只贴了 4 条链路的耦合电容，手焊后 x16；开机前 1–2 分钟仍是 Gen1，由 early-retrain 服务抬到 Gen2 |
| NVLink | 无 | 多卡只能走 PCIe，TP/PP 的通信代价要按 Gen2 8GB/s 算 |

实测算力（单卡，190W）：

| 指标 | 持续 | 短时峰值 |
|---|---|---|
| FP16（cuBLAS 8192³） | 126 TF | 165.8 TF |
| FP32 | 10.9 TF | 12.7 TF |
| FP64（DGEMM 4096³） | 11.12 TF | — |
| D2D 拷贝 | 1605 GB/s | — |

对照 A100 80GB PCIe：FP16 约 40%、FP32 约 56%、带宽约 83%。

⚠️ **算力"解锁"是否生效的判据**：FP32 约 0.395 TF = 仍被限速 32×；本机实测 10.9–12.7 TF 说明补丁已生效（限的是指令发射率，不是显存带宽）。对 LLM 的意义是**预填大幅受益、解码几乎不变**。

## 2. 解锁流程（一次性，但会被内核升级破坏）

前置：Linux x86-64、Secure Boot 关闭、root；驱动必须命中社区补丁的白名单版本（本机 `610.43.02`），且用 **nvidia-open**。

```bash
lspci | grep -i nvidia                      # 期望 GA100 [CMP 170HX]，10de:20c2
sudo apt install nvidia-driver-610-open nvidia-utils-610 build-essential dkms linux-headers-$(uname -r)
sudo reboot                                  # 首次装驱动后先重启一次
git clone --depth 1 https://github.com/amoghmunikote/cmpunlocker
cd cmpunlocker && sudo ./install.sh --profile=8gb
sudo shutdown -r now                         # 必须冷重启：热门 modprobe 不走 SEC2 解锁路径
```

验证：

```bash
nvidia-smi --query-gpu=memory.total,clocks.max.sm,pcie.link.gen.current,pcie.link.width.current --format=csv
# 期望 65536 MiB / 1695 MHz / Gen2 / x16
sudo ./verify.sh                             # 全 ✓（非 root 会把 PCIe 误报成 not active）
journalctl -b | grep SEC2_DEBUG              # 开机早期 WPR/fbSize 重写日志
journalctl -b | grep "early retrain"         # Gen1→Gen2 重训练 SUCCESS
```

## 3. 驱动冻结（最容易翻车的一条）

补丁模块编译到 `/lib/modules/$(uname -r)/updates/cmpunlocker/`，**不在 DKMS 管理内**。一旦 `unattended-upgrades` 装进新内核并重启，新内核下没有 nvidia 模块 → `nvidia-smi` 秒回、`cuInit=100`、所有推理服务 crash-loop。

```bash
# 必须同时 hold 驱动包与内核
sudo apt-mark hold nvidia-driver-610-open nvidia-dkms-610-open nvidia-compute-utils-610 \
  libnvidia-{cfg1,common,compute,decode,encode,extra,fbc1,gl}-610
apt-mark showhold | grep -c nvidia          # 期望 12
```

已经中招时：停止所有 GPU 服务 → 以 root 重跑 `sudo ./install.sh --profile=8gb`（会按当前内核重编）→ **重启**（不是关机，远程起不回来）。详见 [pitfalls.md](pitfalls.md#1-内核升级猝死)。

## 4. 功率墙持久化

`nvidia-smi -pl` 不跨重启，用 oneshot 服务落盘：

```bash
# /usr/local/bin/cmp-power-limit.sh：轮询 nvidia-smi 就绪(≤180s) 后执行 nvidia-smi -pl 190（幂等）
# /etc/systemd/system/cmp-power-limit.service：Type=oneshot, WantedBy=multi-user.target
sudo systemctl daemon-reload && sudo systemctl enable --now cmp-power-limit.service
```

功率墙是**卡级**的：改一次对这张卡上所有服务生效，不要为单个服务去改持久化脚本。

### 还要不要加功率/超频？

先采样再决定：

```bash
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu --format=csv,noheader
```

三个读数一起看就能定性：

- SM 贴住本批次档位（1458/1695）+ util 90–100% + 功耗钳在墙值 → **算力/功耗墙**，提功率换不到频率（实测 190W→250W 稳态频率几乎同）。
- 用 `激活参数 × bpw/8 ÷ 实测 tok/s` 反推的实际带宽只有理论值 ~15% → **不要超显存**，带宽不是瓶颈。
- 稀疏注意力/MoE 对齐这类 ALU 重的 kernel 是例外：SW 功耗墙会把 SM 压到 800–1150MHz，此时提功率能直接换 kernel 时间。

**GPC 时钟 VF 偏置**（8GB 版唯一还能动的软件旋钮，需 root + NVML）：

| offset | 峰值时钟 | 代码解码 | 散文解码 | 10K 预填 | 风险 |
|---|---|---|---|---|---|
| 0（默认） | 1470 MHz | 96.5 / 105.3 | 79.7 / 83.8 | 2179 / 2172 | — |
| +150 | 1605–1620 MHz | 97.9 / 98.3 | 85.3 / 91.6 | **2308 / 2293（+5.8%）** | 低 |
| +250 | 1695 MHz | — | — | **Xid 13 非法指令，CUDA context 被杀** | 高 |

结论：只有预填稳定 +5~6%（解码在噪声内，符合带宽受限原理），功耗不升反降；+250 在本卡直接打挂 context（文档参考卡 +300 才出事，说明本卡耐受度更低，别套别人档位）。**为 6% 预填承担静默损坏风险不划算，默认 offset=0。** 想长期用必须先过几轮全卡 memtest 0 错误。

## 5. 批次频率档怎么判

```bash
nvidia-smi --query-gpu=clocks.max.sm,clocks.max.memory --format=csv
# 1695 / 1728 → 标准档；1410 / 1458 → 低频档
```

低频档 ≠ 次品，是硬件档位；推理场景的实用差距远小于算力基准差距（decode 75→68 tok/s、prefill 985→955）。
