# 运维坑清单

按"踩过几次 + 后果严重度"排序。每条都给判据，不给"注意一下"这种废话。

## 1. 内核升级猝死

**症状**：`nvidia-smi` 报"无法与 NVIDIA 驱动通信"、`lsmod | grep nvidia` 为空、`/proc/driver/nvidia` 不存在、`cuInit=100`、所有 GPU 服务 crash-loop。

**根因**：解锁补丁模块装在 `/lib/modules/$(uname -r)/updates/cmpunlocker/`，**不在 DKMS 管理内**。`unattended-upgrades` 装进新内核并重启后，GRUB 进新内核，而新内核下没有 nvidia 模块。

**诊断（免 sudo）**：

```bash
uname -r; lsmod | grep -i nvidia || echo none
modinfo nvidia 2>&1 | head -3          # not found = 模块缺失
find /lib/modules/$(uname -r) -name 'nvidia.ko*'   # 空 = 当前内核没编
grep 'linux-image' /var/log/apt/history.log | tail -3   # 刚装过新内核即命中
```

分型：**`nvidia-smi` 秒回 + `cuInit=100` = 模块缺失（可救）**；`nvidia-smi` 挂死无响应 = 硬件 hang（只能重启）。

**修复**：停所有 GPU 服务 → `sudo ./install.sh --profile=8gb`（按当前内核重编 + 重建 initramfs）→ **重启**（`shutdown -r`，别 `-h`，远程起不回来）→ 复查显存 65536 / Gen2 / `SEC2_DEBUG` / `cuInit=0`。

**预防**：`apt-mark hold` 12 个 nvidia 包（见 [hardware.md](hardware.md#3-驱动冻结最容易翻车的一条)）+ 固定内核版本。

## 2. 不许 SIGKILL 占用 CMP 卡的进程

硬杀会留下 GSP/ACR 状态，下次驱动 probe 直接失败（卡枚举不到，只能重启整机）。停服务一律给显式超时让它自己退（`systemctl stop` / `podman stop --time 300`），疑似卡死时先看 `nvidia-smi` 是否秒回：秒回是软故障可救，挂死才需要重启。

## 3. 端口与网关互撞（`:8080`）

两件事同时成立：前端网关 `gpu-orch` 监听 `:8080`；`vllm-heretic.service` 也监听 `:8080`。同时起 → 后起的那个 `EADDRINUSE` 退出，而"健康检查"可能正打在前一个服务上，看起来像"新配置没生效"。

判据与处置：

```bash
ss -ltnp | grep ':8080 '            # 看 pid 是不是 orch.py
pgrep -af "vllm serve" | grep -v pgrep
crontab -l | grep ensure_gpu_orch   # 每 5 分钟会把网关拉回来，别只 stop 一次
```

规则：**测 vLLM 变体换端口**（8085），用 `systemd-run --user --unit=<name> --collect` 起瞬态单元；测完 `stop` → 确认 `nvidia-smi memory.used < 3000` → 再恢复原服务。

## 4. 显存没回落就起下一个模型

上一个 vLLM 进程已退出但显存还没释放时起新实例，会加载失败或只加载一半（日志停在 `loading model`，显存几十 GB 不动）。判据：`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` **< 2500** 才继续。

## 5. 服务"就绪"不能看 HTTP 状态码有没有 body

加载期接口常回 **503 + JSON body**，`curl -s`（无 `-f`）拿到 body 也算成功，循环瞬间退出，紧接着的打点全吃 `{"error":{"message":"Loading model"}}`——整轮 A/B 白跑。写 `curl -sf` 或显式匹配 `"ok"`。

## 6. `pkill -f "vllm serve"` 会打死自己

在 ssh 一行命令里它匹配到自己这条命令行 → exit 255，看起来像"命令没执行"。用 `pgrep -f "[v]llm serve"` 或按 PID；同一条命令里既 `pkill -f X` 又 `bash X` 也会自杀。

## 7. 共享网关的"旧模型名互踢"

网关按请求唤醒模型时，任何带**旧模型名**的请求都会触发"停当前 → 起旧的"。遗留客户端不会因为你换了默认模型就改名字，双向请求形成互踢，默认模型的服务反复 stop/start 甚至 SEGV。

修法：把重定向加在**请求解析路径**（把旧名字解析成默认 profile），不要加在模型检测路径（后者必须严格，否则旧模型真在跑时会被误标）。验证三件事：旧名字请求仍 200 且 `active_profile` 不变、显存不跳、日志 `starting vLLM <旧>` 计数不再增长。

## 8. 客户端发的空 assistant 消息

llama.cpp 一类后端对 `role=assistant` 且既无 `content` 也无 `tool_calls` 的消息直接 400，而 agent 客户端在工具轮之间就是这么发的。在代理层把这类消息的 `content` 补成单个空格，别指望客户端改。

## 9. 配置漂移：名字说的量化档不是真跑的档

某次 `model_name` 写 `IQ3_S-MTP`，但 `--pack/--native` 实际指向 **UD-IQ4_XS**。对外报"这个服务是什么档"必须读启动参数，不能读模型名。同理，权重目录名里的 `w8a16`/`oQ4e` 也可能与 `config.json` 的 `quantization_config` 不一致——以 config 为准。

## 10. 引擎子进程名 ≠ unit 名

Strata 的引擎子进程叫 `strata-host`。网关或脚本里"停服务"必须按**实际进程名**兜底，否则 unit 停了显存不释放，下一个负载起不来。

## 11. Strata 的两道门（装之前先验证）

- **架构门**：读发布索引 `strata-feed.json` 的 `engine.packages[].archs` 判断是否含你的 SM（本卡 sm_80）。手改 component 的 `archs` 无效（要签名激活记录），`STRATA_REQUIRE_SIGNED_COMPONENTS=0` 也不放行未签名目录。
- **模型门**：`--model-path` 指向的本地文件集要过 fingerprint 表，否则报 `this is not a model size Strata knows`；自定义/消融权重走 `--pack` 自打包（打包要 50–60GB 空闲内存，30–60 分钟）。

## 12. 打点的三个假数据来源

1. 请求体不带 `"cache_prompt": false` → 前缀缓存命中时 `prompt_n` 只报本次新评估的 token，预填/TTFT 全失真。
2. `temperature 0 + top_k 1` 在这类带 chat 模板的混合架构上**偶发首 token 即 EOS**（`predicted_n=1`，解码 0.0）。固定 `temp 0.7 / top_k 20 / top_p 0.95`，并跑两轮取第二轮。
3. 用流式 chunk 数估算 tps 会严重偏低（一个 chunk 可能含多 token）。只信服务端日志的统计行。

## 13. 跨时段比较等于耍流氓

持续负载下，同一配置自身的降速带实测 **−14% ~ −43%**（预填/解码/长文各异）。任何 A/B 必须**同时段交替跑**，不能拿昨天的数字比今天。

## 14. 别拿 `clocks.max.sm` 当"还有余量"

同型号存在不同批次频率档：`clocks.max.sm` 报 1695，但另一批实际可用档只有 1458。采样时把 `clocks.max.sm` / `power.limit` / `enforced.power.limit` 一起打出来再判断，否则会去做无效的超频/提功耗。
