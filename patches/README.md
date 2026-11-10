# 核心补丁集

按镜像链时点排序的幂等补丁脚本。每个脚本先校验目标锚点唯一（`count(OLD) == 1`，不唯一拒打），
修改后 difflib 自检"只增不删"，可重复执行。目标文件均为容器内 sglang 标准安装路径
（`/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/...`）。

应用方式：`docker exec <容器> python3 /models/MiniMax-H3/<补丁名>.py`（模型挂载目录任选）。

| # | 补丁 | 战役 | 改什么 | 验证方式 |
|---|---|---|---|---|
| 01 | `01_patch_fsdp_lite.py` | 平台移植 | 自研 FSDP-lite（`SGLANG_MM_FSDP_LITE=1`）：dim-0 分片 + 逐模块 all-gather 到预分配池视图，替代 DTK 上不可用的 torch FSDP2 | 加载期不再冻死；分片权重与单卡参考逐位对拍 |
| 02 | `02_patch_usp_flat.py` | RCCL 死锁 | USP 交换层改用 flat 版 `all_gather_into_tensor`，绕开 list 版 `all_gather` 大 payload 死锁 | 与原生实现 bit-exact 对拍 |
| 03 | `03_patch_alloc_warmup.py` | RCCL 死锁 | 模型加载期分配 + 通信器预热：预分配交换 scratch、梯度递增 all_gather 逼出 RCCL lazy buffer | 运行期零 `hipMalloc`（在飞通信 × 分配 = 驱动死锁） |
| 04 | `04_patch_encoder_one_gpu.py` | 编码器死锁环 | 文本编码器单卡驻留（放弃 `device_map="auto"` 多卡摊放，20s 后全体冻结） | 端到端文本编码稳定 |
| 05 | `05_patch_sdpa_bounds.py` | 性能 | memoize varlen attention 的 `cu_seqlens.tolist()`（每步重复主机侧换算） | 单步主机耗时下降 |
| 06 | `06_patch_fix_qkv_direct.py` | **灰暗糊（主战役）** | 直装捷径改显式 opt-in（默认关闭），恢复 weight_loader 钩子——grouped-qkv 按头重排从此真正生效 | 运行时 qkv 分片 vs 重排后 checkpoint max_diff=0；换 prompt latent 余弦 0.9999→0.114；画面正常 |
| 07 | `07_patch_warmup3.py` | RCCL 死锁 | ulysses 通信器梯度递增预热 | 首步即稳定 |
| 08 | `08_patch_ve_gpu.py` | ref2va 十三枪 OOM | VAE 编码改走无 mmap 污染的空卡（毒化是进程级地址空间行为） | fl2va/ref2va 编码路径不再 OOM |
| 09 | `09_patch_vae_sp.py` | 性能（3.6×） | 唤醒上游休眠的跨卡 tile 解码：WORLD 组注入 VAE parallel_state + REPLICATED 化 + latent 广播 + 全 rank 表决回落（防死锁），`SGLANG_MM_VAE_SP=1` 门控 | 同 seed 输出 md5 与单卡完全一致、383,975,424 字节逐位 max=0；解码 105.5s→27.5s |
| 10 | `10_patch_memgovern_v2.py` | VAE 显存累积 | 请求尾 `video_vae/audio_vae .to("cpu")` + `empty_cache()` + `ipc_collect()`（活引用不是空闲块） | A/B 对照连续 3 条零累积（36.3G 恒定） |

## 未收录补丁说明

原战场的 84 个补丁中，其余为：探针类（TEXT_CHAIN / LATENT_DUMP / WEIGHT_ORACLE / MEMTRACE /
VAETIME 等，env 门控默认关，依赖具体调试目标）、实验中间产物（被后续版本取代）与
一次性诊断脚本。需要时可按 04-方法论的探针军规自行编写——军规本身就是从这批脚本里
提炼的。
