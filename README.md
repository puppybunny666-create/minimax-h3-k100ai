# MiniMax-H3 on Hygon K100 AI：国产 8 卡 DCU 上的全模态视频生成

> 官方 Model Zoo 仅支持新一代大显存卡 BW1100（135B BF16 单卡直跑，FP8 仅 BW1100/BW1101）。
> 本项目把 MiniMax-H3 **跨型号移植**到 8× 海光 K100 AI（64GB/卡）：六卡协同、音视频同轨、
> 与另一路 LLM 业务同机共存，持续生产至今。
>
> 范围说明：本仓库覆盖 **bf16 全精度、全模式（t2va/fl2va/ref2va）、生产化部署**。
> INT8 量化与双卡 head 并行的 kernel 级优化路线见
> [DocPang/minimax-h3-k100ai-optimization](https://github.com/DocPang/minimax-h3-k100ai-optimization)
> ——两仓库互补，合起来是 K100AI 上 H3 的完整图景。

![海滩日落——灰暗糊修复后的验证画面](samples/sunset_1344x768_1.jpg)

*上图为灰暗糊修复后的验证画面：提示词要求"海滩日落"，输出中橙黄天空、波浪、湿沙倒影逐词吻合。
修复前同一模型输出的是无任何可辨识内容的灰褐色云雾（见[踩坑实录](docs/02-踩坑实录.md)）。*

## 结果

| 模式 | 任务 | 端到端 | 去噪单步 | 峰值显存/卡 |
|---|---|---|---|---|
| t2va 16:9（1344×768，5.2s 视频） | 文生视频+音频 | **462–471s（7.7–7.9 分钟）** | 28 s/it | ~59 GB |
| t2va 21:9（1536×672） | 同上 | ~10.5 分钟 | ~31 s/it | 59.3 GB |
| t2va 9:16（768×1344） | 竖版 | ~19 分钟（横版 2.4 倍） | 75 s/it | ~59 GB |
| fl2va | 首帧图生视频 | ~517s | — | — |
| ref2va | 参考图+参考视频驱动 | 41–42 分钟（154 s/it，架构性成本） | — | — |

要点：

- **首条与后续同速**——这曾经是两个月悬案，根因在 GPU 硬件队列超订（[踩坑实录 §三](docs/02-踩坑实录.md)）。
- **位级无损**：VAE 跨卡解码与单卡路径输出 md5 一致，解码后 3.84 亿字节逐位对比 max=0。
- 10 条马拉松连续生成零劣化、零泄漏；ref2va 3/3 马拉松显存逐字节平坦。

演示视频：[霓虹夜街红色超跑（16:9）](samples/h3_car_fixed.mp4) ｜ [峡谷航拍（21:9）](samples/h3_canyon_21x9.mp4)

## 这台机器上发生了什么

135GB 的模型装不进 64GB 的卡，官方单卡路线不可用，于是自研了一套并行栈：

- **FSDP-lite**：torch FSDP2 在 DTK 26.04 上整体不可用 → 自研逐块 all-gather 分片
- **RCCL 绕法**：all_to_all / all_gather 多种原语运行期死锁 → 广播环模拟 + 预分配 scratch + 通信器预热
- **六卡布局**：4 卡 DiT 去噪分片 + 1 卡 VAE + 1 卡 Qwen3VL-32B 文本编码器，余下 2 卡跑别的业务
- **唤醒休眠代码**：上游 VAE 里躺着一整套跨卡 tile 分发实现，60 行补丁注入分布式组 → 解码 3.6×
- **一行定标的性能案**：`GPU_MAX_HW_QUEUES` 26→1→4，破解"第二条视频必慢 2.3 倍"的驱动级队列超订

过程中修复了一个最阴险的 bug：所有数值验证全绿、不崩、不 NaN，但模型输出"无条件均值"——
因为自研加载捷径绕过了 weight_loader 钩子，**grouped-qkv 按头重排从未生效**。排查方法论独立成篇。

## 文档导航

| 文档 | 内容 |
|---|---|
| [01-方案](docs/01-方案.md) | 六卡布局、四大组件账、30+ 环境开关、镜像链与重建配方 |
| [02-踩坑实录](docs/02-踩坑实录.md) | 三大战役完整复盘：灰暗糊、队列悬案、OOM 连环案 |
| [03-运维篇](docs/03-运维篇.md) | 显存"泄漏"翻案两次、自愈链设计、断电事故复盘 |
| [04-方法论](docs/04-方法论.md) | 算子对拍参照系、判别实验设计、探针军规、证据分级纪律 |

`patches/`：核心补丁集（幂等脚本，逐个附"改什么/为什么/怎么验证"），按镜像链时点排序。

## 重建配方（概要）

```text
基础镜像（海光官方 Model Zoo 公开定制镜像）:
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang-ubuntu22.04-dtk26.04-py3.10-20260731-MiniMax-H3
官方源码仓库（部署文档 / 流水线源码）:
  https://developer.sourcefind.cn/codes/modelzoo/minimax-h3
权重（ModelScope 公开，135B BF16；FP8 版仅 BW1100/BW1101）:
  https://www.modelscope.cn/models/MiniMax/MiniMax-H3

容器:  --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 200g --network=host
       -v <模型目录>:/models/MiniMax-H3   -v /opt/hyhal/:/opt/hyhal/:ro

补丁:  按 patches/ 序号幂等应用（每个补丁先校验锚点唯一再改，可重复执行）
```

权重不在镜像内（独立目录挂载）；基础镜像含 DTK 26.04、sglang-diffusion 0.5.10rc0、
flash_attn 2.8.3+das 及官方初始适配。完整步骤见 [01-方案](docs/01-方案.md)。

## 致谢与相关

- 海光光合开发者社区（Model Zoo、DTK 手册、DCU 技术社区帖）与 sglang-diffusion / DAS 生态
- 社区公开排障记录（gfx1151 队列停顿分析、ROCm#6437、sglang#33978 同族案例）对本项目队列问题定位的帮助
- 同模型先行项目：[DocPang/minimax-h3-k100ai-optimization](https://github.com/DocPang/minimax-h3-k100ai-optimization)（INT8 kernel 优化与双卡 head 并行，早于本项目三周）
- 同厂姊妹项目：[DocPang/strata-k100ai](https://github.com/DocPang/strata-k100ai)（同硬件上的 LLM 推理优化）

## 许可与声明

- 本仓库只包含**运行方案、补丁与验证记录**，不包含也不分发 MiniMax-H3 模型权重；
  权重从官方渠道获取：[ModelScope MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3)，
  并遵守其许可。
- `samples/` 中的视频与画面帧均为本项目在 K100 AI 上实际生成的验证内容。
