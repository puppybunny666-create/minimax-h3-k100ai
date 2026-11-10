#!/usr/bin/env python3
"""patch_encoder_one_gpu.py — 编码器单卡铺到 GPU1(2上下文实验)。
8 上下文(device_map 铺 8 卡)下 rank0 的 hipMalloc 在驱动层死锁(4 次复现);
此实验把编码器全部放 GPU1: rank0 进程只有 cuda:0 + cuda:1 两个上下文。
显存: GPU1 = DiT 分片 8.7G + 编码器 28G ≈ 37G < 64G ✓
"""
import sys

PATH = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py"

OLD = '''            _max_mem = {0: "2GiB"}
            for _i in range(1, torch.cuda.device_count()):
                _max_mem[_i] = "8GiB"'''

NEW = '''            _max_mem = {0: "0GiB", 1: "30GiB"}
            for _i in range(2, torch.cuda.device_count()):
                _max_mem[_i] = "0GiB"'''.replace("{0:", "{0:").replace("{_i:", "{_i:")  # noqa: keep literal


def main():
    src = open(PATH, encoding="utf-8").read()
    if OLD not in src:
        print("ANCHOR NOT FOUND")
        return 1
    src = src.replace(OLD, NEW, 1)
    open(PATH, "w", encoding="utf-8").write(src)
    print("patched: encoder -> GPU1 only (2-context experiment)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
