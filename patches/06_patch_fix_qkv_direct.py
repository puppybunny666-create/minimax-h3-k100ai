#!/usr/bin/env python3
"""patch_fix_qkv_direct.py — 修复根因: SGLANG_LITE_DIRECT_V1 直装捷径绕过了
weight_loader, 导致 MiniMax-H3 grouped-qkv 重排(_reorder_grouped_qkv_to_qkv)
从未生效 → 主块+refiner 的 q/k/v 切分错乱 → 注意力语义破坏 → 条件失效+灰暗糊.

修法: 捷径改为显式 opt-in (SGLANG_MM_LITE_RAW_DIRECT=1), 默认恢复 weight_loader
路径(重排+TP 语义尊重), 之后照常 _distribute_no_collective 分片.
"""
import sys

PATH = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/loader/fsdp_load.py"
MARK = "# SGLANG_FIX_QKV_DIRECT_V1"

OLD = """            # SGLANG_LITE_DIRECT_V1: lite 下形状一致时直装, 绕过 TP 拼装
            import os as _os_dl
            _direct = (
                _os_dl.environ.get("SGLANG_MM_FSDP_LITE") == "1"
                and weight_loader is not None
                and tuple(full_tensor.shape) == tuple(meta_sharded_param.shape)
            )"""

NEW = """            # SGLANG_LITE_DIRECT_V1: lite 下形状一致时直装, 绕过 TP 拼装
            # SGLANG_FIX_QKV_DIRECT_V1: 直装会绕过 weight_loader 里的
            # grouped-qkv 重排(MiniMax-H3 checkpoint 是按头分组布局), 曾导致
            # q/k/v 切分错乱(灰暗糊根因). 改为显式 opt-in, 默认走 weight_loader.
            import os as _os_dl
            _direct = (
                _os_dl.environ.get("SGLANG_MM_LITE_RAW_DIRECT") == "1"
                and weight_loader is not None
                and tuple(full_tensor.shape) == tuple(meta_sharded_param.shape)
            )"""


def main():
    src = open(PATH, encoding="utf-8").read()
    if MARK in src:
        print("already patched")
        return 0
    if OLD not in src:
        print("ANCHOR NOT FOUND")
        return 1
    if src.count(OLD) != 1:
        print(f"ANCHOR NOT UNIQUE: {src.count(OLD)}")
        return 1
    src = src.replace(OLD, NEW, 1)
    open(PATH, "w", encoding="utf-8").write(src)
    print("patched: qkv-direct bypass disabled by default (SGLANG_MM_LITE_RAW_DIRECT opt-in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
