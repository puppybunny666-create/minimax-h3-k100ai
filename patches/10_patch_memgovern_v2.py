#!/usr/bin/env python3
"""patch_memgovern_v2.py — 显存治理升级: 请求尾 VAE 下放 CPU + 三连回收.
替换 v1 的"仅 empty_cache"块为:
  1. video_vae/audio_vae .to("cpu")  — 打断 GPU 副本跨请求累积(间隙长, 下次加载 ~30s 无感)
  2. torch.cuda.empty_cache()        — 归还空闲块
  3. torch.cuda.ipc_collect()        — 回收 IPC 共享块(多进程服务)
env: SGLANG_MM_REQ_CACHE_CLEAR=1 (默认开)
"""
import sys

PATH = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/decoding.py"
MARK = "# SGLANG_MEMGOVERN_V2"

OLD = '''        # SGLANG_MEMGOVERN_V1b (回收: 请求尾 empty_cache, 治碎片/累积)
        import os as _os_mg2

        if _os_mg2.environ.get("SGLANG_MM_REQ_CACHE_CLEAR", "1") == "1":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
'''

NEW = '''        # SGLANG_MEMGOVERN_V2 (请求尾: VAE 下放 CPU + 三连回收)
        import os as _os_mg2

        if _os_mg2.environ.get("SGLANG_MM_REQ_CACHE_CLEAR", "1") == "1":
            try:
                for _vae_name in ("video_vae", "audio_vae"):
                    _vae_mod = getattr(self, _vae_name, None)
                    if _vae_mod is not None and next(
                        _p.device.type == "cuda"
                        for _p in _vae_mod.parameters()
                    ):
                        _vae_mod.to("cpu")
                        print(f"[mem-gov2] {_vae_name} offloaded to CPU", flush=True)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as _e:
                print(f"[mem-gov2] clear failed: {_e}", flush=True)
'''


def main():
    src = open(PATH, encoding="utf-8").read()
    if MARK in src:
        print("already patched v2")
        return 0
    if src.count(OLD) != 1:
        print(f"ANCHOR NOT UNIQUE: {src.count(OLD)}")
        return 1
    src = src.replace(OLD, NEW, 1)
    open(PATH, "w", encoding="utf-8").write(src)
    print("patched: mem governance v2 (vae offload + triple clear)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
