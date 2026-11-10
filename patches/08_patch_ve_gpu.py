import shutil

P = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/visual_encoding.py"
shutil.copy(P + ".bak_vegpu", P)  # 从原始备份重来
s = open(P).read()

old = """        try:
            return self._forward(batch, server_args)
        except Exception:
            # No later stage will run after an encoder failure.
            minimax_h3_cleanup_temp_dirs(batch)
            raise
        finally:
            # Qwen and the visual tokenizer have both consumed the derived
            # reference-video files by this point. Original localized media
            # remains alive for the following audio stage.
            minimax_h3_cleanup_temp_dirs(batch, owners=("reference_video",))"""

new = """        # SGLANG_VE_GPU_V4: reference image/video VAE encode runs on GPU when
        # the weights are CPU-offloaded (CPU fp32 encode = 20+ min/image).
        # Device chain (4,3,2,1,0), free >= 20GiB. GPU4 works when the
        # ref2va switch enables SGLANG_MM_ENC_OFFLOAD=1 (text_encoder returns
        # to CPU after text encoding, freeing the card for this window);
        # rank cards are fallback (their torch pool also runs tight due to
        # non-torch USP/NCCL scratch). Weights return to CPU afterwards.
        import os as _os_ve
        _ve_moved = False
        _ve_dev = None
        if _os_ve.environ.get("SGLANG_MM_VE_GPU", "1") == "1":
            try:
                _p = next(self.video_vae.parameters())
                if _p.device.type == "cpu" and torch.cuda.is_available():
                    for _di in (4, 3, 2, 1, 0):
                        _free, _total = torch.cuda.mem_get_info(_di)
                        if _free >= 20 * 1024**3:
                            _ve_dev = torch.device("cuda", _di)
                            break
                    if _ve_dev is not None:
                        self.video_vae.to(_ve_dev)
                        _ve_moved = True
                        print(f"[ve-gpu] video_vae moved to cuda:{_ve_dev.index} for encode", flush=True)
                    else:
                        print("[ve-gpu] no rank card with >=20GiB free — CPU encode (slow)", flush=True)
            except Exception as _e:
                print(f"[ve-gpu] move skipped: {_e}", flush=True)
        try:
            return self._forward(batch, server_args)
        except Exception:
            # No later stage will run after an encoder failure.
            minimax_h3_cleanup_temp_dirs(batch)
            raise
        finally:
            # Qwen and the visual tokenizer have both consumed the derived
            # reference-video files by this point. Original localized media
            # remains alive for the following audio stage.
            minimax_h3_cleanup_temp_dirs(batch, owners=("reference_video",))
            if _ve_moved:
                try:
                    self.video_vae.to("cpu")
                    torch.cuda.empty_cache()
                    print("[ve-gpu] video_vae returned to cpu", flush=True)
                except Exception as _e:
                    print(f"[ve-gpu] return failed: {_e}", flush=True)"""

assert s.count(old) == 1, "anchor x%d" % s.count(old)
s = s.replace(old, new)
open(P, "w").write(s)
import py_compile
py_compile.compile(P, doraise=True)
print("PATCHED VE-GPU-V3RANK + py_compile OK")
