#!/usr/bin/env python3
"""patch_vae_sp.py — VAE 跨卡 SP 解码 (SGLANG_MM_VAE_SP=1 门控)
设计:
  1. parallelism_type: SP开→REPLICATED(全rank进forward, 修死锁), 关→原MAIN_RANK_AND_SEND
  2. _vae_sp_bootstrap: 把 WORLD 组注入 VAE bundle 的 parallel state
     (上游 tiled_decode 的 round-robin tile 分布 + all_gather 即刻生效)
  3. latent 广播(集合操作, rank0 发/其余收, 全员参与)
  4. 其余路径(audio/post/offload)全 rank 一致执行, 下游仍取 rank0 广播, 语义零变化
tile 级并行无需 SpatialParallelConv3d (tiles 自包含, 无需 halo 交换);
构造期 flags 保持 False 正确. 失败时自动回落 sp=1 单卡路径. """
import py_compile, shutil

P = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/decoding.py"
shutil.copy(P, P + ".bak_sp1")
src = open(P, encoding="utf-8").read()

def rep(old, new, cnt=1):
    global src
    n = src.count(old)
    assert n == cnt, f"anchor x{n} (want {cnt}): {old[:70]!r}"
    src = src.replace(old, new)

# ---- A. parallelism_type 门控 + bootstrap/broadcast 方法 ----
rep("""    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS""",
"""    @property
    def parallelism_type(self) -> StageParallelismType:
        if __import__("os").environ.get("SGLANG_MM_VAE_SP") == "1":
            return StageParallelismType.REPLICATED
        return StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS

    def _vae_sp_bootstrap(self):
        \"\"\"SGLANG_MM_VAE_SP=1: seed bundled VAE parallel state with the real
        world group so tiled_decode distributes tiles across ranks.\"\"\"
        if getattr(self, "_vae_sp_seeded", False):
            return
        import importlib
        import torch.distributed as _dist
        self._vae_sp_seeded = True
        self._vae_sp_on = False
        if not _dist.is_initialized() or _dist.get_world_size() == 1:
            return
        ws = _dist.get_world_size()
        rank = _dist.get_rank()
        try:
            inner = getattr(self.video_vae, "model", None)
            if inner is None:
                inner = self.video_vae
            pkg = type(inner).__module__.rsplit(".", 1)[0]
            par = importlib.import_module(pkg + ".parallel")
            par.get_parallel_state().update(
                {
                    "sp_enabled": True,
                    "sp_size": ws,
                    "sp_rank": rank,
                    "sp_process_group": _dist.group.WORLD,
                    "group_size": ws,
                    "group_rank": rank,
                    "local_process_group": _dist.group.WORLD,
                    "tp_size": 1,
                    "tp_rank": 0,
                }
            )
            self._vae_sp_on = True
            print(f"[vae-sp] rank{rank}: parallel state seeded sp_size={ws}", flush=True)
        except Exception as _e:
            self._vae_sp_on = False
            print(f"[vae-sp] rank{rank}: bootstrap FAILED ({_e}) — falling back to sp=1", flush=True)

    def _vae_sp_broadcast_latent(self, t):
        \"\"\"全员集合: rank0 发 shape+张量, 其余按 shape 重建接收.\"\"\"
        import torch as _torch
        import torch.distributed as _dist
        if _dist.get_rank() == 0:
            _shp = _torch.tensor(list(t.shape), dtype=_torch.long, device=t.device)
            _dist.broadcast(_shp, src=0)
            _contig = t.contiguous()
            _dist.broadcast(_contig, src=0)
            return t
        _shp = _torch.zeros(t.ndim, dtype=_torch.long, device=t.device)
        _dist.broadcast(_shp, src=0)
        out = _torch.empty(tuple(int(s) for s in _shp), dtype=t.dtype, device=t.device)
        _dist.broadcast(out, src=0)
        return out

    def _vae_sp_vote_ready(self):
        \"\"\"全员表决: 任一 rank 缺 video_vae 则全员关 SP(回落每 rank 独立解码).\"\"\"
        import torch as _torch
        import torch.distributed as _dist
        _dev = _torch.device(_torch.cuda.current_device()) if _torch.cuda.is_available() else "cpu"
        _ok = _torch.tensor(
            1 if (getattr(self, "video_vae", None) is not None and self._vae_sp_on) else 0,
            dtype=_torch.int32,
            device=_dev,
        )
        _dist.all_reduce(_ok, op=_dist.ReduceOp.MIN)
        _torch.cuda.synchronize() if _dev != "cpu" else None
        if int(_ok.item()) != 1:
            self._vae_sp_on = False
            print(f"[vae-sp] rank{_dist.get_rank()}: vote failed — SP off, fallback per-rank decode", flush=True)""")

# ---- B. forward 内: bootstrap + 表决 (插在 t1 计时后) ----
rep("""        _vt_t1 = _vt_time.perf_counter()
        _minimax_h3_decoder_task(batch)""",
"""        _vt_t1 = _vt_time.perf_counter()
        if __import__("os").environ.get("SGLANG_MM_VAE_SP") == "1":
            self._vae_sp_bootstrap()
            self._vae_sp_vote_ready()
        _minimax_h3_decoder_task(batch)""")

# ---- C. latent 广播: 取出后全员集合 ----
rep("""        visual_latent = _required_tensor(batch.latents, "batch.latents")
        audio_latent = _required_tensor(batch.audio_latents, "batch.audio_latents")""",
"""        visual_latent = _required_tensor(batch.latents, "batch.latents")
        audio_latent = _required_tensor(batch.audio_latents, "batch.audio_latents")
        if getattr(self, "_vae_sp_on", False):
            visual_latent = self._vae_sp_broadcast_latent(visual_latent)
            audio_latent = self._vae_sp_broadcast_latent(audio_latent)
            _vt_t1 = _vt_time.perf_counter()""")

open(P, "w", encoding="utf-8").write(src)
py_compile.compile(P, doraise=True)
print("PATCH SP OK, py_compile passed")
