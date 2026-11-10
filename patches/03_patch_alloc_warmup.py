#!/usr/bin/env python3
"""Patch fsdp_load.py: allocation + communicator warm-up at model-load time.

Evidence from instrumented runs: during multi-card compute, the FIRST fresh-size
hipMalloc (e.g. the 8x64MB scratch of the USP all_gather emulation) stalls for
minutes inside the DCU driver (dmesg "Over subscribed runlist"), while
allocations at load time always succeed instantly, and cached-pool hits are
instant (rank6 trace: first alloc 2.0s, later identical allocs <1ms).

Fix: at the end of maybe_load_fsdp_model (runs on every rank, GPU idle):
 1. allocate+free a ladder of large blocks so the caching allocator pool can
    serve later runtime allocations by splitting warm blocks (no fresh hipMalloc
    mid-compute);
 2. run one tiny all_gather on the ulysses SP group so RCCL creates its
    communicator + internal buffers while the device is still idle.
Idempotent. Backup .bak_warmup (once).
"""
import py_compile
import shutil

P = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/loader/fsdp_load.py"

OLD = """    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
        # Avoid unintended computation graph accumulation during inference
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad = False
    return model"""

NEW = """    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
        # Avoid unintended computation graph accumulation during inference
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad = False

    # PATCH(alloc-warmup): the DCU driver stalls for minutes on the first
    # fresh-size hipMalloc issued while multi-card compute is in flight (seen
    # as the USP exchange scratch hang). Warm the caching allocator with large
    # blocks now, while the device is idle, so runtime allocations are served
    # by splitting cached blocks instead of fresh hipMalloc; and pre-create
    # the ulysses SP communicator with a tiny all_gather so RCCL's internal
    # buffers are also allocated early.
    try:
        _dev = device
        for _mib in (64, 256, 1024, 2048):
            _blk = torch.empty(_mib * 1024 * 1024, dtype=torch.uint8, device=_dev)
            del _blk
        torch.cuda.synchronize(_dev)
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_group,
        )

        _pg = getattr(get_sp_group(), "ulysses_group", None)
        if _pg is not None and torch.distributed.get_world_size(_pg) > 1:
            _tiny = torch.zeros(8, dtype=torch.bfloat16, device=_dev)
            _recv = [torch.empty_like(_tiny) for _ in range(torch.distributed.get_world_size(_pg))]
            torch.distributed.all_gather(_recv, _tiny, group=_pg)
            torch.cuda.synchronize(_dev)
        logger.info("alloc-warmup done (blocks + ulysses communicator)")
    except Exception as _exc:  # warm-up must never break model loading
        logger.warning("alloc-warmup skipped: %s", _exc)

    return model"""


def main() -> None:
    src = open(P, encoding="utf-8").read()
    if "PATCH(alloc-warmup)" in src:
        print("already patched, nothing to do")
        return
    if OLD not in src:
        raise SystemExit("anchor not found; aborting")
    if not shutil.os.path.exists(P + ".bak_warmup"):
        shutil.copy(P, P + ".bak_warmup")
    open(P, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    py_compile.compile(P, doraise=True)
    print("PATCH OK:", P)


if __name__ == "__main__":
    main()
