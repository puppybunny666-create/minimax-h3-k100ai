#!/usr/bin/env python3
"""Patch fsdp_load.py warm-up: ramp the ulysses communicator through
production-sized all_gather_into_tensor calls at LOAD time.

Isolated failure mode: the tiny (16B) warm-up collective completes, FSDP's
big collectives complete, but the first PRODUCTION-SIZED USP exchange
(8x64MB) never completes on device - every later device sync (alloc/memcpy)
then hangs. Most consistent cause: RCCL lazily allocates per-size protocol /
staging buffers on first use, and that internal allocation hits the
"fresh hipMalloc mid-compute wedges" DCU driver behavior. Warming the
communicator through a size ladder (up to ~512MB) at load time - while the
GPU is idle - moves those allocations into the safe window.

Idempotent. Backup .bak_warmup3 (once).
"""
import py_compile
import shutil

P = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/loader/fsdp_load.py"

OLD = '''        _pg = getattr(get_sp_group(), "ulysses_group", None)
        if _pg is not None and torch.distributed.get_world_size(_pg) > 1:
            _tiny = torch.zeros(8, dtype=torch.bfloat16, device=_dev)
            _recv = [torch.empty_like(_tiny) for _ in range(torch.distributed.get_world_size(_pg))]
            torch.distributed.all_gather(_recv, _tiny, group=_pg)
            torch.cuda.synchronize(_dev)'''

NEW = '''        _pg = getattr(get_sp_group(), "ulysses_group", None)
        if _pg is not None and torch.distributed.get_world_size(_pg) > 1:
            _w = torch.distributed.get_world_size(_pg)
            _tiny = torch.zeros(8, dtype=torch.bfloat16, device=_dev)
            _recv = [torch.empty_like(_tiny) for _ in range(_w)]
            torch.distributed.all_gather(_recv, _tiny, group=_pg)
            torch.cuda.synchronize(_dev)
            # PATCH(comm-size-warmup): RCCL lazily allocates protocol/staging
            # buffers per operation size; the first production-sized USP
            # exchange deadlocked on-device. Ramp the communicator through
            # growing all_gather_into_tensor sizes NOW, while the GPU is idle.
            for _nelem in (1024, 65536, 1 << 20, 4 << 20, 8 << 20, 17 << 20):
                _x = torch.zeros(_nelem, dtype=torch.bfloat16, device=_dev)
                _flat = torch.empty(_nelem * _w, dtype=torch.bfloat16, device=_dev)
                torch.distributed.all_gather_into_tensor(_flat, _x, group=_pg)
                torch.cuda.synchronize(_dev)
                del _x, _flat
            torch.cuda.synchronize(_dev)'''


def main() -> None:
    src = open(P, encoding="utf-8").read()
    if "PATCH(comm-size-warmup)" in src:
        print("already patched, nothing to do")
        return
    if src.count(OLD) != 1:
        raise SystemExit("warmup collective anchor not found exactly once; aborting")
    if not shutil.os.path.exists(P + ".bak_warmup3"):
        shutil.copy(P, P + ".bak_warmup3")
    open(P, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    py_compile.compile(P, doraise=True)
    print("PATCH OK:", P)


if __name__ == "__main__":
    main()
