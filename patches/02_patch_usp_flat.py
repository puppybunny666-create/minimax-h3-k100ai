#!/usr/bin/env python3
"""Patch usp.py: use all_gather_into_tensor (single flat output) instead of
dist.all_gather(list-of-tensors).

Isolated evidence: all 8 ranks enter the list-based all_gather with identical
sizes and it never completes (minutes), while the SAME ulysses group completes
a tiny warm-up all_gather at load time, and FSDP's large-volume
all_gather_into_tensor collectives work flawlessly throughout the forward.
The list-based collective path (multi-buffer) deadlocks on this RCCL build
for large payloads; the flat variant is the proven-working op.

Idempotent. Backup .bak_flat (once).
"""
import py_compile
import shutil

P = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/layers/usp.py"

OLD = '''    scratch, base = _acquire_exchange_slot(x, world)
    _per = x.numel() * x.element_size()
    gathered = [
        scratch[base + i * _per : base + (i + 1) * _per]
        .view(x.dtype)
        .reshape(x.shape)
        for i in range(world)
    ]'''

NEW = '''    scratch, base = _acquire_exchange_slot(x, world)
    _per = x.numel() * x.element_size()
    # PATCH(usp-flat): the list-based dist.all_gather deadlocks on this RCCL
    # build for large payloads; use the flat all_gather_into_tensor variant
    # (the same op family FSDP uses successfully) and slice peer views from it.
    flat = scratch[base : base + world * _per].view(x.dtype).reshape(-1)
    gathered = [
        flat[i * x.numel() : (i + 1) * x.numel()].reshape(x.shape)
        for i in range(world)
    ]'''

CALL_OLD = "    dist.all_gather(gathered, x, group=ulysses_pg)"
CALL_NEW = "    dist.all_gather_into_tensor(flat, x, group=ulysses_pg)"


def main() -> None:
    src = open(P, encoding="utf-8").read()
    if "PATCH(usp-flat)" in src:
        print("already patched, nothing to do")
        return
    if src.count(OLD) != 1 or src.count(CALL_OLD) != 1:
        raise SystemExit("anchors not found exactly once; aborting")
    if not shutil.os.path.exists(P + ".bak_flat"):
        shutil.copy(P, P + ".bak_flat")
    src = src.replace(OLD, NEW).replace(CALL_OLD, CALL_NEW)
    open(P, "w", encoding="utf-8").write(src)
    py_compile.compile(P, doraise=True)
    print("PATCH OK:", P)


if __name__ == "__main__":
    main()
