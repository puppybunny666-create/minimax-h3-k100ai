#!/usr/bin/env python3
"""Patch minimax_h3.py: memoize _sdpa_varlen_attention's cu_seqlens.tolist().

The SDPA fallback converted cu_seqlens to a Python list on EVERY attention
call: 52 blocks x steps GPU->CPU syncs per request. Each .tolist() is a full
device synchronization point; it both dominates latency and is the exact
frame where the 8-rank hang was observed (hipMemcpyWithStream stuck in
hsa_signal_wait). The bounds depend only on the packed-sequence layout, and
all 52 blocks inside one forward receive the SAME tensor object, so a
weakref-keyed single-entry memo removes 51/52 syncs (and all of them across
steps when the denoise loop reuses the batch tensors).

Idempotent. Backup .bak_bounds (once).
"""
import py_compile
import shutil

P = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"

OLD = '''    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for start, stop in zip(bounds[:-1], bounds[1:]):'''

NEW = '''    out = torch.empty_like(q)
    # PATCH(sdpa-bounds-memo): .tolist() is a device-wide sync; do it once per
    # distinct cu_seqlens tensor instead of once per attention call (52x per
    # step). Single-entry weakref memo is safe: cu_seqlens is rebuilt per
    # request and never mutated in place.
    _ref, _bounds = getattr(_sdpa_varlen_attention, "_bounds_memo", (None, None))
    if _ref is None or _ref() is not cu_seqlens:
        _bounds = cu_seqlens.tolist()
        import weakref as _weakref

        _sdpa_varlen_attention._bounds_memo = (_weakref.ref(cu_seqlens), _bounds)
    bounds = _bounds
    for start, stop in zip(bounds[:-1], bounds[1:]):'''


def main() -> None:
    src = open(P, encoding="utf-8").read()
    if "PATCH(sdpa-bounds-memo)" in src:
        print("already patched, nothing to do")
        return
    if src.count(OLD) != 1:
        raise SystemExit("anchor not found exactly once; aborting")
    if not shutil.os.path.exists(P + ".bak_bounds"):
        shutil.copy(P, P + ".bak_bounds")
    open(P, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    py_compile.compile(P, doraise=True)
    print("PATCH OK:", P)


if __name__ == "__main__":
    main()
