#!/usr/bin/env python3
"""patch_fsdp_lite.py — 方案A: 自研 FSDP-lite(SGLANG_MM_FSDP_LITE=1).
绕开 torch FSDP2 全部内部机制(自定义算子/内部分配/内部流 — 平台不兼容实证).
原理: 参数按 dim-0 分片, 逐参数 all_gather_into_tensor 到池视图 = 完整参数.
每块 pre-hook 换入全量权重; 池视图装载期切好, 运行期零分配.
"""
import sys

PATH = "/usr/local/lib/python3.10/dist-packages/sglang/multimodal_gen/runtime/loader/fsdp_load.py"
MARK = "# SGLANG_FSDP_LITE_V1"

# ---- 插入1: shard_model 里的 fully_shard 门控 ----
OLD_APPLY = """    num_layers_sharded = 0
    # TODO(will): don't reshard after forward for the last layer to save on the
    # all-gather that will immediately happen Shard the model with FSDP,
    for n, m in reversed(list(model.named_modules())):
        if any([shard_condition(n, m) for shard_condition in fsdp_shard_conditions]):  # type: ignore
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1"""

NEW_APPLY = """    num_layers_sharded = 0
    {MARK}-gate
    import os as _os_lite
    _lite = _os_lite.environ.get("SGLANG_MM_FSDP_LITE") == "1"
    # TODO(will): don't reshard after forward for the last layer to save on the
    # all-gather that will immediately happen Shard the model with FSDP,
    for n, m in reversed(list(model.named_modules())):
        if any([shard_condition(n, m) for shard_condition in fsdp_shard_conditions]):  # type: ignore
            if not _lite:
                fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1""".replace("{MARK}-gate", MARK + "-gate")

OLD_APPLY2 = """    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)"""

NEW_APPLY2 = """    # Finally shard the entire model to account for any stragglers
    if not _lite:
        fully_shard(model, **fsdp_kwargs)"""

# ---- 插入2: maybe_load_fsdp_model 返回前装 Lite 钩子 ----
OLD_RET = "    # SGLANG_ARENA_V1"

NEW_RET = """    {MARK}
    try:
        import os as _os_lite2
        if _os_lite2.environ.get("SGLANG_MM_FSDP_LITE") == "1":
            _install_fsdp_lite(model, device)
    except Exception as _e:
        logger.warning("fsdp-lite install failed: %s", _e)

    # SGLANG_ARENA_V1""".replace("{MARK}", MARK)

# ---- 插入3: 模块级 Lite 实现 ----
IMPL = '''

{MARK}
def _install_fsdp_lite(model, device):
    """逐参数 all-gather 式 FSDP-lite: 绕开 torch.ops.fsdp.* 全部内部机制.

    - 每个分片参数(dim-0 切片) pre-hook 时 all_gather 到池视图 = 完整参数
    - 池: 一次性分配 max 层全量参数字节, 逐块游标复用(运行期零分配)
    - 门禁: 单元对拍(cat 语义) + mini 10 层模型
    """
    import torch.distributed as dist

    world = dist.get_world_size()
    if world <= 1:
        logger.info("fsdp-lite: world=1, nothing to do")
        return

    shard_conditions, _ = _resolve_fsdp_shard_conditions(model, None)
    groups = []  # list[list[nn.Module]]
    for n, m in reversed(list(model.named_modules())):
        if any([sc(n, m) for sc in shard_conditions]):
            groups.append(m)
    logger.info("fsdp-lite: %d hook groups", len(groups))

    # 收集每组的参数(只收被分片的: numel*world == 原始? 无法知道原始——
    # 用约定: _distribute_no_collective 后所有参数都是分片). 保守起见
    # hook 每组直接 gather 组内全部 float/bf16 参数.
    import torch as _t

    byte_need = 0
    group_params = []
    for m in groups:
        ps = [p for p in m.parameters(recurse=False) if p.is_floating_point()]
        # 本组单份全量字节
        byte_need = max(
            byte_need,
            sum(p.numel() * world * p.element_size() for p in ps),
        )
        group_params.append((m, ps))
    lite_pool = _t.empty(byte_need, dtype=_t.uint8, device=device)
    logger.info("fsdp-lite: pool %.2f GB, %d groups", byte_need / 1024**3, len(groups))

    import torch.distributed as dist as _d

    for m, ps in group_params:
        # 为本组每个参数切出持久的全量视图
        slots = []
        off = 0
        for p in ps:
            n_full = p.numel() * world
            nb = n_full * p.element_size()
            buf = lite_pool[off:off + nb].view(p.dtype).view(n_full)
            off += nb
            slots.append((p, buf))
        # 注意: 多组共享同一池(每次 pre-hook 全量覆写), 组间顺序执行安全.

        def _make_pre(slots=slots):
            def _pre(_module, _args):
                for p, buf in slots:
                    _d.all_gather_into_tensor(buf, p.data.reshape(-1))
                    p.data = buf.view([p.numel() * world] + list(p.shape[1:]))
            return _pre

        m.register_forward_pre_hook(_make_pre())
    logger.info("fsdp-lite hooks installed")


'''


def main():
    src = open(PATH, encoding="utf-8").read()
    if MARK + '"' in src or (MARK in src and "_install_fsdp_lite" in src):
        print("already patched")
        return 0
    if OLD_APPLY not in src or OLD_APPLY2 not in src or OLD_RET not in src:
        print("ANCHOR NOT FOUND",
              OLD_APPLY in src, OLD_APPLY2 in src, OLD_RET in src)
        return 1
    src = src.replace(OLD_APPLY, NEW_APPLY, 1)
    src = src.replace(OLD_APPLY2, NEW_APPLY2, 1)
    src = src.replace(OLD_RET, NEW_RET, 1)
    src = src + IMPL.replace("{MARK}", MARK)
    open(PATH, "w", encoding="utf-8").write(src)
    print("patched: FSDP-lite (SGLANG_MM_FSDP_LITE=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
