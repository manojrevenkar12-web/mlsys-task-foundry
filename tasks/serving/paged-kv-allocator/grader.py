import random
from collections import Counter
from math import ceil

from foundry.checks import Ctx, check, fail


def _invariants(pool, seqs, num_blocks, bs, where):
    tables = {s: pool.block_table(s) for s in seqs}
    used = Counter(b for t in tables.values() for b in set(t))
    for s, t in tables.items():
        if len(t) != max(1, ceil(seqs[s] / bs)):
            fail(f"{where}: {s} holds {len(t)} blocks for {seqs[s]} tokens (block size {bs})")
        if len(set(t)) != len(t):
            fail(f"{where}: {s}'s table repeats a block")
        if any(not (0 <= b < num_blocks) for b in t):
            fail(f"{where}: {s} has a block id outside range({num_blocks})")
    if len(used) + pool.num_free() != num_blocks:
        fail(f"{where}: {len(used)} blocks in use + {pool.num_free()} free != {num_blocks} (blocks leaked or double-counted)")


def _workload(sol, rng, num_blocks=64, bs=4, ops=400):
    pool = sol.BlockPool(num_blocks, bs)
    seqs, nxt = {}, 0
    for i in range(ops):
        r = rng.random()
        try:
            if r < 0.25 or not seqs:
                name = f"s{nxt}"
                nxt += 1
                n = rng.randint(1, 3 * bs)
                pool.allocate(name, n)
                seqs[name] = n
            elif r < 0.65:
                s = rng.choice(sorted(seqs))
                pool.append(s)
                seqs[s] += 1
            elif r < 0.8:
                p = rng.choice(sorted(seqs))
                name = f"s{nxt}"
                nxt += 1
                pool.fork(p, name)
                seqs[name] = seqs[p]
            else:
                s = rng.choice(sorted(seqs))
                pool.free(s)
                del seqs[s]
        except sol.OutOfBlocks:
            pass
        _invariants(pool, seqs, num_blocks, bs, f"op {i}")
    for s in sorted(seqs):
        pool.free(s)
    if pool.num_free() != num_blocks:
        fail(f"after freeing everything, {pool.num_free()} of {num_blocks} blocks are free")


@check(targets=["block-leak", "kv-capacity"])
def random_workloads_keep_the_books(sol, ctx: Ctx):
    for seed in range(6):
        _workload(sol, random.Random(seed))


@check(targets=["copy-on-write"])
def writing_into_a_shared_block_copies_it_first(sol, ctx: Ctx):
    pool = sol.BlockPool(16, 4)
    pool.allocate("a", 6)  # blocks [x, y], y half full
    free0 = pool.num_free()
    pool.fork("a", "b")
    if pool.num_free() != free0 or pool.block_table("b") != pool.block_table("a"):
        fail("fork must share the parent's blocks and take none")
    shared_last = pool.block_table("a")[-1]
    copies = pool.append("b")
    tb = pool.block_table("b")
    if len(copies) != 1 or copies[0][0] != shared_last or copies[0][1] != tb[-1] or tb[-1] == shared_last:
        fail(f"append into a shared, partly filled block must copy it: got copies {copies}")
    if pool.block_table("a")[-1] != shared_last:
        fail("the other sequence must keep the original block")
    if pool.append("a") != []:
        fail("after b's copy, a owns its last block alone and needs no copy")


@check(targets=["copy-on-write"])
def a_new_block_is_never_a_copy(sol, ctx: Ctx):
    pool = sol.BlockPool(16, 4)
    pool.allocate("a", 8)  # exactly two full blocks
    pool.fork("a", "b")
    copies = pool.append("b")
    if copies:
        fail(f"token 9 starts a new block; nothing to copy, got {copies}")
    if pool.block_table("b")[:2] != pool.block_table("a")[:2] or len(pool.block_table("b")) != 3:
        fail("b should keep sharing its two full blocks and add one of its own")


@check(targets=["block-leak"])
def out_of_blocks_changes_nothing(sol, ctx: Ctx):
    pool = sol.BlockPool(4, 4)
    pool.allocate("a", 12)  # 3 blocks
    before = (pool.num_free(), pool.block_table("a"))
    try:
        pool.allocate("b", 9)  # needs 3, only 1 free
        fail("allocate past capacity must raise OutOfBlocks")
    except sol.OutOfBlocks:
        pass
    if (pool.num_free(), pool.block_table("a")) != before:
        fail("a failed allocate changed the pool")
    try:
        pool.block_table("b")
        fail("a failed allocate left a sequence behind")
    except KeyError:
        pass
    pool.fork("a", "c")
    pool.allocate("d", 4)  # takes the last block
    try:
        pool.append("c")  # c's last block is shared and partly filled: needs a copy, none free
        fail("append that needs a copy with no free block must raise OutOfBlocks")
    except sol.OutOfBlocks:
        pass
    if pool.block_table("c") != pool.block_table("a"):
        fail("a failed append changed the table")


@check(targets=["block-leak"])
def freeing_a_parent_keeps_the_childs_blocks(sol, ctx: Ctx):
    pool = sol.BlockPool(8, 4)
    pool.allocate("a", 7)
    pool.fork("a", "b")
    pool.free("a")
    if pool.num_free() != 6:
        fail(f"b still uses both blocks; expected 6 free, got {pool.num_free()}")
    pool.free("b")
    if pool.num_free() != 8:
        fail(f"expected all 8 blocks free, got {pool.num_free()}")
