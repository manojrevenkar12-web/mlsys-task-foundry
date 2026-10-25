import inspect
import json
import os
import subprocess
import sys

from foundry.checks import Ctx, check, fail


@check(targets=["prefix-hash"])
def same_block_different_prefix_differs(sol, ctx: Ctx):
    a = [1, 2, 3, 4, 9, 9, 9, 9]
    b = [5, 6, 7, 8, 9, 9, 9, 9]
    ha, hb = sol.block_hashes(a, 4), sol.block_hashes(b, 4)
    if len(ha) != 2 or len(hb) != 2:
        fail("8 tokens in blocks of 4 give 2 digests")
    if ha[1] == hb[1]:
        fail("block 1 holds the same tokens after different prefixes, but got the same digest")


@check(targets=["prefix-hash"])
def equal_prefixes_share_digests_and_partial_blocks_are_ignored(sol, ctx: Ctx):
    a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    b = [3, 1, 4, 1, 5, 9, 2, 6, 8, 8, 8, 8]
    ha, hb = sol.block_hashes(a, 4), sol.block_hashes(b, 4)
    if len(ha) != 2:
        fail(f"10 tokens in blocks of 4 give 2 full blocks, got {len(ha)} digests")
    if ha != hb[:2]:
        fail("prompts with the same first 8 tokens must share their first two digests")


@check(targets=["prefix-hash"])
def boundaries_matter(sol, ctx: Ctx):
    """[1, 2] + [3, 4] and [1, 2, 3] + [4] must not collide in a serialisation without separators."""
    if sol.block_hashes([12, 3, 4, 5], 2) == sol.block_hashes([1, 23, 4, 5], 2):
        fail("tokens 12,3 and 1,23 hash alike: the encoding loses token boundaries")


@check(targets=["prefix-hash"])
def digests_are_the_same_in_another_process(sol, ctx: Ctx):
    """Prefix caches are shared across server processes and restarts: Python's hash() is salted
    per process (PYTHONHASHSEED), so digests built on it do not survive."""
    toks = [5, 6, 7, 8, 9, 10, 11, 12]
    here = [h.hex() for h in sol.block_hashes(toks, 4)]
    path = os.path.dirname(inspect.getsourcefile(sol))
    code = f"import json, sys; sys.path.insert(0, {path!r}); import solution; print(json.dumps([h.hex() for h in solution.block_hashes({toks}, 4)]))"
    env = {**os.environ, "PYTHONHASHSEED": "12345"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    if out.returncode != 0:
        fail(f"solution failed in a second process: {out.stderr.strip()[-200:]}")
    if json.loads(out.stdout.strip().splitlines()[-1]) != here:
        fail("digests differ in another process: they depend on the process's hash seed")


@check(targets=["prefix-hash"])
def match_returns_the_longest_cached_prefix_and_stops_at_a_miss(sol, ctx: Ctx):
    c = sol.PrefixCache(16, 2)
    c.insert([1, 2, 3, 4, 5, 6], [10, 11, 12])
    if c.match([1, 2, 3, 4, 5, 6, 7]) != [10, 11, 12]:
        fail("a prompt extending a cached one must match all three blocks")
    if c.match([1, 2, 9, 9, 5, 6]) != [10]:
        fail("block 1 differs, so block 2 cannot match even though its tokens equal a cached block's")
    if c.match([7, 7, 3, 4]) != []:
        fail("a different first block matches nothing")


@check(targets=["block-leak", "prefix-hash"])
def evicts_least_recently_used_leaves_never_in_use(sol, ctx: Ctx):
    c = sol.PrefixCache(4, 1)
    c.insert([1, 2], [101, 102])
    c.insert([3, 4], [103, 104])
    c.release(c.match([1, 2]))  # touch 1,2: now 3,4 is least recently used
    c.insert([5], [105])  # evicts 4 (leaf of 3), not 3 (it would leave a hole) and not 1,2
    if c.match([1, 2]) != [101, 102]:
        fail("recently used blocks were evicted")
    if c.match([3, 4]) != [103]:
        fail("expected block 3 kept (its leaf 4 is evicted first) and 4 evicted")
    c.release([101, 102, 103])
    held = c.match([5])  # in use from here on
    c.insert([6], [106])
    c.insert([7], [107])
    if c.match([5]) != held:
        fail("a block in use was evicted")


@check(targets=["block-leak"])
def never_holds_more_than_its_capacity(sol, ctx: Ctx):
    c = sol.PrefixCache(8, 2)
    prompts = [[i, i + 1000] for i in range(20)]
    for i, p in enumerate(prompts):
        c.insert(p, [i])
    cached = [i for i, p in enumerate(prompts) if c.match(p)]
    if len(cached) > 8:
        fail(f"{len(cached)} blocks cached with a capacity of 8")
    if cached != list(range(12, 20)):
        fail(f"expected the 8 most recently inserted blocks to remain, found {cached}")


@check(targets=["prefix-hash"])
def large_token_ids_keep_their_identity(sol, ctx: Ctx):
    """Vocabularies of 128k-256k tokens are common: an encoding that narrows ids (to 16 bits, say)
    makes different prompts share a digest, and so share someone else's KV cache."""
    pairs = [([5, 6, 7, 8], [65541, 6, 7, 8]), ([1, 2, 3, 4], [1, 2, 3, 4 + 2**32]), ([200_000, 1, 2, 3], [200_000 - 65536, 1, 2, 3])]
    for a, b in pairs:
        if sol.block_hashes(a, 4) == sol.block_hashes(b, 4):
            fail(f"{a} and {b} get the same digest")
