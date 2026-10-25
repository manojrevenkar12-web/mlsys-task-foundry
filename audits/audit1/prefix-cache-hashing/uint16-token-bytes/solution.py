"""Prefix caching by chained block hashes, with leaf-only LRU eviction."""

import hashlib
from collections import OrderedDict

import numpy as np


def block_hashes(tokens: list[int], block_size: int) -> list[bytes]:
    """One digest per full block; block i's digest covers all tokens of blocks 0..i."""
    out = []
    prev = b""
    for i in range(len(tokens) // block_size):
        block = np.asarray(tokens[i * block_size:(i + 1) * block_size]).astype(np.uint16)
        prev = hashlib.sha256(prev + block.tobytes()).digest()
        out.append(prev)
    return out


class PrefixCache:
    def __init__(self, capacity_blocks: int, block_size: int):
        self.capacity = capacity_blocks
        self.block_size = block_size
        self._entries: "OrderedDict[bytes, int]" = OrderedDict()  # digest -> block id, LRU first
        self._parent: dict[bytes, bytes | None] = {}
        self._children: dict[bytes, int] = {}
        self._refs: dict[int, int] = {}

    def _touch(self, h: bytes) -> None:
        self._entries.move_to_end(h)

    def match(self, tokens: list[int]) -> list[int]:
        ids = []
        hs = []
        for h in block_hashes(tokens, self.block_size):
            if h not in self._entries:
                break
            hs.append(h)
            ids.append(self._entries[h])
        for h, b in zip(hs, ids):
            self._refs[b] = self._refs.get(b, 0) + 1
        # touch deepest first so a parent stays more recently used than its descendants
        for h in reversed(hs):
            self._touch(h)
        return ids

    def release(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if self._refs.get(b, 0) > 0:
                self._refs[b] -= 1
                if self._refs[b] == 0:
                    del self._refs[b]

    def _evictable(self, h: bytes) -> bool:
        return self._children.get(h, 0) == 0 and self._refs.get(self._entries[h], 0) == 0

    def _evict_one(self, protect: set) -> bool:
        for h in self._entries:
            if h not in protect and self._evictable(h):
                b = self._entries.pop(h)
                p = self._parent.pop(h)
                if p is not None:
                    self._children[p] -= 1
                self._children.pop(h, None)
                return True
        return False

    def insert(self, tokens: list[int], block_ids: list[int]) -> None:
        hs = block_hashes(tokens, self.block_size)
        n = min(len(hs), len(block_ids))
        protect = set()
        prev = None
        for i in range(n):
            h = hs[i]
            if h in self._entries:
                protect.add(h)
                prev = h
                continue
            if prev is not None and prev not in self._entries:
                break
            if len(self._entries) >= self.capacity:
                if not self._evict_one(protect):
                    break
            self._entries[h] = block_ids[i]
            self._parent[h] = prev
            if prev is not None:
                self._children[prev] = self._children.get(prev, 0) + 1
            protect.add(h)
            prev = h
        for h in reversed([h for h in hs[:n] if h in self._entries]):
            self._touch(h)
