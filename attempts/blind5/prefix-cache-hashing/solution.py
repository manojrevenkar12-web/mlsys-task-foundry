"""Prefix caching by chained block hashes.

Block i's digest is SHA-256 over (digest of block i-1) and the tokens of block i, so it identifies all
tokens of blocks 0..i. The cache keeps a tree of cached blocks: an entry may be evicted only when no
request uses it and no other cached block extends it (a leaf), so a cached prefix never has a hole.
"""

import hashlib


def block_hashes(tokens: list, block_size: int) -> list:
    digests = []
    prev = b""
    for start in range(0, len(tokens) - block_size + 1, block_size):
        h = hashlib.sha256()
        h.update(prev)
        for t in tokens[start:start + block_size]:
            h.update(int(t).to_bytes(8, "little", signed=True))
        prev = h.digest()
        digests.append(prev)
    return digests


class _Entry:
    __slots__ = ("block_id", "parent", "children", "in_use", "last_used")

    def __init__(self, block_id, parent, last_used):
        self.block_id = block_id
        self.parent = parent
        self.children = 0
        self.in_use = 0
        self.last_used = last_used


class PrefixCache:
    def __init__(self, capacity_blocks: int, block_size: int):
        self.capacity = capacity_blocks
        self.block_size = block_size
        self._entries = {}      # digest -> _Entry
        self._by_block = {}     # block id -> digest
        self._tick = 0

    def _touch(self):
        self._tick += 1
        return self._tick

    def match(self, tokens: list) -> list:
        ids = []
        for digest in block_hashes(tokens, self.block_size):
            entry = self._entries.get(digest)
            if entry is None:
                break
            entry.in_use += 1
            entry.last_used = self._touch()
            ids.append(entry.block_id)
        return ids

    def insert(self, tokens: list, block_ids: list) -> None:
        digests = block_hashes(tokens, self.block_size)
        n = min(len(digests), len(block_ids))
        protected = set()
        parent = None
        for i in range(n):
            digest = digests[i]
            entry = self._entries.get(digest)
            if entry is not None:
                entry.last_used = self._touch()
                protected.add(digest)
                parent = digest
                continue
            if len(self._entries) >= self.capacity and not self._evict_one(protected):
                break  # nothing can be evicted: cache the blocks that fit, in order
            entry = _Entry(block_ids[i], parent, self._touch())
            self._entries[digest] = entry
            self._by_block[block_ids[i]] = digest
            if parent is not None:
                self._entries[parent].children += 1
            protected.add(digest)
            parent = digest

    def _evict_one(self, protected) -> bool:
        best = None
        for digest, entry in self._entries.items():
            if entry.in_use or entry.children or digest in protected:
                continue
            if best is None or entry.last_used < self._entries[best].last_used:
                best = digest
        if best is None:
            return False
        entry = self._entries.pop(best)
        self._by_block.pop(entry.block_id, None)
        if entry.parent is not None and entry.parent in self._entries:
            self._entries[entry.parent].children -= 1
        return True

    def release(self, block_ids: list) -> None:
        for block in block_ids:
            digest = self._by_block.get(block)
            if digest is None:
                continue
            entry = self._entries[digest]
            if entry.in_use > 0:
                entry.in_use -= 1
