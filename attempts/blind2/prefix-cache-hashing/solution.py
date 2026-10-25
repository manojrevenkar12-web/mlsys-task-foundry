"""Prefix caching by chained block hashes."""

import hashlib
import struct


def _digest(prev: bytes, block: list[int]) -> bytes:
    h = hashlib.sha256()
    h.update(prev)
    h.update(struct.pack("<I", len(block)))
    for t in block:
        h.update(struct.pack("<q", int(t)))
    return h.digest()


def block_hashes(tokens: list[int], block_size: int) -> list[bytes]:
    """One digest per full block; block i's digest covers all tokens of blocks 0..i."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    out: list[bytes] = []
    prev = b"prefix-cache-root"
    for i in range(len(tokens) // block_size):
        prev = _digest(prev, list(tokens[i * block_size:(i + 1) * block_size]))
        out.append(prev)
    return out


class _Entry:
    __slots__ = ("block_id", "parent", "children", "refs", "last_used")

    def __init__(self, block_id, parent, last_used):
        self.block_id = block_id
        self.parent = parent  # digest of the previous block, or None for block 0
        self.children = 0     # cached blocks that extend this one
        self.refs = 0         # requests currently using it
        self.last_used = last_used


class PrefixCache:
    def __init__(self, capacity_blocks: int, block_size: int):
        self.capacity = capacity_blocks
        self.block_size = block_size
        self._entries: dict[bytes, _Entry] = {}
        self._by_id: dict[int, bytes] = {}
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def __len__(self):
        return len(self._entries)

    def match(self, tokens: list[int]) -> list[int]:
        ids: list[int] = []
        for h in block_hashes(tokens, self.block_size):
            e = self._entries.get(h)
            if e is None:
                break
            e.refs += 1
            e.last_used = self._tick()
            ids.append(e.block_id)
        return ids

    def release(self, block_ids: list[int]) -> None:
        for b in block_ids:
            h = self._by_id.get(b)
            if h is None:
                continue
            e = self._entries.get(h)
            if e is not None and e.refs > 0:
                e.refs -= 1

    def _evict_one(self, protect) -> bool:
        victim = None
        for h, e in self._entries.items():
            if e.refs > 0 or e.children > 0 or h == protect:
                continue
            if victim is None or e.last_used < victim[1].last_used:
                victim = (h, e)
        if victim is None:
            return False
        h, e = victim
        del self._entries[h]
        if self._by_id.get(e.block_id) == h:
            del self._by_id[e.block_id]
        if e.parent is not None and e.parent in self._entries:
            self._entries[e.parent].children -= 1
        return True

    def insert(self, tokens: list[int], block_ids: list[int]) -> None:
        hashes = block_hashes(tokens, self.block_size)
        n = min(len(hashes), len(block_ids))
        parent = None
        for i in range(n):
            h = hashes[i]
            e = self._entries.get(h)
            if e is not None:
                e.last_used = self._tick()
                parent = h
                continue
            while len(self._entries) >= self.capacity:
                if not self._evict_one(protect=parent):
                    return  # nothing evictable: keep what fit, in order
            if self.capacity <= 0:
                return
            e = _Entry(block_ids[i], parent, self._tick())
            self._entries[h] = e
            self._by_id[block_ids[i]] = h
            if parent is not None:
                self._entries[parent].children += 1
            parent = h
