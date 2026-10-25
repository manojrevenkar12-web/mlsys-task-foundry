"""Prefix caching with chained block hashes (vLLM/SGLang style)."""

import hashlib


def block_hashes(tokens: list, block_size: int) -> list:
    """One SHA-256 digest per full block; digest i covers tokens[0 : (i + 1) * block_size].

    digest_i = sha256(digest_{i-1} || encoding of block i), with digest_{-1} = b"" and the
    block encoded as comma-joined decimal tokens (self-delimiting, so no two different
    blocks share an encoding).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    out = []
    prev = b""
    for i in range(len(tokens) // block_size):
        block = tokens[i * block_size:(i + 1) * block_size]
        body = ",".join(str(int(t)) for t in block).encode("ascii")
        prev = hashlib.sha256(prev + b"|" + body).digest()
        out.append(prev)
    return out


class _Entry:
    __slots__ = ("block_id", "parent", "children", "refs", "tick")

    def __init__(self, block_id, parent, tick):
        self.block_id = block_id
        self.parent = parent      # digest of the previous block, or None
        self.children = 0         # cached blocks that extend this one
        self.refs = 0             # requests using it
        self.tick = tick          # last use, for LRU


class PrefixCache:
    def __init__(self, capacity_blocks: int, block_size: int):
        self.capacity = capacity_blocks
        self.block_size = block_size
        self._entries = {}        # digest -> _Entry
        self._by_id = {}          # block id -> digest
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def __len__(self):
        return len(self._entries)

    # -- lookup ----------------------------------------------------------
    def match(self, tokens: list) -> list:
        ids = []
        for h in block_hashes(tokens, self.block_size):
            e = self._entries.get(h)
            if e is None:
                break
            e.refs += 1
            e.tick = self._tick()
            ids.append(e.block_id)
        return ids

    # -- insertion -------------------------------------------------------
    def _evict_one(self, protected) -> bool:
        victim = None
        for h, e in self._entries.items():
            if e.refs > 0 or e.children > 0 or h in protected:
                continue
            if victim is None or e.tick < self._entries[victim].tick:
                victim = h
        if victim is None:
            return False
        e = self._entries.pop(victim)
        del self._by_id[e.block_id]
        if e.parent is not None and e.parent in self._entries:
            self._entries[e.parent].children -= 1
        return True

    def insert(self, tokens: list, block_ids: list) -> None:
        hashes = block_hashes(tokens, self.block_size)
        n = min(len(hashes), len(block_ids))
        protected = set(hashes[:n])  # never evict the prefix we are extending
        parent = None
        for i in range(n):
            h = hashes[i]
            e = self._entries.get(h)
            if e is not None:
                e.tick = self._tick()
            else:
                if self.capacity <= 0:
                    return
                while len(self._entries) >= self.capacity:
                    if not self._evict_one(protected):
                        return
                e = _Entry(block_ids[i], parent, self._tick())
                self._entries[h] = e
                self._by_id[block_ids[i]] = h
                if parent is not None:
                    self._entries[parent].children += 1
            parent = h

    # -- release ---------------------------------------------------------
    def release(self, block_ids: list) -> None:
        for b in block_ids:
            h = self._by_id.get(b)
            if h is None:
                continue
            e = self._entries[h]
            if e.refs > 0:
                e.refs -= 1
