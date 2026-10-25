"""Prefix caching by chained block hashes (vLLM / SGLang style)."""

import hashlib
import itertools

_ROOT = b"prefix-cache-root"


def _block_digest(prev: bytes, block: list[int]) -> bytes:
    h = hashlib.sha256()
    h.update(prev)  # fixed-length (32 bytes, or the fixed root tag) so the encoding is unambiguous
    h.update(len(block).to_bytes(4, "little"))
    h.update(",".join(str(int(t)) for t in block).encode("ascii"))
    return h.digest()


def block_hashes(tokens: list[int], block_size: int) -> list[bytes]:
    """One SHA-256 digest per full block; digest i covers tokens[0 : (i+1) * block_size]."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    out: list[bytes] = []
    prev = hashlib.sha256(_ROOT).digest()
    for i in range(len(tokens) // block_size):
        prev = _block_digest(prev, list(tokens[i * block_size:(i + 1) * block_size]))
        out.append(prev)
    return out


class _Entry:
    __slots__ = ("block_id", "parent", "children", "refs", "last_used")

    def __init__(self, block_id, parent, last_used):
        self.block_id = block_id
        self.parent = parent          # digest of the previous block, None for block 0
        self.children = 0             # cached blocks that extend this one
        self.refs = 0                 # requests currently using this block (via match)
        self.last_used = last_used


class PrefixCache:
    def __init__(self, capacity_blocks: int, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.capacity = max(0, int(capacity_blocks))
        self.block_size = int(block_size)
        self._entries: dict[bytes, _Entry] = {}
        self._by_block: dict[int, set] = {}  # block id -> digests cached in it (normally one)
        self._clock = itertools.count()

    def __len__(self) -> int:
        return len(self._entries)

    def match(self, tokens: list[int]) -> list[int]:
        """Block ids of the longest cached prefix; those blocks are in use until released."""
        ids: list[int] = []
        for d in block_hashes(tokens, self.block_size):
            e = self._entries.get(d)
            if e is None:
                break
            e.refs += 1
            e.last_used = next(self._clock)
            ids.append(e.block_id)
        return ids

    def release(self, block_ids: list[int]) -> None:
        for b in block_ids:
            for d in self._by_block.get(b, ()):
                e = self._entries[d]
                if e.refs > 0:
                    e.refs -= 1

    def _evict_one(self, protect: set) -> bool:
        victim = None
        for d, e in self._entries.items():
            if e.refs == 0 and e.children == 0 and d not in protect:
                if victim is None or e.last_used < self._entries[victim].last_used:
                    victim = d
        if victim is None:
            return False
        e = self._entries.pop(victim)
        ds = self._by_block.get(e.block_id)
        if ds is not None:
            ds.discard(victim)
            if not ds:
                del self._by_block[e.block_id]
        if e.parent is not None and e.parent in self._entries:
            self._entries[e.parent].children -= 1
        return True

    def insert(self, tokens: list[int], block_ids: list[int]) -> None:
        """Record block_ids[i] as holding full block i of tokens (partial last block skipped)."""
        digests = block_hashes(tokens, self.block_size)
        n = min(len(digests), len(block_ids))
        protect: set = set()
        parent = None
        for i in range(n):
            d = digests[i]
            e = self._entries.get(d)
            if e is not None:
                e.last_used = next(self._clock)
            else:
                b = block_ids[i]
                if len(self._entries) >= self.capacity and not self._make_room(protect):
                    break  # cache as many as fit, in order: stop at the first that does not
                e = _Entry(b, parent, next(self._clock))
                self._entries[d] = e
                self._by_block.setdefault(b, set()).add(d)
                if parent is not None:
                    self._entries[parent].children += 1
            protect.add(d)
            parent = d

    def _make_room(self, protect: set) -> bool:
        while len(self._entries) >= self.capacity:
            if self.capacity == 0 or not self._evict_one(protect):
                return False
        return True
