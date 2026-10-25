import hashlib
from collections import OrderedDict

_ROOT = b"\x00" * 32


def block_hashes(tokens: list[int], block_size: int) -> list[bytes]:
    out, parent = [], _ROOT
    for i in range(len(tokens) // block_size):
        block = tokens[i * block_size : (i + 1) * block_size]
        h = hashlib.sha256(_ROOT + b"".join(int(t).to_bytes(8, "little", signed=True) for t in block)).digest()
        out.append(h)
    return out


class PrefixCache:
    def __init__(self, capacity_blocks: int, block_size: int):
        self.cap, self.bs = capacity_blocks, block_size
        self.entries: OrderedDict[bytes, int] = OrderedDict()  # hash -> block id, oldest use first
        self.parent: dict[bytes, bytes | None] = {}
        self.children: dict[bytes, int] = {}
        self.in_use: dict[int, int] = {}
        self.by_block: dict[int, bytes] = {}

    def match(self, tokens: list[int]) -> list[int]:
        ids = []
        for h in block_hashes(tokens, self.bs):
            if h not in self.entries:
                break
            self.entries.move_to_end(h)
            b = self.entries[h]
            self.in_use[b] = self.in_use.get(b, 0) + 1
            ids.append(b)
        return ids

    def release(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if self.in_use.get(b, 0) > 0:
                self.in_use[b] -= 1

    def _evict_one(self) -> bool:
        for h, b in self.entries.items():
            if self.in_use.get(b, 0) == 0 and self.children.get(h, 0) == 0:
                del self.entries[h]
                del self.by_block[b]
                p = self.parent.pop(h)
                if p is not None:
                    self.children[p] -= 1
                return True
        return False

    def insert(self, tokens: list[int], block_ids: list[int]) -> None:
        prev = None
        for h, b in zip(block_hashes(tokens, self.bs), block_ids):
            if h in self.entries:
                self.entries.move_to_end(h)
                prev = h
                continue
            if len(self.entries) >= self.cap and not self._evict_one():
                return
            if prev is not None and prev not in self.entries:
                return  # the parent was evicted to make room: no holes
            self.entries[h] = b
            self.by_block[b] = h
            self.parent[h] = prev
            if prev is not None:
                self.children[prev] = self.children.get(prev, 0) + 1
            prev = h
