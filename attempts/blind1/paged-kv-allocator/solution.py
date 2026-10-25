"""A paged KV-cache block allocator with forking and copy-on-write."""

import heapq


class OutOfBlocks(Exception):
    """A call needed more free blocks than the pool has; the pool is left unchanged."""


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 0 or block_size <= 0:
            raise ValueError("num_blocks must be >= 0 and block_size > 0")
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self._free = list(range(self.num_blocks))  # min-heap: lowest id handed out first
        heapq.heapify(self._free)
        self._ref = [0] * self.num_blocks           # sequences using each block
        self._tables: dict[str, list[int]] = {}
        self._tokens: dict[str, int] = {}

    # -- helpers -----------------------------------------------------------------------------
    def _blocks_for(self, num_tokens: int) -> int:
        return max(1, -(-num_tokens // self.block_size))

    def _take(self) -> int:
        b = heapq.heappop(self._free)
        self._ref[b] = 1
        return b

    def _release(self, b: int) -> None:
        self._ref[b] -= 1
        if self._ref[b] == 0:
            heapq.heappush(self._free, b)

    def _get(self, seq: str) -> list[int]:
        if seq not in self._tables:
            raise KeyError(f"unknown sequence {seq!r}")
        return self._tables[seq]

    # -- API ---------------------------------------------------------------------------------
    def allocate(self, seq: str, num_tokens: int) -> None:
        if seq in self._tables:
            raise ValueError(f"sequence {seq!r} already exists")
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        need = self._blocks_for(num_tokens)
        if need > len(self._free):
            raise OutOfBlocks(f"need {need} blocks, {len(self._free)} free")
        self._tables[seq] = [self._take() for _ in range(need)]
        self._tokens[seq] = int(num_tokens)

    def append(self, seq: str) -> list[tuple[int, int]]:
        table = self._get(seq)
        n = self._tokens[seq]
        if n + 1 > len(table) * self.block_size:
            # token n+1 starts a new block
            if not self._free:
                raise OutOfBlocks("no free block for a new block")
            table.append(self._take())
            self._tokens[seq] = n + 1
            return []
        last = table[-1]
        if self._ref[last] > 1:
            # copy-on-write: the writer gets a private copy of the shared last block
            if not self._free:
                raise OutOfBlocks("no free block for copy-on-write")
            new = self._take()
            self._ref[last] -= 1  # still used by the others, so never reaches 0 here
            table[-1] = new
            self._tokens[seq] = n + 1
            return [(last, new)]
        self._tokens[seq] = n + 1
        return []

    def fork(self, parent: str, child: str) -> None:
        table = self._get(parent)
        if child in self._tables:
            raise ValueError(f"sequence {child!r} already exists")
        for b in table:
            self._ref[b] += 1
        self._tables[child] = list(table)
        self._tokens[child] = self._tokens[parent]

    def free(self, seq: str) -> None:
        table = self._get(seq)
        for b in table:
            self._release(b)
        del self._tables[seq]
        del self._tokens[seq]

    def block_table(self, seq: str) -> list[int]:
        return list(self._get(seq))

    def num_free(self) -> int:
        return len(self._free)

    # extra introspection (not part of the required interface)
    def num_tokens(self, seq: str) -> int:
        self._get(seq)
        return self._tokens[seq]
