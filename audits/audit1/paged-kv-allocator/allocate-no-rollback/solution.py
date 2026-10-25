"""A paged KV-cache block allocator with reference-counted blocks, forking and copy-on-write."""

import heapq


class OutOfBlocks(Exception):
    pass


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))
        heapq.heapify(self._free)
        self._ref = [0] * num_blocks
        self._tables: dict[str, list[int]] = {}
        self._tokens: dict[str, int] = {}

    def _take(self) -> int:
        b = heapq.heappop(self._free)
        self._ref[b] = 1
        return b

    def _release(self, b: int) -> None:
        self._ref[b] -= 1
        if self._ref[b] == 0:
            heapq.heappush(self._free, b)

    def allocate(self, seq: str, num_tokens: int) -> None:
        if seq in self._tables:
            raise ValueError(f"sequence {seq!r} already exists")
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        need = max(1, -(-num_tokens // self.block_size))
        table = []
        for _ in range(need):
            if not self._free:
                raise OutOfBlocks(f"need {need} blocks, got {len(table)}")
            table.append(self._take())
        self._tables[seq] = table
        self._tokens[seq] = num_tokens

    def append(self, seq: str) -> list[tuple[int, int]]:
        table = self._tables[seq]
        n = self._tokens[seq]
        idx = n // self.block_size
        if idx == len(table):
            if not self._free:
                raise OutOfBlocks("no free block for a new token block")
            table.append(self._take())
            self._tokens[seq] = n + 1
            return []
        last = table[idx]
        if self._ref[last] > 1:
            if not self._free:
                raise OutOfBlocks("no free block for copy-on-write")
            new = self._take()
            self._ref[last] -= 1
            table[idx] = new
            self._tokens[seq] = n + 1
            return [(last, new)]
        self._tokens[seq] = n + 1
        return []

    def fork(self, parent: str, child: str) -> None:
        if child in self._tables:
            raise ValueError(f"sequence {child!r} already exists")
        table = self._tables[parent]
        for b in table:
            self._ref[b] += 1
        self._tables[child] = list(table)
        self._tokens[child] = self._tokens[parent]

    def free(self, seq: str) -> None:
        table = self._tables.pop(seq)
        del self._tokens[seq]
        for b in table:
            self._release(b)

    def block_table(self, seq: str) -> list[int]:
        return list(self._tables[seq])

    def num_free(self) -> int:
        return len(self._free)
