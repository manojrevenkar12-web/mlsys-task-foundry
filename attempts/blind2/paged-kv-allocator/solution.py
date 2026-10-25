"""A paged KV-cache block allocator with forking and copy-on-write."""

import heapq


class OutOfBlocks(Exception):
    pass


def _blocks_for(num_tokens: int, block_size: int) -> int:
    return max(1, -(-num_tokens // block_size))


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 0 or block_size <= 0:
            raise ValueError("num_blocks must be >= 0 and block_size > 0")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))  # min-heap of free block ids
        heapq.heapify(self._free)
        self._ref = [0] * num_blocks
        self._tables: dict[str, list[int]] = {}
        self._tokens: dict[str, int] = {}

    # -- internals -------------------------------------------------------------------------------
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

    # -- API -------------------------------------------------------------------------------------
    def allocate(self, seq: str, num_tokens: int) -> None:
        if seq in self._tables:
            raise ValueError(f"sequence {seq!r} already exists")
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        need = _blocks_for(num_tokens, self.block_size)
        if need > len(self._free):
            raise OutOfBlocks(f"need {need} blocks, {len(self._free)} free")
        self._tables[seq] = [self._take() for _ in range(need)]
        self._tokens[seq] = num_tokens

    def append(self, seq: str) -> list[tuple[int, int]]:
        table = self._get(seq)
        n = self._tokens[seq]
        needed = _blocks_for(n + 1, self.block_size)
        if needed > len(table):
            # token n + 1 starts a new block
            if not self._free:
                raise OutOfBlocks("no free block for a new block")
            table.append(self._take())
            self._tokens[seq] = n + 1
            return []
        last = table[-1]
        if self._ref[last] > 1:
            # writing into a shared block: copy on write
            if not self._free:
                raise OutOfBlocks("no free block for copy-on-write")
            new = self._take()
            self._ref[last] -= 1  # still > 0: another sequence keeps it
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
