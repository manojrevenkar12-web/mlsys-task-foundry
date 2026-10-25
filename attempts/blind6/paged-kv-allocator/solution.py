"""Paged KV-cache block pool with fork and copy-on-write."""

import heapq


class OutOfBlocks(Exception):
    pass


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 0 or block_size <= 0:
            raise ValueError("bad pool shape")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks))  # a sorted list is a valid min-heap
        heapq.heapify(self._free)
        self._ref = [0] * num_blocks
        self._table = {}   # seq -> list of block ids
        self._len = {}     # seq -> number of tokens

    # -- helpers -------------------------------------------------------
    def _take(self) -> int:
        b = heapq.heappop(self._free)
        self._ref[b] = 1
        return b

    def _release(self, b: int) -> None:
        self._ref[b] -= 1
        if self._ref[b] == 0:
            heapq.heappush(self._free, b)

    def _blocks_for(self, n: int) -> int:
        return max(1, -(-n // self.block_size))

    # -- API -----------------------------------------------------------
    def allocate(self, seq: str, num_tokens: int) -> None:
        if seq in self._table:
            raise ValueError(f"sequence {seq!r} already exists")
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        need = self._blocks_for(num_tokens)
        if need > len(self._free):
            raise OutOfBlocks(f"need {need} blocks, {len(self._free)} free")
        self._table[seq] = [self._take() for _ in range(need)]
        self._len[seq] = num_tokens

    def append(self, seq: str) -> list:
        table = self._table[seq]
        n = self._len[seq]
        copies = []
        if n > 0 and n % self.block_size == 0:
            # the new token starts a new block
            if not self._free:
                raise OutOfBlocks("no free block for a new block")
            table.append(self._take())
        else:
            last = table[-1]
            if self._ref[last] > 1:
                if not self._free:
                    raise OutOfBlocks("no free block for copy-on-write")
                dst = self._take()
                self._ref[last] -= 1  # still used by someone else, never reaches 0 here
                table[-1] = dst
                copies.append((last, dst))
        self._len[seq] = n + 1
        return copies

    def fork(self, parent: str, child: str) -> None:
        if parent not in self._table:
            raise KeyError(parent)
        if child in self._table:
            raise ValueError(f"sequence {child!r} already exists")
        tbl = list(self._table[parent])
        for b in tbl:
            self._ref[b] += 1
        self._table[child] = tbl
        self._len[child] = self._len[parent]

    def free(self, seq: str) -> None:
        tbl = self._table.pop(seq)  # KeyError for unknown sequence
        del self._len[seq]
        for b in tbl:
            self._release(b)

    def block_table(self, seq: str) -> list:
        return list(self._table[seq])

    def num_free(self) -> int:
        return len(self._free)
