"""A paged KV-cache block allocator with forking and copy-on-write.

Each sequence has a block table (physical block ids in order). A block is shared by reference count:
fork() copies the table and bumps the counts; append() copies the last block first if the writer would
write into a block that another sequence still uses. Every call that needs free blocks checks the
count before it mutates anything, so a failed call leaves the pool unchanged.
"""

from collections import deque


class OutOfBlocks(Exception):
    pass


def _ceil_div(a, b):
    return -(-a // b)


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = deque(range(num_blocks))
        self._refs = [0] * num_blocks
        self._tables = {}   # seq -> list of block ids
        self._tokens = {}   # seq -> number of tokens

    # ---- helpers -------------------------------------------------------------------------------
    def _take(self) -> int:
        block = self._free.popleft()
        self._refs[block] = 1
        return block

    def _release(self, block: int) -> None:
        self._refs[block] -= 1
        if self._refs[block] == 0:
            self._free.append(block)

    # ---- API -----------------------------------------------------------------------------------
    def allocate(self, seq: str, num_tokens: int) -> None:
        if seq in self._tables:
            raise ValueError(f"sequence {seq!r} already exists")
        need = max(1, _ceil_div(num_tokens, self.block_size))
        if need > len(self._free):
            raise OutOfBlocks(f"need {need} blocks, {len(self._free)} free")
        self._tables[seq] = [self._take() for _ in range(need)]
        self._tokens[seq] = num_tokens

    def append(self, seq: str) -> list:
        """Make room for one more token. Returns (source, destination) copies the caller must make."""
        table = self._tables[seq]
        n = self._tokens[seq]
        if n > 0 and n % self.block_size == 0:
            # the new token starts a new block
            if not self._free:
                raise OutOfBlocks("no free block for a new block")
            table.append(self._take())
            self._tokens[seq] = n + 1
            return []
        last = table[-1]
        if self._refs[last] > 1:
            # the last block is shared: the writer gets a private copy first
            if not self._free:
                raise OutOfBlocks("no free block for copy-on-write")
            dst = self._take()
            self._refs[last] -= 1
            table[-1] = dst
            self._tokens[seq] = n + 1
            return [(last, dst)]
        self._tokens[seq] = n + 1
        return []

    def fork(self, parent: str, child: str) -> None:
        if child in self._tables:
            raise ValueError(f"sequence {child!r} already exists")
        table = list(self._tables[parent])
        for block in table:
            self._refs[block] += 1
        self._tables[child] = table
        self._tokens[child] = self._tokens[parent]

    def free(self, seq: str) -> None:
        if seq not in self._tables:
            return
        for block in self._tables.pop(seq):
            self._release(block)
        del self._tokens[seq]

    def block_table(self, seq: str) -> list:
        return list(self._tables[seq])

    def num_free(self) -> int:
        return len(self._free)
