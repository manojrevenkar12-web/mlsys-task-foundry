from math import ceil


class OutOfBlocks(Exception):
    pass


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_list = list(range(num_blocks - 1, -1, -1))  # pop() hands out low ids first
        self.refs = [0] * num_blocks
        self.tables: dict[str, list[int]] = {}
        self.lengths: dict[str, int] = {}

    def _take(self) -> int:
        b = self.free_list.pop()
        self.refs[b] = 1
        return b

    def _release(self, b: int) -> None:
        self.refs[b] = 0
        if b not in self.free_list:
            self.free_list.append(b)

    def allocate(self, seq: str, num_tokens: int) -> None:
        if seq in self.tables:
            raise ValueError(f"{seq} exists")
        need = max(1, ceil(num_tokens / self.block_size))
        if need > len(self.free_list):
            raise OutOfBlocks(f"{seq} needs {need} blocks, {len(self.free_list)} free")
        self.tables[seq] = [self._take() for _ in range(need)]
        self.lengths[seq] = num_tokens

    def append(self, seq: str) -> list[tuple[int, int]]:
        table, n = self.tables[seq], self.lengths[seq]
        if n % self.block_size == 0 and n // self.block_size == len(table):
            if not self.free_list:
                raise OutOfBlocks(f"{seq} needs a new block")
            table.append(self._take())
            self.lengths[seq] = n + 1
            return []
        last = table[-1]
        if self.refs[last] > 1:  # shared: copy before writing
            if not self.free_list:
                raise OutOfBlocks(f"{seq} needs a block to copy its shared last block")
            new = self._take()
            self.refs[last] -= 1
            table[-1] = new
            self.lengths[seq] = n + 1
            return [(last, new)]
        self.lengths[seq] = n + 1
        return []

    def fork(self, parent: str, child: str) -> None:
        if child in self.tables:
            raise ValueError(f"{child} exists")
        self.tables[child] = list(self.tables[parent])
        self.lengths[child] = self.lengths[parent]
        for b in self.tables[child]:
            self.refs[b] += 1

    def free(self, seq: str) -> None:
        for b in self.tables.pop(seq):
            self._release(b)
        del self.lengths[seq]

    def block_table(self, seq: str) -> list[int]:
        return list(self.tables[seq])

    def num_free(self) -> int:
        return len(self.free_list)
