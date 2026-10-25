"""Write a mutant from the reference by exact text substitutions (used while authoring tasks).

    python scripts/mutate.py TASK_DIR NAME "[mode, ...]" "what is wrong" OLD NEW [OLD NEW ...]

Each OLD must occur exactly once across the reference's files, so a mutant cannot silently
drift from the code it claims to change."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> None:
    task, name, modes, note, *pairs = sys.argv[1:]
    if not pairs or len(pairs) % 2:
        raise SystemExit("give OLD NEW pairs")
    files = {f.name: f.read_text() for f in (Path(task) / "reference").iterdir() if f.is_file()}
    for old, new in zip(pairs[::2], pairs[1::2], strict=True):
        hits = [n for n, t in files.items() if old in t]
        count = sum(t.count(old) for t in files.values())
        if count != 1:
            raise SystemExit(f"{name}: {old[:60]!r} occurs {count} times in the reference, expected once")
        files[hits[0]] = files[hits[0]].replace(old, new)
    out = Path(task) / "mutants" / name
    out.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (out / n).write_text(t)
    (out / "mutant.yaml").write_text(yaml.safe_dump({"modes": yaml.safe_load(modes), "note": note}, sort_keys=False, width=120))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
