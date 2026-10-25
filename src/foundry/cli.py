"""foundry: prove tasks (forge), grade submissions, run attempt batches, serve the console."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)


@app.command()
def forge(only: str = typer.Option("", help="comma-separated task ids; default all")) -> None:
    """Prove every task: reference passes, every mutant is rejected, APIs current, rubric lint."""
    from . import forge as fg

    doc = fg.forge_all(only=[x for x in only.split(",") if x] or None)
    path = fg.write(doc)
    s = doc["summary"]
    for t in doc["tasks"]:
        typer.echo(
            f"{'ok ' if t['ok'] else 'BAD'} {t['id']:32s} mutants {t['killed']}/{len(t['mutants'])}  {'; '.join(t['problems'])[:150]}"
        )
    typer.echo(f"{s['ok']}/{s['tasks']} tasks ok, {s['killed']}/{s['mutants']} mutants killed -> {path}")
    if s["ok"] != s["tasks"]:
        raise typer.Exit(1)


@app.command("grade")
def grade_cmd(task_id: str, submission: Path) -> None:
    """Grade one submission folder (solution.py and/or answer.yaml) against a task."""
    from .runner import grade
    from .task import get

    g = grade(get(task_id), submission)
    typer.echo(json.dumps(g.to_dict(), indent=1))
    if not g.passed:
        raise typer.Exit(1)


@app.command("list")
def list_cmd() -> None:
    """The tasks, by area."""
    from .task import discover

    for t in discover():
        typer.echo(f"{t.area:10s} {t.id:32s} d{t.difficulty} {t.kind:12s} {t.title}")


@app.command()
def attempt(label: str, force: bool = typer.Option(False, help="rewrite an existing report (dev runs only)")) -> None:
    """Grade attempts/<label>/<task>/ submissions; the first run is recorded once."""
    from . import attempts

    doc = attempts.run_attempts(label, force=force)
    s = doc["summary"]
    typer.echo(f"{label}: passed {s['passed']}/{s['attempted']} ({s['not_attempted']} not attempted) {json.dumps(s['by_area'])}")


@app.command()
def audit(label: str, force: bool = typer.Option(False, help="rewrite an existing report (dev runs only)")) -> None:
    """Run audits/<label>/<task>/<variant>/ (deliberately wrong submissions); report escapes."""
    from . import attempts

    doc = attempts.run_audit(label, force=force)
    typer.echo(f"{label}: {doc['summary']['escaped']} of {doc['summary']['variants']} wrong submissions escaped")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """The review console and its API."""
    import uvicorn

    uvicorn.run("foundry.api:create_app", host=host, port=port, factory=True)


if __name__ == "__main__":  # pragma: no cover
    app()
