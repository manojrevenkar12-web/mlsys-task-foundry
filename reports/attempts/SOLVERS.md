# Who solved what

Every blind set was written by a model agent that saw only the solver's files (prompt, starter,
assets) in a separate folder, never the graders, references, mutants or other sets. Each set's
first run is in `<label>.json` / `<label>.md` and is never rewritten.

| Set | Tasks | Solver | Could run code |
|---|---|---|---|
| blind1 | 22 core | the author's model (the session's default Claude agent, the model that wrote the tasks) | yes |
| blind2 | 22 core | the author's model | no: written by reasoning alone; Python only to read assets |
| blind3 | 5 hard | the author's model | yes |
| blind4 | 5 hard | the author's model | no |
| blind5 | 27 | Claude Haiku (a smaller model) | yes |
| blind6 | 27 | Claude Sonnet (a mid-size model) | yes |

The author's model solving tasks it wrote is the weakest test of difficulty: the tasks encode what
that model already knows. The smaller models give a difficulty curve; a fair difficulty estimate for
frontier models needs solvers from other model families and human experts.
