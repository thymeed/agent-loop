# csvrow experiment: design only (pre-registered 2026-10-30, nothing run)

Status: written before any attempt. Do not edit the rule after seeing results; add a dated note instead.
Do not change code or run Docker while an eval runs: each attempt is a fresh process that loads the
code on disk, and extra load distorts timings.

## Question
`csvrow` has never been solved (0 of 18 attempts across the baseline and the three thinking variants).
The diagnosis found: the worker mishandles doubled quotes and the quoted-empty field; 12 of 18
attempts were lost to an unusable reply or a bad file path; the lead's plan can mislead the worker;
the prompt's file-block example is `pkg/module.py`, which is not the project's layout (`csvrow/parse.py`).
Is the limit the prompt, or the models?

## Variants (all on `csvrow` only, same models and `config.yaml` unless stated)
| Id | Change | How it is built |
|----|--------|-----------------|
| control | none | `experiments/control.yaml` |
| path | the file-block example in the worker prompt uses a real path from the project | code change in `agent_loop/nodes.py` (line ~122), behind a new whitelisted option so it is a patch, not a default |
| no-plan | the worker gets the task and tests with no lead plan (the lead call is skipped) | code change in `nodes.py`, same option mechanism |
| qwen3-worker | worker is `qwen3:8b` with `reasoning: false` | config patch only, no code |

`path` and `no-plan` need code and a test showing the option does something, so they are a separate
unit (a top-level `prompt_variants` list in the patch, whitelisted like `model_options`).

Change note, 2026-10-30, before any run: the original fourth row was `plan-only` (plan without code).
Reading `nodes.py` showed the lead's prompt already says "No code, no preamble", so that variant would
have measured nothing new. The owner replaced it with `no-plan`, which tests the diagnosis finding
that the plan can mislead the worker. Nothing else changed.

Change note, 2026-10-30, after control attempt #10: the csvrow tests were extended from 6 to 11.
Attempt #10 passed with a doubled-quote bug (`"a"",b"` came back split). The earlier 0 of 18 attempts
used the 6-test version; a stricter test file cannot turn those failures into passes, so 0/18 stays
valid. The control is to be re-scored against the new tests, with no new model runs.

## Sample
10 attempts per variant, `--only csvrow --attempts 10`: 40 attempts, about 33 minutes per variant at
the roughly 200 s a csvrow attempt took in the baseline. Run one variant at a time.
Pooled history is 0/18, so at a true rate near 5% the chance of 3 or more passes in 10 is about 1%:
three passes is not noise. One flipped attempt moves a variant by 10 points.

## What counts
- Pass = the run ends with the task's own tests green in the sandbox (the existing eval verdict).
- Infrastructure errors count as failures. If a variant has more than 2 errors out of 10, rerun that
  variant once from scratch and use the rerun.
- d = variant passes minus control passes, out of 10.

## Decision table (one row applies; rows cover every integer d)
| d | Outcome |
|---|---------|
| d <= 2 | DROP the variant: not shown to help. |
| 3 <= d <= 5 | REPEAT: run 10 more attempts of the variant and of control. Over the combined 20 each, ADOPT-CANDIDATE if d >= 6, otherwise DROP. |
| d >= 6 | ADOPT-CANDIDATE. |

ADOPT-CANDIDATE is not adoption. A candidate must also run the other five eval tasks at
`--attempts 3` and lose no more than one pass in total against the baseline before it becomes default.

## Secondary measure (reported, never decides)
Per variant, the share of attempts whose first worker reply applied at least one file block. Read from
the traces; if the traces do not expose it cleanly, say so and report "not measured".

## Report
One line per variant: passes/10, d, mean seconds, errors, outcome from the table. Nulls are reported
the same as wins. Reports go to `runs/evals/` with labels `csvrow-<id>`.

## Results (added 2026-10-30; the rule above is unchanged)
All four ran on the 11-test fixture. The control was re-scored from its saved code with no new model
runs (its original report, written under 6 tests, says 2/10; attempt #10 had a doubled-quote bug).

| Variant | Passes /10 | d | Mean s | Errors | Outcome |
|---|---|---|---|---|---|
| control (re-scored) | 1 | - | 29 | 0 | - |
| path | 1 | 0 | 32 | 0 | DROP |
| no-plan | 0 | -1 | 27 | 0 | DROP |
| qwen3-worker | 0 | -1 | 53 | 0 | DROP |

Reports: `runs/evals/csvrow-control-20261030-215034-e2f3.json`, `csvrow-path-20261030-230447-375b.json`,
`csvrow-noplan-20261030-231214-ce6e.json`, `csvrow-qwen3-worker-20261030-231652-60ac.json`.
Every failure ended at the rejection cap (recorded as `capped`).

Secondary measure (first worker reply applied at least one file block, wrong-path files included):
control 9/10, path 10/10, no-plan 10/10, qwen3-worker 10/10. The stray `pkg/csvrow/parse.py` write that
the prompt's example path invites happened in 3/10 control attempts and 0/10 path attempts.

Conclusion: no variant is shown to help, so none is a candidate and nothing changes in the defaults.
Moving the example path removed the stray-file failure but not the low pass rate, and dropping the plan
or switching the worker to qwen3:8b did not help either. With 10 attempts per variant a difference of one
attempt is noise (the table's own note: one flip moves a variant 10 points). The likely limit is the model
size (4B lead, 7B coder), but this experiment did not test that directly. Not pinned on purpose: csvrow
quoting cases the task text does not decide (a quote inside an unwrapped field, text after a closing quote).
