# Long-horizon loop: epistasis-ceiling

Model for every worker and reviewer turn: `opencode-go/deepseek-v4.1-flash`.
Started 2026-11-16. Owner: Main agent.

## What this project is

How far do single-mutant measurements get you on a double mutant? Measured on
818,362 double mutants from 65 ProteinGym assays. The foundation is committed and
green; what remains is depth, verification and a second look at every claim.

## Why a loop, and what went wrong last time

The previous project's loop timed out on every fire because `loop_step.py` had no
terminal state and re-ran the whole verification sweep each time. Worse, a worker
published a number taken from a reviewer without measuring it, and a 7-commit
identity mistake sat in public history.

So this loop is built differently:

1. **One writer per artefact.** A step either owns a file or does not touch it.
   No two workers edit `results/` concurrently — that is how overlapping chunk
   ranges once double-counted 2,400 conformers.
2. **Ground truth over the ledger.** `scripts/loop_step.py` reads disk, never
   trust in `LOOP_STATE.json`.
3. **A terminal state.** When every gate passes, the loop reports DONE. The
   script cannot disable its own scheduler entry, so DONE means: the runner
   disables the automation. A DONE fire does no further work.
3b. **A fire that cannot finish its step must leave a resumable tree.** FIX /
   REDTEAM can exceed one fire's wall clock. Commit partial progress before the
   fire ends; the next fire re-reads disk and resumes from what exists. An
   uncommitted tree plus a timeout is a lost fire, and that is what cost two
   consecutive runs on 2026-11-16.
4. **Claim verification is a step, not a habit.** `scripts/audit_claims.py` checks
   every number in the README against a committed CSV. Nothing ships on prose.
5. **A reviewer that must find something.** Each review round names at least one
   weakness or declares, per claim, why the search came up empty.

## State machine (one step per fire)

| step | gate |
|---|---|
| 1. AUDIT | `scripts/audit_claims.py` exits 0: every README number matches a CSV |
| 2. DEPTH | residual structure probed beyond the magnitude prior; result committed |
| 3. REDTEAM | a reviewer subagent files findings; each gets a disposition |
| 4. FIX | findings actioned, tests green |
| 5. DOCS | README limits reflect what was actually found; figures embedded once |
| 6. SHIP | fresh clone: tests pass, all figure links resolve, CI green |
| 7. DONE | terminal |

## Rails

- Every number on the page comes from a committed artefact; `audit_claims.py`
  enforces it mechanically.
- A test must fail before a fix.
- Negative results are results. The ESM probe failed and is published as a
  failure, with its interval.
- Never quote a reviewer-supplied number without measuring it first.
- Long jobs run under `systemd-run --user`, never `setsid nohup`.
- Verify with the count, not the prose: `grep -c`, `pytest -q`, `git status`.
