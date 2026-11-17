#!/usr/bin/env python3
"""The epistasis-ceiling loop's single decided-on-each-fire step.

Ground truth comes from disk, never from the ledger. When every gate passes this
returns step 99 DONE and the automation disables itself - the previous project's
loop failed every fire because it had no terminal state.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "LOOP_STATE.json"
PY = sys.executable


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"step": 1}


def save(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def audit_ok() -> bool:
    rc, _ = run([PY, "scripts/audit_claims.py"])
    return rc == 0


def tests_ok() -> bool:
    import os
    env = {**os.environ, "PYTHONPATH": "src"}
    p = subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    return p.returncode == 0


def detect() -> tuple[int, str]:
    """Which step is genuinely next, judged from what exists on disk."""
    if not (ROOT / "results" / "assay_summary.csv").exists():
        return 1, "BUILD"
    if not audit_ok():
        return 2, "AUDIT"
    if not (ROOT / "results" / "depth.json").exists():
        return 3, "DEPTH"
    if not (ROOT / "results" / "redteam.json").exists():
        return 4, "REDTEAM"
    rt = json.loads((ROOT / "results" / "redteam.json").read_text())
    if rt.get("findings_unactioned", 1) > 0:
        return 5, "FIX"
    if not tests_ok():
        return 5, "FIX"
    if not (ROOT / "results" / "ship_check.json").exists():
        return 6, "SHIP"
    ship = json.loads((ROOT / "results" / "ship_check.json").read_text())
    if not ship.get("ok"):
        return 6, "SHIP"
    return 99, "DONE"


NEXT = {
    "BUILD": f"{PY} scripts/build_pairs.py --data <ProteinGym dir> --index <DMS_substitutions.csv>",
    "AUDIT": f"{PY} scripts/audit_claims.py   # fix README or artefact so they agree",
    "DEPTH": "probe residual structure beyond the magnitude prior; commit results/depth.json",
    "REDTEAM": "spawn a reviewer subagent on deepseek-v4.1-flash; it must file findings",
    "FIX": "action every finding in results/redteam.json, then re-run tests",
    "SHIP": "fresh clone: pytest, figure links, CI; write results/ship_check.json",
    "DONE": "none - project shipped; disable this automation",
}


def main() -> int:
    st = load()
    step, name = detect()
    st.update(step=step, step_name=name, checked_at=now())
    out = {"step": step, "step_name": name, "next": NEXT[name]}
    if name == "DONE":
        out["shipped"] = True
        save(st)
        print(json.dumps(out, indent=2))
        return 0
    save(st)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
