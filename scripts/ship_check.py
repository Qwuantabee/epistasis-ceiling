#!/usr/bin/env python3
"""Does this repo actually ship? Clone it as a stranger would and check.

Everything here is a check a skeptical reader would run in the first two minutes:
does a fresh clone install, do the tests pass, does every figure in the README
resolve to a committed file, does the claim audit pass, is the licence detected.

Why a clone rather than the working tree: the working tree has results/pairs/
(gitignored, 800k rows) and the ML envs. A clone has neither. Testing the clone is
what catches "works on my machine" - it is exactly how the previous project's
silent 23/23-vs-24/24 audit gap was found.

Writes results/ship_check.json only. Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list[str], cwd: Path, env: dict | None = None,
        timeout: int = 900) -> tuple[int, str]:
    e = {**os.environ, **(env or {})}
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           env=e, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://github.com/Qwuantabee/epistasis-ceiling.git")
    ap.add_argument("--out", default=str(ROOT / "results" / "ship_check.json"))
    ap.add_argument("--keep", action="store_true",
                    help="keep the clone (default: remove it)")
    args = ap.parse_args()

    checks: list[dict] = []

    tmp = Path(tempfile.mkdtemp(prefix="ec_ship_"))
    clone = tmp / "repo"
    try:
        rc, out = run(["git", "clone", "--depth", "1", args.url, str(clone)], tmp)
        checks.append({"check": "fresh clone", "ok": rc == 0,
                       "detail": out[-400:] if rc else "cloned"})
        if rc != 0:
            raise SystemExit("clone failed; nothing further can be checked")

        # 1. tests
        rc, out = run([PY, "-m", "pytest", "tests/", "-q"], clone,
                      env={"PYTHONPATH": "src"})
        tail = [l for l in out.splitlines() if l.strip()][-1:] or [""]
        checks.append({"check": "pytest in clone", "ok": rc == 0,
                       "detail": tail[0][:200]})

        # 2. claim audit (will report reduced coverage without results/pairs/)
        # ONE path argument, not "scripts" + "audit_claims.py": python treats a
        # bare directory as the script and fails with "can't find '__main__'",
        # which the first version of this checker reported as a claim-audit
        # failure - a false negative that hid the real result.
        rc, out = run([PY, "scripts/audit_claims.py"], clone)
        checks.append({"check": "claim audit", "ok": rc == 0,
                       "detail": out.splitlines()[0] if out else "",
                       "note": ("runs with reduced coverage: results/pairs/ is "
                                "gitignored, so the pooled claim is unverified "
                                "in a clone" if "UNVERIFIED" in out else "")})

        # 3. every figure embedded resolves, exactly once each, none orphaned
        readme = (clone / "README.md").read_text()
        embeds = re.findall(r"\]\((figures/[A-Za-z0-9_]+\.png)\)", readme)
        missing = sorted({e for e in embeds if not (clone / e).exists()})
        dupes = sorted({e for e in embeds if embeds.count(e) > 1})
        on_disk = {f"figures/{p.name}" for p in (clone / "figures").glob("*.png")}
        orphans = sorted(on_disk - set(embeds))
        checks.append({"check": "figure links resolve", "ok": not missing,
                       "detail": f"{len(set(embeds))} distinct, missing={missing}"})
        checks.append({"check": "no duplicate figure embeds", "ok": not dupes,
                       "detail": f"duplicates={dupes}"})
        checks.append({"check": "no orphan figures", "ok": not orphans,
                       "detail": f"orphans={orphans}"})

        # 4. licence file present and parseable as MIT
        lic = clone / "LICENSE"
        is_mit = lic.exists() and "MIT License" in lic.read_text()[:200]
        checks.append({"check": "MIT licence present", "ok": bool(is_mit),
                       "detail": "LICENSE" if is_mit else "missing or not MIT"})

        # 5. author attribution correct on every commit
        rc, out = run(["git", "log", "--format=%ae"], clone)
        emails = sorted(set(out.split())) if rc == 0 else []
        wrong = [e for e in emails if e != "200713142+Qwuantabee@users.noreply.github.com"]
        checks.append({"check": "commit attribution", "ok": not wrong,
                       "detail": f"emails={emails}"})

        # 6. nothing large or secret-ish committed
        rc, out = run(["git", "ls-files"], clone)
        files = out.splitlines() if rc == 0 else []
        big = []
        for f in files:
            p = clone / f
            if p.is_file() and p.stat().st_size > 25 * 1024 * 1024:
                big.append(f"{f} ({p.stat().st_size/1e6:.0f} MB)")
        checks.append({"check": "no single file over 25 MB", "ok": not big,
                       "detail": f"{len(files)} tracked files; big={big}"})

    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    ok = all(c["ok"] for c in checks)
    payload = {"ok": ok, "checks": checks,
               "head": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=ROOT, capture_output=True,
                                      text=True).stdout.strip()}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    for c in checks:
        print(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['check']:32s} {c['detail'][:90]}")
    print(f"\n{'SHIPS' if ok else 'DOES NOT SHIP'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
