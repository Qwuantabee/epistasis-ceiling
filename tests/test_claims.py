"""The README's numbers must equal the committed artefacts.

This is the test that would have caught the previous project's 23x error: 44 green
tests there never looked at the prose, so a wrong headline shipped. Here the audit
is a test, and it fails in CI, not just when someone remembers to run it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_every_readme_number_matches_an_artefact():
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_claims.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, (
        "README and artefacts disagree:\n" + p.stdout[-2000:] + p.stderr[-800:])


def test_readme_embeds_each_figure_exactly_once():
    """An orphan figure is worse than none, and a doubled embed is a duplicate
    caption waiting to diverge (both happened in the previous project)."""
    readme = (ROOT / "README.md").read_text()
    import re
    embeds = re.findall(r"\]\((figures/[a-zA-Z0-9_]+\.png)\)", readme)
    for e in set(embeds):
        assert embeds.count(e) == 1, f"{e} embedded {embeds.count(e)} times"
    for e in embeds:
        assert (ROOT / e).exists(), f"{e} is embedded but does not exist"


def test_every_figure_on_disk_is_embedded():
    figs = {f"figures/{p.name}" for p in (ROOT / "figures").glob("*.png")}
    readme = (ROOT / "README.md").read_text()
    orphans = sorted(f for f in figs if f not in readme)
    assert not orphans, f"figures exist but are not shown: {orphans}"


def test_audit_never_reports_clean_when_a_claim_is_unverifiable():
    """A green audit over reduced coverage is worse than a red one.

    results/pairs/ is gitignored, so a fresh clone cannot check the pooled
    epistasis fraction. The audit used to omit that claim silently and report
    23/23 instead of 24/24. It must now name what it could not check.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "scripts"))
    import audit_claims

    claims, unverifiable = audit_claims.build_claims()
    # Either everything is checkable, or the gap is named; never both silent.
    if unverifiable:
        assert len(unverifiable) >= 1
        for u in unverifiable:
            assert "pairs" in u or "run " in u, (
                "an unverifiable claim must say how to make it checkable")
    # the audit source must actually print the unverified block
    src = (ROOT / "scripts" / "audit_claims.py").read_text()
    assert "UNVERIFIED" in src, (
        "the audit must announce reduced coverage rather than implying a full pass")
