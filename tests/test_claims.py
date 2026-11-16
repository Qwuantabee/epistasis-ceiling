"""The README's numbers must equal the committed artefacts.

This is the test that would have caught the previous project's 23x error: 44 green
tests there never looked at the prose, so a wrong headline shipped. Here the audit
is a test, and it fails in CI, not just when someone remembers to run it.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_every_readme_number_matches_an_artefact():
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_claims.py"),
                        "--strict"], capture_output=True, text=True, cwd=ROOT)
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


def _audit_module():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import audit_claims
    return audit_claims


def test_audit_never_reports_clean_when_a_claim_is_unverifiable(tmp_path):
    """A green audit over reduced coverage is worse than a red one.

    TEST-CHANGE (review round 2, code R2-02): the old body only asserted that the
    word 'UNVERIFIED' appeared in the audit's SOURCE; the list it guarded was never
    appended to, so deleting any artefact still gave exit 0 and 'every published
    number matches'. This version deletes each required artefact in turn from a
    copy of results/ and requires the audit to FAIL and to name it, and does the
    same for a key removed from inside an artefact.
    """
    import shutil

    ac = _audit_module()
    src = tmp_path / "full"
    src.mkdir()
    for name in ac.REQUIRED_ARTEFACTS:
        shutil.copy(ROOT / "results" / name, src / name)
    claims, missing = ac.build_claims(src)
    assert not missing, f"the complete copy should check clean: {missing}"
    n_full = len(claims)

    for name in ac.REQUIRED_ARTEFACTS:
        d = tmp_path / name.replace(".", "_")
        shutil.copytree(src, d)
        (d / name).unlink()
        _, missing = ac.build_claims(d)
        assert name in missing, f"deleting {name} was not reported as missing"

    # the CLI path CI actually runs: exit 1, artefact named, no clean message
    d = tmp_path / "heldout_json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_claims.py"),
                        "--results", str(d)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 1, p.stdout[-1500:]
    assert "MISSING ARTEFACT" in p.stdout and "heldout.json" in p.stdout
    assert "every published number matches" not in p.stdout

    # a key that vanishes inside an artefact is a failure too, not a skipped claim
    d = tmp_path / "key_gone"
    shutil.copytree(src, d)
    pu = json.loads((d / "prior_units.json").read_text())
    del pu["pooled_across_assays_sd_scaled"]
    (d / "prior_units.json").write_text(json.dumps(pu))
    claims, missing = ac.build_claims(d)
    assert any("pooled_across_assays_sd_scaled" in m for m in missing), missing
    assert len(claims) < n_full


# ---- mutation tests: the audit must REJECT a wrong README ------------------------
# Measured before this change (scripts/audit_claims.py at 64cf0cd, the substring
# search, against the README at 64cf0cd): of the mutations below, 11 also apply to
# that page and it caught 0 of the 11. Each one here edits the README IN MEMORY and
# must make the audit fail.

def _swap_cells(text, label_a, label_b, col):
    lines = text.split("\n")
    ia = next(i for i, l in enumerate(lines) if re.match(r"\|\s*" + label_a + r"\s*\|", l))
    ib = next(i for i, l in enumerate(lines) if re.match(r"\|\s*" + label_b + r"\s*\|", l))
    ca, cb = lines[ia].split("|"), lines[ib].split("|")
    assert ca[col + 1] != cb[col + 1], "swapping equal cells would not be a mutation"
    ca[col + 1], cb[col + 1] = cb[col + 1], ca[col + 1]
    lines[ia], lines[ib] = "|".join(ca), "|".join(cb)
    return "\n".join(lines)


def _sub(pattern, repl):
    def f(text):
        out, n = re.subn(pattern, repl, text, count=1)
        assert n == 1, f"mutation pattern no longer matches the README: {pattern}"
        return out
    return f


MUTATIONS = {
    # the three the brief names
    "headline AUC 0.884 -> 0.784": _sub(r"AUC 0\.884", "AUC 0.784"),
    "swap the R2 of the position and pair rows": lambda t: _swap_cells(
        t, r"\+ position", r"\+ position \+ position-pair rank 3", 1),
    "flip a sign: pooled sd-scaled rho +0.105 -> -0.105":
        _sub(r"\*\*\+0\.105\*\*", "**-0.105**"),
    # the rest of the families the reviewer used
    "table AUC 0.884 -> 0.784": _sub(
        r"(\| median AUC \(threshold-free direction\) \| \*\*)0\.884", r"\g<1>0.784"),
    "swap the Spearman and AUC table values": lambda t: _swap_cells(
        t, r"median assay Spearman \(additive vs observed\)",
        r"median AUC \(threshold-free direction\)", 1),
    "swap the bias cells of plain additivity and the own offset": lambda t: _swap_cells(
        t, r"plain additivity \(recomputed\)", r"\+ the assay's own offset", 4),
    "ESM head (a) now beats its control": _sub(
        r"Head \(a\) scores \+0\.026(\s+)against a shuffled control of \+0\.055",
        r"Head (a) scores +0.055\1against a shuffled control of +0.026"),
    "flip a sign: within-assay rho +0.313 -> -0.313":
        _sub(r"\*\*\+0\.313\*\* in raw units", "**-0.313** in raw units"),
    "flip a sign: wild-type bias +1.15 -> -1.15": _sub(r"used and \+1\.15", "used and -1.15"),
    "vacuous spelling: best R2 0.381 -> 0.999": _sub(
        r"(\| \+ position \+ position-pair rank 3 \| \*\*)0\.381", r"\g<1>0.999"),
    "pooled numerator 332,751 -> 332,157": _sub(r"332,751", "332,157"),
    "positive in 33/33 -> 3/33": _sub(r"positive in 33/33", "positive in 3/33"),
    "recalibration 0.500 -> 0.050": _sub(r"removes a median \*\*0\.500\*\*",
                                         "removes a median **0.050**"),
    "verdict word: leave-one-lab-out INCONCLUSIVE -> SUPPORTED": _sub(
        r"artefact's verdict is \*\*INCONCLUSIVE\*\*", "artefact's verdict is **SUPPORTED**"),
    "verdict word: residue INCONCLUSIVE -> SUPPORTED": _sub(
        r"Verdict \*\*INCONCLUSIVE\*\*", "Verdict **SUPPORTED**"),
    "deleted sentence: the t(4) critical value": _sub(
        r"with the t\(4\) critical value of \*\*2\.776\*\*", "with a larger critical value"),
}


@pytest.fixture(scope="module")
def audit_setup():
    ac = _audit_module()
    claims, missing = ac.build_claims()
    assert not missing, missing
    readme = (ROOT / "README.md").read_text()
    base = ac.audit(readme, claims, strict=True)
    assert base["ok"], "the unmutated README must pass, or the mutation test is vacuous: " \
        + "; ".join(base["problems"][:5] + base["estimator"][:3])
    return ac, claims, readme


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_audit_rejects_a_wrong_readme(audit_setup, name):
    ac, claims, readme = audit_setup
    mutated = MUTATIONS[name](readme)
    assert mutated != readme
    out = ac.audit(mutated, claims, strict=True)
    assert not out["ok"], f"the audit passed a README with: {name}"


def test_strict_mode_fails_on_an_unanchored_claim(audit_setup):
    """--strict is what CI runs: a claim checked only as 'appears somewhere' is
    exactly the hole review round 2 measured, so it must not pass silently."""
    ac, claims, readme = audit_setup
    loose = claims + [ac.Claim("an unanchored claim", ["0.884"], "test", ())]
    assert ac.audit(readme, loose, strict=False)["ok"]
    out = ac.audit(readme, loose, strict=True)
    assert not out["ok"] and out["unanchored"] == ["an unanchored claim"]


def test_esm_verdict_table_matches_the_artefact():
    """The ESM verdict table states VERDICTS, not rho values.

    scripts/audit_claims.py checks numbers; a verdict word is not a number, so
    nothing mechanical stopped the README's two-column table from drifting away
    from results/esm_scale.json. This closes that: every (head, sample size) pair
    the page claims has to be the verdict the artefact recorded, or the page is
    describing a run that did not happen.
    """
    import json
    import re

    scale_p = ROOT / "results" / "esm_scale.json"
    if not scale_p.exists():
        pytest.skip("no corpus-scale ESM artefact; run scripts/esm_scale.py")
    d = json.loads(scale_p.read_text())
    readme = (ROOT / "README.md").read_text()

    primary = ((d.get("primary") or {}).get("heads")) or {}
    full = (((d.get("sensitivity") or [{}])[0]).get("heads")) or {}
    for hid, label in (("a", "(a) the shipped PCA(100)+RidgeCV estimator"),
                       ("b", "(b) early-stopped linear on window embeddings, "
                             "no interaction terms"),
                       ("c", "(c) the free prior always present, ESM block on top")):
        want_partial = (primary.get(hid) or {}).get("verdict", {}).get("verdict")
        want_full = (full.get(hid) or {}).get("verdict", {}).get("verdict")
        line = next((l for l in readme.splitlines() if l.startswith(f"| {label}")), None)
        assert line is not None, f"the README no longer tabulates head {hid}"
        assert want_partial and want_partial in line, (
            f"head {hid}: README row does not state {want_partial!r}; the artefact "
            f"recorded that verdict for the capped run")
        assert want_full and want_full in line, (
            f"head {hid}: README row does not state the full-corpus verdict "
            f"{want_full!r}")
