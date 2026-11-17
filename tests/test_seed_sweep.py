"""The seed-aggregated verdict (DESIGN.md) re-derives from committed inputs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import seed_sweep  # noqa: E402

ART = json.loads((ROOT / "results" / "seed_sweep.json").read_text())


def test_seed_list_is_the_committed_0_to_199():
    assert seed_sweep.load_seeds(ROOT / "results" / "seed_sweep_seeds.txt") \
        == list(range(200))
    assert [r["seed"] for r in ART["per_seed"]] == list(range(200))


def test_every_seed_row_rederives():
    _, other, corpus = seed_sweep.groups(ROOT / "results" / "assay_summary.csv")
    for row in ART["per_seed"]:
        assert seed_sweep.per_seed(other, corpus, row["seed"]) == row


def test_the_single_seed_in_heldout_json_is_one_of_these_draws():
    _, other, corpus = seed_sweep.groups(ROOT / "results" / "assay_summary.csv")
    h = json.loads((ROOT / "results" / "heldout.json").read_text())
    stored = h["H1_lab_family"]["other_labs"]
    r = seed_sweep.per_seed(other, corpus, h["bootstrap"]["seed"])
    assert [r["lo"], r["hi"]] == stored["auc"]["ci95"]
    assert r["verdict"] == stored["verdict_vs_corpus"]["auc"]


def test_aggregate_applies_the_rule():
    assert seed_sweep.aggregate(ART["per_seed"]) == ART["aggregate"]
    counts = ART["aggregate"]["verdict_counts"]
    stable = [v for v, n in counts.items() if n >= 190]
    want = stable[0] if stable else "INCONCLUSIVE (seed-dependent)"
    assert ART["aggregate"]["verdict"] == want


def test_rule_branches():
    row = {"lo": 0.9, "hi": 0.95, "clearance": 0.01}
    ok = [dict(row, seed=i, verdict="SUPPORTED") for i in range(190)]
    bad = [dict(row, seed=i, verdict="NOT_SUPPORTED") for i in range(190, 200)]
    assert seed_sweep.aggregate(ok + bad)["verdict"] == "SUPPORTED"
    bad11 = [dict(row, seed=i, verdict="NOT_SUPPORTED") for i in range(189, 200)]
    assert seed_sweep.aggregate(ok[:189] + bad11)["verdict"] \
        == "INCONCLUSIVE (seed-dependent)"
