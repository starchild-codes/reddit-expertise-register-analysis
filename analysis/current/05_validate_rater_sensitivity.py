"""Validate the deterministic external-rater sensitivity release outputs."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "current" / "rater_sensitivity"


def read(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def match(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    return next(row for row in rows if all(row[key] == value for key, value in criteria.items()))


def main() -> None:
    assignments = read("tier_assignments_three_raters.csv")
    assert len(assignments) == 9
    assert all(row[key] in {"low", "medium", "high"} for row in assignments for key in row if key.endswith("tier"))
    for column, expected in (("author_original_tier", (3, 3, 3)), ("rater1_tier", (2, 4, 3)), ("rater2_tier", (1, 5, 3))):
        assert tuple(sum(row[column] == tier for row in assignments) for tier in ("low", "medium", "high")) == expected

    pairs = read("pairwise_agreement.csv")
    assert len(pairs) == 3
    external = match(pairs, comparison="rater1_vs_rater2")
    assert external["n_agreed"] == "8" and abs(float(external["raw_agreement"]) - 8 / 9) < 1e-12
    assert abs(float(external["cohen_kappa"]) - .82) < 1e-12
    assert abs(float(match(pairs, comparison="author_original_vs_rater1")["cohen_kappa"]) - .5) < 1e-12
    assert abs(float(match(pairs, comparison="author_original_vs_rater2")["cohen_kappa"]) - 2 / 3) < 1e-12

    fleiss = read("fleiss_kappa_three_raters.csv")
    assert len(fleiss) == 1 and abs(float(fleiss[0]["fleiss_kappa"]) - 17 / 26) < 1e-12

    outcomes = read("tier_mapping_sensitivity_all_outcomes.csv")
    assert len(outcomes) == 20
    assert sum(row["mapping_name"] == "rater1" for row in outcomes) == 10
    assert sum(row["mapping_name"] == "rater2" for row in outcomes) == 10
    assert all(float(row["bh_fdr_q"]) >= .05 for row in outcomes)
    assert set(row["permutations"] for row in outcomes if row["mapping_name"] == "rater1") == {"1260"}
    assert set(row["permutations"] for row in outcomes if row["mapping_name"] == "rater2") == {"504"}

    summary = read("tier_mapping_sensitivity_summary.csv")
    first, second = match(summary, mapping_name="rater1"), match(summary, mapping_name="rater2")
    assert first["smallest_p_outcome"] == second["smallest_p_outcome"] == "causal_per_100"
    assert abs(float(first["smallest_raw_p"]) - 26 / 1260) < 1e-12
    assert abs(float(first["corresponding_fdr_q"]) - 26 / 126) < 1e-12
    assert abs(float(second["smallest_raw_p"]) - 25 / 504) < 1e-12
    assert abs(float(second["corresponding_fdr_q"]) - 145 / 378) < 1e-12
    assert first["any_q_below_05"] == second["any_q_below_05"] == "False"
    print("Validated: external assignments, agreement statistics, 20 exact sensitivity results, and no external-mapping BH q < .05.")


if __name__ == "__main__":
    main()
