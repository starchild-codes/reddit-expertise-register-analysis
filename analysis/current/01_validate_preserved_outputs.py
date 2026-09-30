"""Validate canonical preserved outputs against the final submitted manuscript."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "current"

def read(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def main() -> None:
    audit = read("table1_subreddit_composition_ai_audit.csv")
    exact = read("table3_exact_cluster_permutation_fdr.csv")
    adjusted = read("table4_topic_length_adjusted_cluster_results.csv")
    classifier = read("table5_grouped_classifier_performance_permutation.csv")
    assert len(audit) == 9 and sum(int(r["total_posts"]) for r in audit) == 1778
    assert sum(int(r["ai_relevant_posts"]) for r in audit) == 1299
    syl = next(r for r in exact if r["outcome"] == "avg_syllables_per_word")
    assert abs(float(syl["exact_p"]) - 0.05357142857142857) < 1e-12
    assert all(float(r["fdr_p"]) >= 0.5357142857142857 for r in exact)
    assert len(adjusted) == 10 and all(float(r["fdr_p"]) >= 0.8482142857142857 for r in adjusted)
    logit = next(r for r in classifier if r["model"] == "logistic_regression")
    assert abs(float(logit["balanced_accuracy"]) - 0.33358954283613435) < 1e-12
    print("Validated: 1,778 posts; 1,299 AI-screened posts; 10 primary and 10 adjusted outcomes; grouped classifier results.")

if __name__ == "__main__":
    main()
