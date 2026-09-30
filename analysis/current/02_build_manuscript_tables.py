"""Build the manuscript-facing Table 2 from preserved cluster-aware result CSVs.

This is a deterministic column join, not a new statistical analysis.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "current"

def load(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def main() -> None:
    exact = {r["outcome"]: r for r in load("table3_exact_cluster_permutation_fdr.csv")}
    adjusted = {r["outcome"]: r for r in load("table4_topic_length_adjusted_cluster_results.csv")}
    fields = ["outcome", "low_subreddit_mean", "medium_subreddit_mean", "high_subreddit_mean", "exact_p", "fdr_p", "adjusted_cluster_perm_p", "adjusted_fdr_p"]
    with (RESULTS / "table2_cluster_permutation_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for outcome, row in exact.items():
            adj = adjusted[outcome]
            w.writerow({
                "outcome": outcome, "low_subreddit_mean": row["low_subreddit_mean"],
                "medium_subreddit_mean": row["medium_subreddit_mean"], "high_subreddit_mean": row["high_subreddit_mean"],
                "exact_p": row["exact_p"], "fdr_p": row["fdr_p"],
                "adjusted_cluster_perm_p": adj["adjusted_cluster_perm_p"], "adjusted_fdr_p": adj["fdr_p"],
            })

if __name__ == "__main__":
    main()
