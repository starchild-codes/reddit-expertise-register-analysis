"""Recompute external-rater agreement and exact tier-mapping sensitivity.

The outcomes are preserved subreddit means, so the independent unit is the
subreddit.  The permutation test enumerates every allocation with the observed
tier sizes for each external rater; it never forces the original 3/3/3 design.
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "current"
OUT = RESULTS / "rater_sensitivity"
TIERS = ("low", "medium", "high")
SUBREDDITS = (
    "explainlikeimfive", "Futurology", "GenerativeAI", "ChatGPT", "OpenAI",
    "learnmachinelearning", "LocalLLaMA", "MachineLearning", "deeplearning",
)
MAPPINGS = {
    "author_original": {
        "explainlikeimfive": "low", "Futurology": "low", "GenerativeAI": "low",
        "ChatGPT": "medium", "OpenAI": "medium", "learnmachinelearning": "medium",
        "LocalLLaMA": "high", "MachineLearning": "high", "deeplearning": "high",
    },
    "rater1": {
        "explainlikeimfive": "low", "ChatGPT": "low",
        "Futurology": "medium", "GenerativeAI": "medium", "OpenAI": "medium",
        "learnmachinelearning": "medium", "LocalLLaMA": "high",
        "MachineLearning": "high", "deeplearning": "high",
    },
    "rater2": {
        "explainlikeimfive": "low", "ChatGPT": "medium", "Futurology": "medium",
        "GenerativeAI": "medium", "OpenAI": "medium", "learnmachinelearning": "medium",
        "LocalLLaMA": "high", "MachineLearning": "high", "deeplearning": "high",
    },
}


def write_csv(name: str, fieldnames: list[str], records: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def cohen_kappa(left: list[str], right: list[str]) -> tuple[float, float, int]:
    """Return raw agreement, unweighted nominal Cohen kappa, and agreements."""
    total = len(left)
    agreed = sum(a == b for a, b in zip(left, right))
    observed = agreed / total
    expected = sum(
        (left.count(tier) / total) * (right.count(tier) / total) for tier in TIERS
    )
    return observed, (observed - expected) / (1 - expected), agreed


def fleiss_kappa(ratings: list[list[str]]) -> tuple[float, float, float]:
    """Compute nominal Fleiss kappa for items x raters labels."""
    items, raters = len(ratings), len(ratings[0])
    category_proportions = {
        tier: sum(item.count(tier) for item in ratings) / (items * raters)
        for tier in TIERS
    }
    item_agreements = [
        (sum(item.count(tier) ** 2 for tier in TIERS) - raters) / (raters * (raters - 1))
        for item in ratings
    ]
    mean_observed = sum(item_agreements) / items
    expected = sum(value ** 2 for value in category_proportions.values())
    return (mean_observed - expected) / (1 - expected), mean_observed, expected


def allocations(counts: tuple[int, int, int], n: int) -> list[tuple[tuple[int, ...], ...]]:
    """Enumerate labelled low/medium/high group allocations with fixed sizes."""
    all_indices = tuple(range(n))
    result = []
    for low in combinations(all_indices, counts[0]):
        remaining = tuple(index for index in all_indices if index not in low)
        for medium in combinations(remaining, counts[1]):
            high = tuple(index for index in remaining if index not in medium)
            result.append((low, medium, high))
    return result


def dispersion(values: list[float], groups: tuple[tuple[int, ...], ...]) -> float:
    """Weighted between-tier sum of squares around the nine-subreddit mean."""
    grand_mean = sum(values) / len(values)
    return sum(
        len(group) * ((sum(values[index] for index in group) / len(group)) - grand_mean) ** 2
        for group in groups
    )


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running_minimum = 1.0
    for rank in range(count, 0, -1):
        index = ordered[rank - 1]
        running_minimum = min(running_minimum, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running_minimum)
    return adjusted


def format_number(value: float) -> str:
    return format(value, ".16g")


def main() -> None:
    source = RESULTS / "subreddit_linguistic_descriptives.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        descriptive_rows = list(csv.DictReader(handle))
    by_subreddit = {row["subreddit"]: row for row in descriptive_rows}
    if set(by_subreddit) != set(SUBREDDITS) or len(by_subreddit) != 9:
        raise ValueError("The canonical descriptive CSV must contain exactly the nine manuscript subreddits.")
    outcomes = [
        name for name in descriptive_rows[0]
        if name not in {"subreddit", "expertise_tier"}
    ]

    assignment_rows = [
        {
            "subreddit": subreddit,
            "author_original_tier": MAPPINGS["author_original"][subreddit],
            "rater1_tier": MAPPINGS["rater1"][subreddit],
            "rater2_tier": MAPPINGS["rater2"][subreddit],
        }
        for subreddit in SUBREDDITS
    ]
    write_csv(
        "tier_assignments_three_raters.csv",
        ["subreddit", "author_original_tier", "rater1_tier", "rater2_tier"], assignment_rows,
    )

    pairwise_rows = []
    for left_name, right_name in (
        ("author_original", "rater1"),
        ("author_original", "rater2"),
        ("rater1", "rater2"),
    ):
        left = [MAPPINGS[left_name][subreddit] for subreddit in SUBREDDITS]
        right = [MAPPINGS[right_name][subreddit] for subreddit in SUBREDDITS]
        raw, kappa, agreed = cohen_kappa(left, right)
        pairwise_rows.append({
            "comparison": f"{left_name}_vs_{right_name}", "n_communities": len(SUBREDDITS),
            "n_agreed": agreed, "raw_agreement": format_number(raw),
            "cohen_kappa": format_number(kappa),
        })
    write_csv(
        "pairwise_agreement.csv",
        ["comparison", "n_communities", "n_agreed", "raw_agreement", "cohen_kappa"], pairwise_rows,
    )

    all_ratings = [
        [MAPPINGS[mapping][subreddit] for mapping in ("author_original", "rater1", "rater2")]
        for subreddit in SUBREDDITS
    ]
    kappa, mean_observed, expected = fleiss_kappa(all_ratings)
    write_csv(
        "fleiss_kappa_three_raters.csv",
        ["n_communities", "n_raters", "fleiss_kappa", "mean_item_agreement", "expected_agreement"],
        [{"n_communities": 9, "n_raters": 3, "fleiss_kappa": format_number(kappa),
          "mean_item_agreement": format_number(mean_observed), "expected_agreement": format_number(expected)}],
    )

    all_sensitivity_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    adjusted_status = (
        "not_publicly_recomputable: subreddit-level topic/length-adjusted residual means "
        "are not included in the public repository"
    )
    for mapping_name in ("rater1", "rater2"):
        mapping = MAPPINGS[mapping_name]
        counts = tuple(sum(mapping[subreddit] == tier for subreddit in SUBREDDITS) for tier in TIERS)
        null_allocations = allocations(counts, len(SUBREDDITS))
        observed_groups = tuple(
            tuple(index for index, subreddit in enumerate(SUBREDDITS) if mapping[subreddit] == tier)
            for tier in TIERS
        )
        mapping_rows = []
        for outcome in outcomes:
            values = [float(by_subreddit[subreddit][outcome]) for subreddit in SUBREDDITS]
            observed = dispersion(values, observed_groups)
            extreme = sum(
                dispersion(values, candidate) >= observed - 1e-12
                for candidate in null_allocations
            )
            mapping_rows.append({
                "mapping_name": mapping_name, "low_n": counts[0], "medium_n": counts[1], "high_n": counts[2],
                "outcome": outcome, "observed_between_tier_dispersion": format_number(observed),
                "permutation_p": format_number(extreme / len(null_allocations)),
                "bh_fdr_q": "", "permutations": len(null_allocations),
            })
        q_values = benjamini_hochberg([float(row["permutation_p"]) for row in mapping_rows])
        for row, q_value in zip(mapping_rows, q_values):
            row["bh_fdr_q"] = format_number(q_value)
        all_sensitivity_rows.extend(mapping_rows)
        strongest = min(mapping_rows, key=lambda row: float(row["permutation_p"]))
        summary_rows.append({
            "mapping_name": mapping_name, "low_n": counts[0], "medium_n": counts[1], "high_n": counts[2],
            "smallest_raw_p": strongest["permutation_p"], "smallest_p_outcome": strongest["outcome"],
            "corresponding_fdr_q": strongest["bh_fdr_q"],
            "any_q_below_05": any(float(row["bh_fdr_q"]) < .05 for row in mapping_rows),
            "adjusted_analysis_status": adjusted_status,
        })
    write_csv(
        "tier_mapping_sensitivity_all_outcomes.csv",
        ["mapping_name", "low_n", "medium_n", "high_n", "outcome", "observed_between_tier_dispersion",
         "permutation_p", "bh_fdr_q", "permutations"], all_sensitivity_rows,
    )
    write_csv(
        "tier_mapping_sensitivity_summary.csv",
        ["mapping_name", "low_n", "medium_n", "high_n", "smallest_raw_p", "smallest_p_outcome",
         "corresponding_fdr_q", "any_q_below_05", "adjusted_analysis_status"], summary_rows,
    )
    print("Wrote external-rater agreement and exact alternative-mapping sensitivity outputs.")


if __name__ == "__main__":
    main()
