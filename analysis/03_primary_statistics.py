from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "reddit_posts_features.csv"
OUTPUT_PATH = ROOT / "results" / "primary_statistics.csv"

TIER_ORDER = ["low", "medium", "high"]

FEATURES = [
    "fk_grade",
    "gunning_fog",
    "smog",
    "avg_sentence_length",
    "avg_syllables_per_word",
    "type_token_ratio",
    "mattr",
    "hedge_per_100",
    "causal_per_100",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_compound",
    "word_count",
]


def cohens_d_independent(a, b):
    """
    Reproduces the original study's high-vs-low effect-size calculation:
    difference in means divided by the square root of the average
    of the two sample variances.
    """
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled_sd = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)

    if pooled_sd == 0:
        return 0.0

    return (a.mean() - b.mean()) / pooled_sd


def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}\n"
            "Run analysis/02_linguistic_features.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"expertise_tier"} | set(FEATURES)
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    rows = []

    print(f"Loaded {len(df):,} posts")
    print("\nPrimary Kruskal-Wallis tests and high-vs-low Cohen's d\n")

    for feature in FEATURES:
        groups = [
            df.loc[df["expertise_tier"] == tier, feature].dropna()
            for tier in TIER_ORDER
        ]

        if any(len(group) == 0 for group in groups):
            h_stat = np.nan
            p_value = np.nan
        else:
            h_stat, p_value = stats.kruskal(*groups)

        low = df.loc[
            df["expertise_tier"] == "low",
            feature
        ].dropna()

        medium = df.loc[
            df["expertise_tier"] == "medium",
            feature
        ].dropna()

        high = df.loc[
            df["expertise_tier"] == "high",
            feature
        ].dropna()

        d_high_low = cohens_d_independent(high, low)

        row = {
            "feature": feature,
            "n_low": len(low),
            "n_medium": len(medium),
            "n_high": len(high),
            "mean_low": low.mean(),
            "mean_medium": medium.mean(),
            "mean_high": high.mean(),
            "sd_low": low.std(ddof=1),
            "sd_medium": medium.std(ddof=1),
            "sd_high": high.std(ddof=1),
            "kruskal_h": h_stat,
            "p_value": p_value,
            "significance": significance_label(p_value)
            if pd.notna(p_value)
            else "",
            "cohens_d_high_vs_low": d_high_low,
        }

        rows.append(row)

        p_display = (
            f"{p_value:.4g}"
            if pd.notna(p_value)
            else "NA"
        )

        print(
            f"{feature:<28} "
            f"H={h_stat:>8.3f}  "
            f"p={p_display:<10} "
            f"d={d_high_low:>7.3f}"
        )

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
