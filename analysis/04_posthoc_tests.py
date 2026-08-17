from pathlib import Path

import pandas as pd
import scikit_posthocs as sp


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "reddit_posts_features.csv"
OUTPUT_PATH = ROOT / "results" / "posthoc_dunn_bonferroni.csv"

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


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}\n"
            "Run analysis/02_linguistic_features.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {"expertise_tier"} | set(FEATURES)
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    all_rows = []

    print(f"Loaded {len(df):,} posts")
    print("\nDunn post-hoc tests with Bonferroni correction\n")

    for feature in FEATURES:
        subset = df[
            ["expertise_tier", feature]
        ].dropna()

        subset = subset[
            subset["expertise_tier"].isin(TIER_ORDER)
        ]

        if subset["expertise_tier"].nunique() < 2:
            continue

        matrix = sp.posthoc_dunn(
            subset,
            val_col=feature,
            group_col="expertise_tier",
            p_adjust="bonferroni"
        )

        pairs = [
            ("low", "medium"),
            ("low", "high"),
            ("medium", "high"),
        ]

        print(f"--- {feature} ---")

        for tier_a, tier_b in pairs:
            p_adj = matrix.loc[tier_a, tier_b]

            row = {
                "feature": feature,
                "tier_a": tier_a,
                "tier_b": tier_b,
                "p_bonferroni": p_adj,
                "significant_0_05": p_adj < 0.05,
            }

            all_rows.append(row)

            print(
                f"{tier_a:<7} vs {tier_b:<7} "
                f"p_adj={p_adj:.6g}"
            )

        print()

    results = pd.DataFrame(all_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
