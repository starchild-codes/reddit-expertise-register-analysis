from pathlib import Path

import pandas as pd
import scikit_posthocs as sp


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "posthoc_dunn_bonferroni.csv"
)


TIER_ORDER = [
    "low",
    "medium",
    "high",
]


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


PAIRS = [
    (
        "low",
        "medium"
    ),
    (
        "low",
        "high"
    ),
    (
        "medium",
        "high"
    ),
]


def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{INPUT_PATH}\n"
            "Run analysis/02_linguistic_features.py first."
        )


    df = pd.read_csv(
        INPUT_PATH
    )


    required = {
        "expertise_tier"
    } | set(
        FEATURES
    )


    missing = (
        required
        - set(
            df.columns
        )
    )


    if missing:

        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )


    df["expertise_tier"] = (
        df["expertise_tier"]
        .astype(str)
        .str.lower()
        .str.strip()
    )


    print(
        f"Loaded {len(df):,} posts"
    )

    print(
        "\nDunn post-hoc tests "
        "with Bonferroni correction\n"
    )


    rows = []


    for feature in FEATURES:

        subset = (
            df[
                [
                    "expertise_tier",
                    feature
                ]
            ]
            .dropna()
            .copy()
        )


        subset = (
            subset[
                subset[
                    "expertise_tier"
                ]
                .isin(
                    TIER_ORDER
                )
            ]
        )


        if (
            subset[
                "expertise_tier"
            ]
            .nunique()
            < 2
        ):
            continue


        matrix = sp.posthoc_dunn(
            subset,
            val_col=feature,
            group_col="expertise_tier",
            p_adjust="bonferroni",
        )


        print(
            f"--- {feature} ---"
        )


        for (
            tier_a,
            tier_b
        ) in PAIRS:

            p_bonferroni = (
                matrix.loc[
                    tier_a,
                    tier_b
                ]
            )


            rows.append({
                "feature":
                    feature,

                "tier_a":
                    tier_a,

                "tier_b":
                    tier_b,

                "p_bonferroni":
                    p_bonferroni,
            })


            print(
                f"{tier_a:<7} vs "
                f"{tier_b:<7} "
                f"p={p_bonferroni:.6g}"
            )


        print()


    results = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "tier_a",
            "tier_b",
            "p_bonferroni",
        ]
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    results.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "Saved post-hoc results to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
