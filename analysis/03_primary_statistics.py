from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "primary_statistics.csv"
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


def cohens_d_high_vs_low(
    high,
    low
):
    """
    Reproduce the effect-size convention used in the study:

    (mean_high - mean_low)
    /
    sqrt((variance_high + variance_low) / 2)
    """

    high = (
        pd.Series(high)
        .dropna()
        .astype(float)
    )

    low = (
        pd.Series(low)
        .dropna()
        .astype(float)
    )


    if (
        len(high) < 2
        or len(low) < 2
    ):
        return np.nan


    denominator = np.sqrt(
        (
            high.var(
                ddof=1
            )
            +
            low.var(
                ddof=1
            )
        )
        / 2
    )


    if (
        denominator == 0
        or pd.isna(
            denominator
        )
    ):
        return np.nan


    return (
        high.mean()
        - low.mean()
    ) / denominator


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
        "\nPrimary Kruskal-Wallis tests "
        "and high-vs-low Cohen's d\n"
    )


    rows = []


    for feature in FEATURES:

        low = (
            df.loc[
                df[
                    "expertise_tier"
                ] == "low",
                feature
            ]
            .dropna()
            .astype(float)
        )

        medium = (
            df.loc[
                df[
                    "expertise_tier"
                ] == "medium",
                feature
            ]
            .dropna()
            .astype(float)
        )

        high = (
            df.loc[
                df[
                    "expertise_tier"
                ] == "high",
                feature
            ]
            .dropna()
            .astype(float)
        )


        if (
            len(low) == 0
            or len(medium) == 0
            or len(high) == 0
        ):

            h_stat = np.nan
            p_value = np.nan

        else:

            h_stat, p_value = (
                stats.kruskal(
                    low,
                    medium,
                    high,
                )
            )


        d_high_low = (
            cohens_d_high_vs_low(
                high,
                low
            )
        )


        rows.append({
            "feature":
                feature,

            "low_mean":
                low.mean(),

            "low_sd":
                low.std(
                    ddof=1
                ),

            "medium_mean":
                medium.mean(),

            "medium_sd":
                medium.std(
                    ddof=1
                ),

            "high_mean":
                high.mean(),

            "high_sd":
                high.std(
                    ddof=1
                ),

            "kruskal_h":
                h_stat,

            "p_value":
                p_value,

            "cohens_d_high_vs_low":
                d_high_low,
        })


        print(
            f"{feature:<28} "
            f"H={h_stat:>8.3f}  "
            f"p={p_value:.6g}  "
            f"d={d_high_low:+.3f}"
        )


    results = pd.DataFrame(
        rows
    )


    # ========================================================
    # BH correction across the 14 primary omnibus tests
    # ========================================================

    valid_mask = (
        results[
            "p_value"
        ]
        .notna()
    )


    results[
        "fdr_q_14_metric_family"
    ] = np.nan


    if valid_mask.sum() > 0:

        _, q_values, _, _ = (
            multipletests(
                results.loc[
                    valid_mask,
                    "p_value"
                ].astype(float),
                alpha=0.05,
                method="fdr_bh",
            )
        )


        results.loc[
            valid_mask,
            "fdr_q_14_metric_family"
        ] = q_values


    # Exact canonical output order.
    results = results[
        [
            "feature",
            "low_mean",
            "low_sd",
            "medium_mean",
            "medium_sd",
            "high_mean",
            "high_sd",
            "kruskal_h",
            "p_value",
            "cohens_d_high_vs_low",
            "fdr_q_14_metric_family",
        ]
    ]


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    results.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "\nSaved primary statistics to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
