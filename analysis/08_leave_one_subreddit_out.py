from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

ACRONYM_INPUT_PATH = (
    ROOT / "data" / "reddit_posts_features_acronyms.csv"
)

BASE_INPUT_PATH = (
    ROOT / "data" / "reddit_posts_features.csv"
)

ALL_OUTPUT_PATH = (
    ROOT / "results" / "leave_one_subreddit_out.csv"
)

PARTIAL_OUTPUT_PATH = (
    ROOT
    / "results"
    / "leave_one_subreddit_out_partial_correlations.csv"
)

INFLUENCE_OUTPUT_PATH = (
    ROOT
    / "results"
    / "leave_one_subreddit_out_influential_subreddits.csv"
)


SUBREDDIT_COL = "subreddit"
TIER_COL = "expertise_tier"
WORD_COUNT_COL = "word_count"

TIER_ORDER = [
    "low",
    "medium",
    "high",
]

TIER_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


PRIMARY_METRICS = [
    "avg_syllables_per_word",
    "avg_syllables_per_word_without_acronyms",
    "avg_syllables_per_word_acronyms_normalized",
    "acronym_density_per_100_words",
]

SECONDARY_METRICS = [
    "sentiment_compound",
    "sentiment_neg",
]

EXPLORATORY_METRICS = [
    "fk_grade",
    "gunning_fog",
    "smog",
    "mattr",
    "hedge_per_100",
    "causal_per_100",
]

ALL_CANDIDATE_METRICS = (
    PRIMARY_METRICS
    + SECONDARY_METRICS
    + EXPLORATORY_METRICS
)


def cohens_d(group_a, group_b):
    group_a = (
        pd.Series(group_a)
        .dropna()
    )

    group_b = (
        pd.Series(group_b)
        .dropna()
    )

    if (
        len(group_a) < 2
        or len(group_b) < 2
    ):
        return np.nan

    pooled_sd = np.sqrt(
        (
            group_a.std(ddof=1) ** 2
            + group_b.std(ddof=1) ** 2
        )
        / 2
    )

    if (
        pooled_sd == 0
        or pd.isna(pooled_sd)
    ):
        return np.nan

    return (
        group_a.mean()
        - group_b.mean()
    ) / pooled_sd


def run_dunn(df, metric):
    subset = (
        df[
            [
                TIER_COL,
                metric,
            ]
        ]
        .dropna()
    )

    try:
        result = sp.posthoc_dunn(
            subset,
            val_col=metric,
            group_col=TIER_COL,
            p_adjust="bonferroni",
        )

        return {
            "dunn_high_vs_low_p":
                result.loc[
                    "high",
                    "low"
                ],

            "dunn_high_vs_medium_p":
                result.loc[
                    "high",
                    "medium"
                ],

            "dunn_low_vs_medium_p":
                result.loc[
                    "low",
                    "medium"
                ],
        }

    except Exception:
        return {
            "dunn_high_vs_low_p": np.nan,
            "dunn_high_vs_medium_p": np.nan,
            "dunn_low_vs_medium_p": np.nan,
        }


def get_tier_values(df, metric):
    return {
        tier: (
            df.loc[
                df[TIER_COL] == tier,
                metric
            ]
            .dropna()
        )
        for tier in TIER_ORDER
    }


def run_metric_analysis(
    df,
    metric,
    dropped_subreddit
):
    values = get_tier_values(
        df,
        metric
    )

    if any(
        len(values[tier]) < 2
        for tier in TIER_ORDER
    ):
        return None

    low = values["low"]
    medium = values["medium"]
    high = values["high"]

    h_stat, p_value = stats.kruskal(
        low,
        medium,
        high,
    )

    dunn = run_dunn(
        df,
        metric
    )

    d_high_vs_low = cohens_d(
        high,
        low
    )

    d_high_vs_medium = cohens_d(
        high,
        medium
    )

    low_mean = low.mean()
    medium_mean = medium.mean()
    high_mean = high.mean()

    # Direction checks are only meaningful for metrics for which
    # the manuscript makes directional statements.
    if metric == "sentiment_neg":
        direction_preserved = (
            high_mean < low_mean
            and high_mean < medium_mean
        )

    elif metric in [
        "avg_syllables_per_word",
        "avg_syllables_per_word_without_acronyms",
        "avg_syllables_per_word_acronyms_normalized",
        "acronym_density_per_100_words",
        "sentiment_compound",
    ]:
        direction_preserved = (
            high_mean > low_mean
            and high_mean > medium_mean
        )

    else:
        direction_preserved = np.nan

    if metric in (
        PRIMARY_METRICS
        + SECONDARY_METRICS
    ):
        pattern_preserved = (
            bool(direction_preserved)
            and p_value < 0.05
            and dunn[
                "dunn_high_vs_low_p"
            ] < 0.05
            and dunn[
                "dunn_high_vs_medium_p"
            ] < 0.05
        )
    else:
        pattern_preserved = np.nan

    return {
        "dropped_subreddit":
            dropped_subreddit,

        "metric":
            metric,

        "n_posts":
            len(
                pd.concat(
                    [
                        low,
                        medium,
                        high
                    ]
                )
            ),

        "low_mean":
            low_mean,

        "medium_mean":
            medium_mean,

        "high_mean":
            high_mean,

        "kruskal_h":
            h_stat,

        "kruskal_p":
            p_value,

        "dunn_high_vs_low_p":
            dunn[
                "dunn_high_vs_low_p"
            ],

        "dunn_high_vs_medium_p":
            dunn[
                "dunn_high_vs_medium_p"
            ],

        "dunn_low_vs_medium_p":
            dunn[
                "dunn_low_vs_medium_p"
            ],

        "d_high_vs_low":
            d_high_vs_low,

        "d_high_vs_medium":
            d_high_vs_medium,

        "direction_preserved":
            direction_preserved,

        "pattern_preserved":
            pattern_preserved,
    }


def extract_p_value(result):
    for column in result.columns:
        normalized = (
            column.lower()
            .replace("-", "")
            .replace("_", "")
        )

        if "pval" in normalized:
            return (
                result[column]
                .iloc[0]
            )

    raise KeyError(
        "Could not identify Pingouin p-value column. "
        f"Columns returned: {result.columns.tolist()}"
    )


def run_partial_corr(
    df,
    metric,
    covariates,
    dropped_subreddit
):
    required = [
        metric,
        "tier_numeric",
        *covariates,
    ]

    if any(
        column not in df.columns
        for column in required
    ):
        return None

    subset = (
        df[
            required
        ]
        .dropna()
    )

    if len(subset) < 10:
        return None

    result = pg.partial_corr(
        data=subset,
        x=metric,
        y="tier_numeric",
        covar=covariates,
        method="pearson",
    )

    r_value = (
        result["r"]
        .iloc[0]
    )

    p_value = extract_p_value(
        result
    )

    return {
        "dropped_subreddit":
            dropped_subreddit,

        "metric":
            metric,

        "covariates":
            " + ".join(
                covariates
            ),

        "n":
            len(subset),

        "r_partial":
            r_value,

        "p_value":
            p_value,

        "significant_0_05":
            p_value < 0.05,
    }


def identify_influential_subreddits(
    results
):
    rows = []

    for metric in (
        results["metric"]
        .dropna()
        .unique()
    ):
        metric_df = (
            results[
                results["metric"]
                == metric
            ]
            .copy()
        )

        baseline = metric_df[
            metric_df[
                "dropped_subreddit"
            ]
            == "FULL_DATASET"
        ]

        loo = metric_df[
            metric_df[
                "dropped_subreddit"
            ]
            != "FULL_DATASET"
        ].copy()

        if (
            baseline.empty
            or loo.empty
        ):
            continue

        baseline_h = (
            baseline[
                "kruskal_h"
            ]
            .iloc[0]
        )

        baseline_d = (
            baseline[
                "d_high_vs_low"
            ]
            .iloc[0]
        )

        loo["h_change"] = (
            loo["kruskal_h"]
            - baseline_h
        )

        loo["d_change"] = (
            loo["d_high_vs_low"]
            - baseline_d
        )

        largest_h_decrease = (
            loo.loc[
                loo[
                    "h_change"
                ]
                .idxmin()
            ]
        )

        largest_d_decrease = (
            loo.loc[
                loo[
                    "d_change"
                ]
                .idxmin()
            ]
        )

        largest_d_increase = (
            loo.loc[
                loo[
                    "d_change"
                ]
                .idxmax()
            ]
        )

        rows.append({
            "metric":
                metric,

            "baseline_h":
                baseline_h,

            "baseline_d_high_vs_low":
                baseline_d,

            "largest_h_decrease_when_dropping":
                largest_h_decrease[
                    "dropped_subreddit"
                ],

            "largest_h_decrease_amount":
                largest_h_decrease[
                    "h_change"
                ],

            "largest_d_decrease_when_dropping":
                largest_d_decrease[
                    "dropped_subreddit"
                ],

            "largest_d_decrease_amount":
                largest_d_decrease[
                    "d_change"
                ],

            "largest_d_increase_when_dropping":
                largest_d_increase[
                    "dropped_subreddit"
                ],

            "largest_d_increase_amount":
                largest_d_increase[
                    "d_change"
                ],

            "any_leave_one_out_non_significant":
                bool(
                    (
                        loo["kruskal_p"]
                        >= 0.05
                    )
                    .any()
                ),

            "all_high_vs_low_direction_preserved":
                bool(
                    (
                        loo["high_mean"]
                        > loo["low_mean"]
                    )
                    .all()
                )
                if metric
                != "sentiment_neg"
                else bool(
                    (
                        loo["high_mean"]
                        < loo["low_mean"]
                    )
                    .all()
                ),
        })

    return pd.DataFrame(
        rows
    )


def main():
    # Prefer the acronym-enriched dataset so that LOSO uses
    # exactly the same acronym-adjusted variables generated by
    # analysis/07_acronym_robustness.py.
    if ACRONYM_INPUT_PATH.exists():
        input_path = (
            ACRONYM_INPUT_PATH
        )

    elif BASE_INPUT_PATH.exists():
        input_path = (
            BASE_INPUT_PATH
        )

        print(
            "Warning: acronym-enriched dataset not found. "
            "Acronym robustness variables will be skipped."
        )

    else:
        raise FileNotFoundError(
            "No feature dataset found. "
            "Run analysis/02_linguistic_features.py first."
        )

    df = pd.read_csv(
        input_path
    )

    required = {
        SUBREDDIT_COL,
        TIER_COL,
        WORD_COUNT_COL,
        "avg_syllables_per_word",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    df[TIER_COL] = (
        df[TIER_COL]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["tier_numeric"] = (
        df[TIER_COL]
        .map(TIER_MAP)
    )

    if (
        df["tier_numeric"]
        .isna()
        .any()
    ):
        bad_labels = (
            df.loc[
                df[
                    "tier_numeric"
                ]
                .isna(),
                TIER_COL
            ]
            .unique()
        )

        raise ValueError(
            "Unexpected expertise-tier labels: "
            f"{bad_labels.tolist()}"
        )

    available_metrics = [
        metric
        for metric
        in ALL_CANDIDATE_METRICS
        if metric in df.columns
    ]

    print(
        f"Loaded {len(df):,} posts "
        f"from {input_path.name}"
    )

    print(
        f"Using {len(available_metrics)} metrics"
    )

    subreddits = sorted(
        df[
            SUBREDDIT_COL
        ]
        .dropna()
        .unique()
    )

    print(
        f"Found {len(subreddits)} subreddits:"
    )

    for subreddit in subreddits:
        print(
            f"  {subreddit}"
        )

    # ============================================================
    # FULL DATASET + LOSO
    # ============================================================

    rows = []

    for metric in available_metrics:
        result = run_metric_analysis(
            df,
            metric,
            "FULL_DATASET",
        )

        if result is not None:
            rows.append(
                result
            )

    for dropped in subreddits:
        print(
            f"\nDropping {dropped}..."
        )

        subset = (
            df[
                df[
                    SUBREDDIT_COL
                ]
                != dropped
            ]
            .copy()
        )

        for metric in available_metrics:
            result = run_metric_analysis(
                subset,
                metric,
                dropped,
            )

            if result is not None:
                rows.append(
                    result
                )

    results = pd.DataFrame(
        rows
    )

    ALL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        ALL_OUTPUT_PATH,
        index=False
    )

    # ============================================================
    # PARTIAL CORRELATIONS
    # ============================================================

    partial_specs = [
        (
            "avg_syllables_per_word",
            [
                WORD_COUNT_COL
            ],
        ),
        (
            "sentiment_compound",
            [
                WORD_COUNT_COL
            ],
        ),
        (
            "sentiment_neg",
            [
                WORD_COUNT_COL
            ],
        ),
    ]

    if (
        "acronym_density_per_100_words"
        in df.columns
    ):
        partial_specs.extend([
            (
                "avg_syllables_per_word",
                [
                    WORD_COUNT_COL,
                    "acronym_density_per_100_words",
                ],
            ),
            (
                "avg_syllables_per_word_without_acronyms",
                [
                    WORD_COUNT_COL
                ],
            ),
            (
                "avg_syllables_per_word_acronyms_normalized",
                [
                    WORD_COUNT_COL
                ],
            ),
        ])

    partial_rows = []

    for metric, covariates in partial_specs:
        if metric not in df.columns:
            continue

        baseline = run_partial_corr(
            df,
            metric,
            covariates,
            "FULL_DATASET",
        )

        if baseline is not None:
            partial_rows.append(
                baseline
            )

        for dropped in subreddits:
            subset = (
                df[
                    df[
                        SUBREDDIT_COL
                    ]
                    != dropped
                ]
                .copy()
            )

            result = run_partial_corr(
                subset,
                metric,
                covariates,
                dropped,
            )

            if result is not None:
                partial_rows.append(
                    result
                )

    partial_results = (
        pd.DataFrame(
            partial_rows
        )
    )

    partial_results.to_csv(
        PARTIAL_OUTPUT_PATH,
        index=False
    )

    # ============================================================
    # INFLUENTIAL SUBREDDIT SUMMARY
    # ============================================================

    influence = (
        identify_influential_subreddits(
            results
        )
    )

    influence.to_csv(
        INFLUENCE_OUTPUT_PATH,
        index=False
    )

    # ============================================================
    # MAIN RESULT SUMMARY
    # ============================================================

    main = (
        results[
            results["metric"]
            == "avg_syllables_per_word"
        ]
        .copy()
    )

    if not main.empty:
        print(
            "\n"
            + "=" * 72
        )
        print(
            "AVG SYLLABLES PER WORD — LOSO RESULTS"
        )
        print(
            "=" * 72
        )

        print(
            main[
                [
                    "dropped_subreddit",
                    "low_mean",
                    "medium_mean",
                    "high_mean",
                    "kruskal_h",
                    "kruskal_p",
                    "d_high_vs_low",
                    "dunn_high_vs_low_p",
                    "dunn_high_vs_medium_p",
                    "pattern_preserved",
                ]
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

        loo = main[
            main[
                "dropped_subreddit"
            ]
            != "FULL_DATASET"
        ]

        print(
            "\nRobustness summary:"
        )

        print(
            "Kruskal-Wallis significant in all 9 runs: "
            f"{bool((loo['kruskal_p'] < 0.05).all())}"
        )

        print(
            "High > Low in all 9 runs: "
            f"{bool((loo['high_mean'] > loo['low_mean']).all())}"
        )

        print(
            "High > Medium in all 9 runs: "
            f"{bool((loo['high_mean'] > loo['medium_mean']).all())}"
        )

        print(
            "Strict pattern preserved in all 9 runs: "
            f"{bool(loo['pattern_preserved'].all())}"
        )

        if not loo.empty:
            minimum_d_row = (
                loo.loc[
                    loo[
                        "d_high_vs_low"
                    ]
                    .idxmin()
                ]
            )

            print(
                "Smallest high-vs-low Cohen's d: "
                f"{minimum_d_row['d_high_vs_low']:.4f} "
                f"when dropping "
                f"{minimum_d_row['dropped_subreddit']}"
            )

    print(
        "\nSaved:"
    )

    print(
        ALL_OUTPUT_PATH
    )

    print(
        PARTIAL_OUTPUT_PATH
    )

    print(
        INFLUENCE_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
