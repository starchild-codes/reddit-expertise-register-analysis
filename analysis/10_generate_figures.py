from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

FEATURES_PATH = DATA_DIR / "reddit_posts_features.csv"
ACRONYM_FEATURES_PATH = DATA_DIR / "reddit_posts_features_acronyms.csv"

PRIMARY_RESULTS_PATH = RESULTS_DIR / "primary_statistics.csv"
PARTIAL_RESULTS_PATH = RESULTS_DIR / "partial_correlations.csv"
RF_IMPORTANCE_PATH = RESULTS_DIR / "random_forest_feature_importance.csv"

LOSO_ALL_PATH = (
    RESULTS_DIR
    / "leave_one_subreddit_out_all_metrics.csv"
)

LOSO_PRIMARY_PATH = (
    RESULTS_DIR
    / "leave_one_subreddit_out_primary_syllables.csv"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# General configuration
# ============================================================

TIER_ORDER = [
    "low",
    "medium",
    "high",
]

SUBREDDIT_ORDER = [
    "explainlikeimfive",
    "Futurology",
    "GenerativeAI",
    "ChatGPT",
    "learnmachinelearning",
    "OpenAI",
    "LocalLLaMA",
    "MachineLearning",
    "deeplearning",
]


def save_figure(filename):
    """
    Apply common layout settings and save the active figure.
    """
    plt.tight_layout()

    output_path = FIGURES_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


def tier_mean_sd(df, metric):
    """
    Return mean, SD, and count by expertise tier.
    """
    summary = (
        df.groupby("expertise_tier")[metric]
        .agg(["mean", "std", "count"])
        .reindex(TIER_ORDER)
    )

    return summary


# ============================================================
# Load feature dataset
# ============================================================

if ACRONYM_FEATURES_PATH.exists():
    data_path = ACRONYM_FEATURES_PATH

elif FEATURES_PATH.exists():
    data_path = FEATURES_PATH

else:
    raise FileNotFoundError(
        "Could not find either:\n"
        f"{ACRONYM_FEATURES_PATH}\n"
        "or\n"
        f"{FEATURES_PATH}"
    )


df = pd.read_csv(data_path)

df["expertise_tier"] = (
    df["expertise_tier"]
    .astype(str)
    .str.lower()
    .str.strip()
)


print(
    "Using feature dataset:",
    data_path
)


# ============================================================
# Figure 1
# Mean word count by expertise tier
# ============================================================

if "word_count" in df.columns:

    summary = tier_mean_sd(
        df,
        "word_count",
    )

    x = np.arange(
        len(TIER_ORDER)
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        capsize=5,
    )

    plt.xticks(
        x,
        [
            "Low",
            "Medium",
            "High",
        ],
    )

    plt.ylabel(
        "Mean word count"
    )

    plt.xlabel(
        "Expertise tier"
    )

    plt.title(
        "Mean Word Count by Expertise Tier"
    )

    save_figure(
        "figure_01_word_count_by_tier.png"
    )


# ============================================================
# Figure 2
# Mean syllables per word by expertise tier
# ============================================================

if "avg_syllables_per_word" in df.columns:

    summary = tier_mean_sd(
        df,
        "avg_syllables_per_word",
    )

    x = np.arange(
        len(TIER_ORDER)
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        capsize=5,
    )

    plt.xticks(
        x,
        [
            "Low",
            "Medium",
            "High",
        ],
    )

    plt.ylabel(
        "Average syllables per word"
    )

    plt.xlabel(
        "Expertise tier"
    )

    plt.title(
        "Lexical Complexity by Expertise Tier"
    )

    save_figure(
        "figure_02_syllables_by_tier.png"
    )


# ============================================================
# Figure 3
# Flesch-Kincaid grade distribution
# ============================================================

if "fk_grade" in df.columns:

    box_data = []

    for tier in TIER_ORDER:

        values = (
            df.loc[
                df["expertise_tier"] == tier,
                "fk_grade",
            ]
            .dropna()
            .astype(float)
        )

        box_data.append(
            values
        )

    plt.figure(
        figsize=(7, 5)
    )

    plt.boxplot(
        box_data,
        tick_labels=[
            "Low",
            "Medium",
            "High",
        ],
        showfliers=False,
    )

    plt.ylabel(
        "Flesch-Kincaid grade level"
    )

    plt.xlabel(
        "Expertise tier"
    )

    plt.title(
        "Flesch-Kincaid Grade-Level Distribution"
    )

    save_figure(
        "figure_03_fk_grade_distribution.png"
    )


# ============================================================
# Figure 4
# Acronym robustness
# ============================================================

acronym_metrics = {
    "Original": (
        "avg_syllables_per_word"
    ),
    "Acronyms removed": (
        "avg_syllables_per_word_without_acronyms"
    ),
    "Acronyms normalized": (
        "avg_syllables_per_word_acronyms_normalized"
    ),
}

available_acronym_metrics = {
    label: metric
    for label, metric in acronym_metrics.items()
    if metric in df.columns
}


if len(
    available_acronym_metrics
) >= 2:

    labels = list(
        available_acronym_metrics.keys()
    )

    x = np.arange(
        len(TIER_ORDER)
    )

    width = (
        0.8
        / len(labels)
    )

    plt.figure(
        figsize=(9, 5.5)
    )

    for i, (
        label,
        metric,
    ) in enumerate(
        available_acronym_metrics.items()
    ):

        means = (
            df.groupby(
                "expertise_tier"
            )[metric]
            .mean()
            .reindex(
                TIER_ORDER
            )
        )

        offset = (
            i
            - (
                len(labels) - 1
            ) / 2
        ) * width

        plt.bar(
            x + offset,
            means,
            width=width,
            label=label,
        )

    plt.xticks(
        x,
        [
            "Low",
            "Medium",
            "High",
        ],
    )

    plt.xlabel(
        "Expertise tier"
    )

    plt.ylabel(
        "Average syllables per word"
    )

    plt.title(
        "Acronym Robustness of Lexical Complexity"
    )

    plt.legend()

    save_figure(
        "figure_04_acronym_robustness.png"
    )


# ============================================================
# Figure 5
# LOSO Cohen's d
# ============================================================

loso_df = None


if LOSO_ALL_PATH.exists():

    loso_df = pd.read_csv(
        LOSO_ALL_PATH
    )

    print(
        "Using full LOSO file:",
        LOSO_ALL_PATH
    )

elif LOSO_PRIMARY_PATH.exists():

    loso_df = pd.read_csv(
        LOSO_PRIMARY_PATH
    )

    print(
        "Using primary LOSO file:",
        LOSO_PRIMARY_PATH
    )

else:

    print(
        "No LOSO file found."
    )


if (
    loso_df is not None
    and not loso_df.empty
):

    # Full all-metric LOSO file
    if "metric" in loso_df.columns:

        syllable_loso = (
            loso_df[
                loso_df["metric"]
                == "avg_syllables_per_word"
            ]
            .copy()
        )

    else:

        # Compact syllables-only file
        syllable_loso = (
            loso_df.copy()
        )


    # Support either column naming convention
    if (
        "cohens_d_high_vs_low"
        in syllable_loso.columns
    ):

        d_column = (
            "cohens_d_high_vs_low"
        )

    elif (
        "d_high_vs_low"
        in syllable_loso.columns
    ):

        d_column = (
            "d_high_vs_low"
        )

    else:

        d_column = None


    if (
        d_column is not None
        and
        "dropped_subreddit"
        in syllable_loso.columns
    ):

        syllable_loso = (
            syllable_loso
            .dropna(
                subset=[
                    d_column
                ]
            )
            .copy()
        )

        plt.figure(
            figsize=(10, 5.5)
        )

        x = np.arange(
            len(
                syllable_loso
            )
        )

        plt.bar(
            x,
            syllable_loso[
                d_column
            ],
        )

        labels = (
            syllable_loso[
                "dropped_subreddit"
            ]
            .astype(str)
            .replace(
                {
                    "FULL_DATASET":
                    "Full dataset"
                }
            )
        )

        plt.xticks(
            x,
            labels,
            rotation=45,
            ha="right",
        )

        plt.axhline(
            y=0,
            linewidth=1,
        )

        plt.ylabel(
            "Cohen's d: high vs. low"
        )

        plt.xlabel(
            "Subreddit removed"
        )

        plt.title(
            "Leave-One-Subreddit-Out Effect Size"
        )

        save_figure(
            "figure_05_loso_cohens_d.png"
        )


# ============================================================
# Figure 6
# Hedging by expertise tier
# ============================================================

if "hedge_per_100" in df.columns:

    summary = tier_mean_sd(
        df,
        "hedge_per_100",
    )

    x = np.arange(
        len(TIER_ORDER)
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        capsize=5,
    )

    plt.xticks(
        x,
        [
            "Low",
            "Medium",
            "High",
        ],
    )

    plt.xlabel(
        "Expertise tier"
    )

    plt.ylabel(
        "Hedge terms per 100 words"
    )

    plt.title(
        "Hedging by Expertise Tier"
    )

    save_figure(
        "figure_06_hedging_by_tier.png"
    )


# ============================================================
# Figure 7
# Compound sentiment by subreddit
# ============================================================

if (
    "sentiment_compound"
    in df.columns
    and
    "subreddit"
    in df.columns
):

    sentiment_summary = (
        df.groupby(
            [
                "subreddit",
                "expertise_tier",
            ]
        )[
            "sentiment_compound"
        ]
        .mean()
        .reset_index()
        .sort_values(
            "sentiment_compound"
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    x = np.arange(
        len(
            sentiment_summary
        )
    )

    plt.bar(
        x,
        sentiment_summary[
            "sentiment_compound"
        ],
    )

    plt.xticks(
        x,
        sentiment_summary[
            "subreddit"
        ],
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Mean VADER compound sentiment"
    )

    plt.xlabel(
        "Subreddit"
    )

    plt.title(
        "Compound Sentiment by Subreddit"
    )

    plt.axhline(
        y=0,
        linewidth=1,
    )

    save_figure(
        "figure_07_sentiment_by_subreddit.png"
    )


# ============================================================
# Figure 8
# Normalized subreddit linguistic profiles
# ============================================================

heatmap_features = [
    "fk_grade",
    "gunning_fog",
    "smog",
    "avg_sentence_length",
    "avg_syllables_per_word",
    "mattr",
    "hedge_per_100",
    "causal_per_100",
    "sentiment_compound",
    "word_count",
]

heatmap_features = [
    feature
    for feature in heatmap_features
    if feature in df.columns
]


if (
    len(
        heatmap_features
    ) >= 2
    and
    "subreddit" in df.columns
):

    profile = (
        df.groupby(
            "subreddit"
        )[
            heatmap_features
        ]
        .mean()
    )

    available_subreddits = [
        subreddit
        for subreddit in SUBREDDIT_ORDER
        if subreddit
        in profile.index
    ]

    remaining = [
        subreddit
        for subreddit in profile.index
        if subreddit
        not in available_subreddits
    ]

    profile = profile.reindex(
        available_subreddits
        + sorted(
            remaining
        )
    )

    normalized = profile.copy()

    for column in normalized.columns:

        minimum = normalized[
            column
        ].min()

        maximum = normalized[
            column
        ].max()

        if maximum == minimum:

            normalized[
                column
            ] = 0.5

        else:

            normalized[
                column
            ] = (
                normalized[
                    column
                ]
                - minimum
            ) / (
                maximum
                - minimum
            )


    plt.figure(
        figsize=(12, 7)
    )

    image = plt.imshow(
        normalized.values,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label=(
            "Normalized value "
            "(0 = lowest, 1 = highest)"
        ),
    )

    plt.xticks(
        np.arange(
            len(
                normalized.columns
            )
        ),
        normalized.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        np.arange(
            len(
                normalized.index
            )
        ),
        normalized.index,
    )

    plt.title(
        "Normalized Linguistic Profiles "
        "Across Subreddits"
    )

    save_figure(
        "figure_08_subreddit_feature_heatmap.png"
    )


# ============================================================
# Figure 9
# Partial correlations controlling for word count
# ============================================================

if PARTIAL_RESULTS_PATH.exists():

    partial = pd.read_csv(
        PARTIAL_RESULTS_PATH
    )

    required_partial_columns = {
        "feature",
        "covariates",
        "r_partial",
    }

    if required_partial_columns.issubset(
        partial.columns
    ):

        length_controlled = (
            partial[
                partial["covariates"]
                == "word_count"
            ]
            .copy()
        )

        if not length_controlled.empty:

            length_controlled = (
                length_controlled
                .sort_values(
                    "r_partial"
                )
            )

            plt.figure(
                figsize=(9, 6)
            )

            y = np.arange(
                len(
                    length_controlled
                )
            )

            plt.barh(
                y,
                length_controlled[
                    "r_partial"
                ],
            )

            plt.yticks(
                y,
                length_controlled[
                    "feature"
                ],
            )

            plt.axvline(
                x=0,
                linewidth=1,
            )

            plt.xlabel(
                "Partial correlation with "
                "expertise tier"
            )

            plt.ylabel(
                "Linguistic feature"
            )

            plt.title(
                "Partial Correlations "
                "Controlling for Word Count"
            )

            save_figure(
                "figure_09_partial_correlations.png"
            )


# ============================================================
# Figure 10
# Random Forest feature importance
# ============================================================

if RF_IMPORTANCE_PATH.exists():

    importance = pd.read_csv(
        RF_IMPORTANCE_PATH
    )

    if {
        "feature",
        "importance",
    }.issubset(
        importance.columns
    ):

        importance = (
            importance
            .sort_values(
                "importance",
                ascending=True,
            )
        )

        plt.figure(
            figsize=(9, 6)
        )

        y = np.arange(
            len(
                importance
            )
        )

        plt.barh(
            y,
            importance[
                "importance"
            ],
        )

        plt.yticks(
            y,
            importance[
                "feature"
            ],
        )

        plt.xlabel(
            "Random Forest feature importance"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "Random Forest Feature Importance"
        )

        save_figure(
            "figure_10_random_forest_importance.png"
        )


# ============================================================
# Optional supplementary LOSO H-statistic figure
# ============================================================

if (
    loso_df is not None
    and not loso_df.empty
):

    if "metric" in loso_df.columns:

        syllable_loso = (
            loso_df[
                loso_df["metric"]
                == "avg_syllables_per_word"
            ]
            .copy()
        )

    else:

        syllable_loso = (
            loso_df.copy()
        )


    if (
        "kruskal_h"
        in syllable_loso.columns
        and
        "dropped_subreddit"
        in syllable_loso.columns
    ):

        syllable_loso = (
            syllable_loso
            .dropna(
                subset=[
                    "kruskal_h"
                ]
            )
        )

        plt.figure(
            figsize=(10, 5.5)
        )

        x = np.arange(
            len(
                syllable_loso
            )
        )

        plt.bar(
            x,
            syllable_loso[
                "kruskal_h"
            ],
        )

        labels = (
            syllable_loso[
                "dropped_subreddit"
            ]
            .astype(str)
            .replace(
                {
                    "FULL_DATASET":
                    "Full dataset"
                }
            )
        )

        plt.xticks(
            x,
            labels,
            rotation=45,
            ha="right",
        )

        plt.ylabel(
            "Kruskal-Wallis H"
        )

        plt.xlabel(
            "Subreddit removed"
        )

        plt.title(
            "Leave-One-Subreddit-Out "
            "Kruskal-Wallis Statistic"
        )

        save_figure(
            "supplementary_loso_kruskal_h.png"
        )


print(
    "\nFigure generation complete."
)

print(
    "Figures saved to:",
    FIGURES_DIR
)
