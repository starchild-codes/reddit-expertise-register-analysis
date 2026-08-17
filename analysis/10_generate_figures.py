from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features.csv"
)

ACRONYM_DATA_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features_acronyms.csv"
)

PRIMARY_RESULTS_PATH = (
    ROOT
    / "results"
    / "primary_statistics.csv"
)

PARTIAL_RESULTS_PATH = (
    ROOT
    / "results"
    / "partial_correlations.csv"
)

IMPORTANCE_PATH = (
    ROOT
    / "results"
    / "random_forest_feature_importance.csv"
)

LOSO_PATH = (
    ROOT
    / "results"
    / "leave_one_subreddit_out.csv"
)

FIGURE_DIR = (
    ROOT
    / "figures"
)


TIER_ORDER = [
    "low",
    "medium",
    "high",
]


def ensure_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def save_figure(fig, filename):
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FIGURE_DIR
        / filename
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {output_path}"
    )


def get_primary_row(
    primary_results,
    feature
):
    row = primary_results[
        primary_results["feature"]
        == feature
    ]

    if row.empty:
        return None

    return row.iloc[0]


def figure_01_syllables_by_tier(
    df,
    primary_results
):
    feature = (
        "avg_syllables_per_word"
    )

    means = (
        df.groupby(
            "expertise_tier"
        )[feature]
        .mean()
        .reindex(
            TIER_ORDER
        )
    )

    sems = (
        df.groupby(
            "expertise_tier"
        )[feature]
        .sem()
        .reindex(
            TIER_ORDER
        )
    )

    result = get_primary_row(
        primary_results,
        feature
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        TIER_ORDER,
        means.values,
        yerr=sems.values,
        capsize=5,
    )

    for bar, value in zip(
        bars,
        means.values
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel(
        "Expected-audience expertise tier"
    )

    ax.set_ylabel(
        "Mean syllables per word"
    )

    if result is not None:
        ax.set_title(
            "Syllable-based lexical complexity by expertise tier\n"
            f"H={result['kruskal_h']:.2f}, "
            f"p={result['p_value']:.3g}, "
            f"d={result['cohens_d_high_vs_low']:.3f}"
        )
    else:
        ax.set_title(
            "Syllable-based lexical complexity by expertise tier"
        )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_01_syllables_by_tier.png"
    )


def figure_02_fk_boxplot(
    df,
    primary_results
):
    feature = "fk_grade"

    data = [
        df.loc[
            df[
                "expertise_tier"
            ]
            == tier,
            feature
        ].dropna()
        for tier
        in TIER_ORDER
    ]

    result = get_primary_row(
        primary_results,
        feature
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        data,
        labels=[
            "Low",
            "Medium",
            "High"
        ],
        showfliers=False,
    )

    ax.set_xlabel(
        "Expected-audience expertise tier"
    )

    ax.set_ylabel(
        "Flesch-Kincaid grade level"
    )

    if result is not None:
        ax.set_title(
            "Readability grade level by expertise tier\n"
            f"H={result['kruskal_h']:.2f}, "
            f"p={result['p_value']:.3g}"
        )
    else:
        ax.set_title(
            "Readability grade level by expertise tier"
        )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_02_fk_grade_boxplot.png"
    )


def figure_03_word_count(
    df,
    primary_results
):
    feature = "word_count"

    means = (
        df.groupby(
            "expertise_tier"
        )[feature]
        .mean()
        .reindex(
            TIER_ORDER
        )
    )

    sems = (
        df.groupby(
            "expertise_tier"
        )[feature]
        .sem()
        .reindex(
            TIER_ORDER
        )
    )

    result = get_primary_row(
        primary_results,
        feature
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        TIER_ORDER,
        means.values,
        yerr=sems.values,
        capsize=5,
    )

    for bar, value in zip(
        bars,
        means.values
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 3,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel(
        "Expected-audience expertise tier"
    )

    ax.set_ylabel(
        "Mean post length (words)"
    )

    if result is not None:
        ax.set_title(
            "Post length by expertise tier\n"
            f"H={result['kruskal_h']:.2f}, "
            f"p={result['p_value']:.3g}, "
            f"d={result['cohens_d_high_vs_low']:.3f}"
        )
    else:
        ax.set_title(
            "Post length by expertise tier"
        )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_03_word_count_by_tier.png"
    )


def figure_04_subreddit_heatmap(
    df
):
    metrics = [
        "avg_syllables_per_word",
        "fk_grade",
        "hedge_per_100",
        "causal_per_100",
        "mattr",
        "word_count",
        "sentiment_compound",
    ]

    labels = [
        "Syllables/word",
        "FK grade",
        "Hedging/100",
        "Causal/100",
        "MATTR",
        "Word count",
        "Sentiment",
    ]

    subset = (
        df.groupby(
            "subreddit"
        )[metrics]
        .mean()
    )

    tier_lookup = (
        df[
            [
                "subreddit",
                "expertise_tier",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "subreddit"
        )[
            "expertise_tier"
        ]
    )

    order = []

    for tier in TIER_ORDER:
        tier_subreddits = sorted(
            tier_lookup[
                tier_lookup == tier
            ]
            .index
            .tolist()
        )

        order.extend(
            tier_subreddits
        )

    subset = subset.reindex(
        order
    )

    denominator = (
        subset.max()
        - subset.min()
    )

    denominator = (
        denominator
        .replace(
            0,
            np.nan
        )
    )

    normalized = (
        subset
        - subset.min()
    ) / denominator

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    image = ax.imshow(
        normalized.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        range(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(
                normalized.index
            )
        )
    )

    ax.set_yticklabels(
        normalized.index
    )

    ax.set_title(
        "Normalized linguistic profiles across subreddits"
    )

    fig.colorbar(
        image,
        ax=ax,
        label=(
            "Within-feature normalized mean "
            "(0 = lowest, 1 = highest)"
        ),
    )

    save_figure(
        fig,
        "figure_04_subreddit_heatmap.png"
    )


def figure_05_rf_importance(
    importance_df
):
    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.barh(
        importance_df[
            "feature"
        ],
        importance_df[
            "importance"
        ],
    )

    ax.set_xlabel(
        "Random Forest feature importance"
    )

    ax.set_title(
        "Feature importance for expertise-tier classification"
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_05_random_forest_importance.png"
    )


def figure_06_partial_correlations(
    partial_df
):
    subset = partial_df[
        partial_df[
            "covariates"
        ]
        == "word_count"
    ].copy()

    if subset.empty:
        print(
            "Skipping partial-correlation figure: "
            "word_count model not found."
        )
        return

    subset = subset.sort_values(
        "r_partial"
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        subset[
            "feature"
        ],
        subset[
            "r_partial"
        ],
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Partial correlation with expertise tier\n"
        "(controlling for word count)"
    )

    ax.set_title(
        "Length-adjusted associations with expertise tier"
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_06_partial_correlations.png"
    )


def figure_07_sentiment_by_subreddit(
    df
):
    sentiment = (
        df.groupby(
            "subreddit"
        )[
            "sentiment_compound"
        ]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        sentiment.index,
        sentiment.values,
    )

    ax.axvline(
        0,
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Mean VADER compound sentiment"
    )

    ax.set_title(
        "Compound sentiment by subreddit"
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_07_sentiment_by_subreddit.png"
    )


def figure_08_acronym_robustness(
    acronym_df
):
    metrics = [
        (
            "avg_syllables_per_word",
            "Original"
        ),
        (
            "avg_syllables_per_word_without_acronyms",
            "Acronyms removed"
        ),
        (
            "avg_syllables_per_word_acronyms_normalized",
            "Acronyms normalized"
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    x = np.arange(
        len(
            TIER_ORDER
        )
    )

    width = 0.25

    for index, (
        metric,
        label
    ) in enumerate(metrics):

        means = (
            acronym_df
            .groupby(
                "expertise_tier"
            )[metric]
            .mean()
            .reindex(
                TIER_ORDER
            )
        )

        ax.bar(
            x
            + (
                index - 1
            )
            * width,
            means.values,
            width,
            label=label,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    ax.set_ylabel(
        "Mean syllables per word"
    )

    ax.set_xlabel(
        "Expected-audience expertise tier"
    )

    ax.set_title(
        "Acronym robustness of syllable-based lexical complexity"
    )

    ax.legend()

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_08_acronym_robustness.png"
    )


def figure_09_loso_effect_size(
    loso_df
):
    subset = loso_df[
        loso_df[
            "metric"
        ]
        == "avg_syllables_per_word"
    ].copy()

    if subset.empty:
        print(
            "Skipping LOSO effect-size figure."
        )
        return

    full = subset[
        subset[
            "dropped_subreddit"
        ]
        == "FULL_DATASET"
    ]

    loo = subset[
        subset[
            "dropped_subreddit"
        ]
        != "FULL_DATASET"
    ].copy()

    loo = loo.sort_values(
        "dropped_subreddit"
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        loo[
            "dropped_subreddit"
        ],
        loo[
            "d_high_vs_low"
        ],
        marker="o",
    )

    if not full.empty:
        baseline = (
            full[
                "d_high_vs_low"
            ]
            .iloc[0]
        )

        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1,
            label=(
                f"Full dataset d={baseline:.3f}"
            ),
        )

        ax.legend()

    ax.set_ylabel(
        "High-vs-low Cohen's d"
    )

    ax.set_xlabel(
        "Subreddit omitted"
    )

    ax.set_title(
        "Leave-one-subreddit-out stability of the primary effect"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_09_loso_effect_size.png"
    )


def figure_10_loso_h_statistic(
    loso_df
):
    subset = loso_df[
        loso_df[
            "metric"
        ]
        == "avg_syllables_per_word"
    ].copy()

    if subset.empty:
        print(
            "Skipping LOSO H-statistic figure."
        )
        return

    full = subset[
        subset[
            "dropped_subreddit"
        ]
        == "FULL_DATASET"
    ]

    loo = subset[
        subset[
            "dropped_subreddit"
        ]
        != "FULL_DATASET"
    ].copy()

    loo = loo.sort_values(
        "dropped_subreddit"
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        loo[
            "dropped_subreddit"
        ],
        loo[
            "kruskal_h"
        ],
        marker="o",
    )

    if not full.empty:
        baseline = (
            full[
                "kruskal_h"
            ]
            .iloc[0]
        )

        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1,
            label=(
                f"Full dataset H={baseline:.2f}"
            ),
        )

        ax.legend()

    ax.set_ylabel(
        "Kruskal-Wallis H"
    )

    ax.set_xlabel(
        "Subreddit omitted"
    )

    ax.set_title(
        "Leave-one-subreddit-out stability of the primary omnibus statistic"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    save_figure(
        fig,
        "figure_10_loso_h_statistic.png"
    )


def main():
    ensure_file(
        DATA_PATH
    )

    ensure_file(
        PRIMARY_RESULTS_PATH
    )

    ensure_file(
        PARTIAL_RESULTS_PATH
    )

    ensure_file(
        IMPORTANCE_PATH
    )

    ensure_file(
        LOSO_PATH
    )

    df = pd.read_csv(
        DATA_PATH
    )

    primary_results = pd.read_csv(
        PRIMARY_RESULTS_PATH
    )

    partial_results = pd.read_csv(
        PARTIAL_RESULTS_PATH
    )

    importance = pd.read_csv(
        IMPORTANCE_PATH
    )

    loso = pd.read_csv(
        LOSO_PATH
    )

    print(
        f"Loaded {len(df):,} posts"
    )

    figure_01_syllables_by_tier(
        df,
        primary_results
    )

    figure_02_fk_boxplot(
        df,
        primary_results
    )

    figure_03_word_count(
        df,
        primary_results
    )

    figure_04_subreddit_heatmap(
        df
    )

    figure_05_rf_importance(
        importance
    )

    figure_06_partial_correlations(
        partial_results
    )

    figure_07_sentiment_by_subreddit(
        df
    )

    if ACRONYM_DATA_PATH.exists():
        acronym_df = pd.read_csv(
            ACRONYM_DATA_PATH
        )

        figure_08_acronym_robustness(
            acronym_df
        )

    else:
        print(
            "Skipping acronym robustness figure: "
            "run analysis/07_acronym_robustness.py first."
        )

    figure_09_loso_effect_size(
        loso
    )

    figure_10_loso_h_statistic(
        loso
    )

    print(
        "\nFigure generation complete."
    )


if __name__ == "__main__":
    main()
