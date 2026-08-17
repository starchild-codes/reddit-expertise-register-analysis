from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

BASE_FEATURES_PATH = DATA_DIR / "reddit_posts_features.csv"
ACRONYM_FEATURES_PATH = DATA_DIR / "reddit_posts_features_acronyms.csv"

OUTPUT_PATH = RESULTS_DIR / "leave_one_subreddit_out_all_metrics.csv"
PARTIAL_OUTPUT_PATH = RESULTS_DIR / "leave_one_subreddit_out_partial_correlations.csv"
INFLUENCE_OUTPUT_PATH = RESULTS_DIR / "leave_one_subreddit_out_influential_subreddits.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

TIER_ORDER = ["low", "medium", "high"]

TIER_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

PRIMARY_METRICS = [
    "avg_syllables_per_word",
]

ACRONYM_METRICS = [
    "avg_syllables_per_word_without_acronyms",
    "avg_syllables_per_word_acronyms_normalized",
    "acronym_density_per_100_words",
]

SENTIMENT_METRICS = [
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


# ============================================================
# Helpers
# ============================================================

def cohens_d(group_a, group_b):
    """
    Cohen's d using the historical study convention:
    square root of the average of the two sample variances.

    This matches the effect-size calculation used in the manuscript.
    """
    a = pd.Series(group_a).dropna().astype(float)
    b = pd.Series(group_b).dropna().astype(float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled_sd = np.sqrt(
        (a.var(ddof=1) + b.var(ddof=1)) / 2
    )

    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan

    return (a.mean() - b.mean()) / pooled_sd


def partial_correlation(df, x_col, y_col, covariates):
    """
    Pearson partial correlation using residualization.

    x_col:
        linguistic variable

    y_col:
        ordinal tier code

    covariates:
        list of variables to control
    """
    required = [x_col, y_col] + covariates
    sub = df[required].dropna().astype(float).copy()

    n = len(sub)

    if n <= len(covariates) + 2:
        return {
            "n": n,
            "r_partial": np.nan,
            "p_value": np.nan,
        }

    X = np.column_stack(
        [np.ones(n)] +
        [sub[c].to_numpy() for c in covariates]
    )

    x = sub[x_col].to_numpy()
    y = sub[y_col].to_numpy()

    beta_x = np.linalg.lstsq(X, x, rcond=None)[0]
    beta_y = np.linalg.lstsq(X, y, rcond=None)[0]

    residual_x = x - X @ beta_x
    residual_y = y - X @ beta_y

    r = np.corrcoef(residual_x, residual_y)[0, 1]

    if np.isnan(r):
        p = np.nan
    elif abs(r) >= 1:
        p = 0.0
    else:
        k = len(covariates)
        dfree = n - k - 2

        t_stat = r * np.sqrt(
            dfree / (1 - r ** 2)
        )

        p = 2 * stats.t.sf(
            abs(t_stat),
            df=dfree
        )

    return {
        "n": n,
        "r_partial": r,
        "p_value": p,
    }


def run_metric_analysis(df, metric, dropped_subreddit):
    """
    Run Kruskal-Wallis, Dunn post hoc tests,
    descriptive statistics, and effect sizes
    for one metric in one LOSO iteration.
    """

    clean = df[
        ["expertise_tier", metric]
    ].dropna().copy()

    tier_values = {}

    for tier in TIER_ORDER:
        tier_values[tier] = clean.loc[
            clean["expertise_tier"] == tier,
            metric
        ].astype(float)

    if any(len(tier_values[t]) == 0 for t in TIER_ORDER):
        return None

    H, p_kw = stats.kruskal(
        tier_values["low"],
        tier_values["medium"],
        tier_values["high"],
    )

    dunn = sp.posthoc_dunn(
        clean,
        val_col=metric,
        group_col="expertise_tier",
        p_adjust="bonferroni",
    )

    low_mean = tier_values["low"].mean()
    medium_mean = tier_values["medium"].mean()
    high_mean = tier_values["high"].mean()

    d_high_low = cohens_d(
        tier_values["high"],
        tier_values["low"],
    )

    d_high_medium = cohens_d(
        tier_values["high"],
        tier_values["medium"],
    )

    high_above_both = (
        high_mean > low_mean
        and high_mean > medium_mean
    )

    high_low_sig = (
        dunn.loc["high", "low"] < 0.05
    )

    high_medium_sig = (
        dunn.loc["high", "medium"] < 0.05
    )

    pattern_preserved = (
        high_above_both
        and p_kw < 0.05
        and high_low_sig
        and high_medium_sig
    )

    return {
        "dropped_subreddit": dropped_subreddit,
        "metric": metric,

        "n_total": len(clean),
        "n_low": len(tier_values["low"]),
        "n_medium": len(tier_values["medium"]),
        "n_high": len(tier_values["high"]),

        "low_mean": low_mean,
        "medium_mean": medium_mean,
        "high_mean": high_mean,

        "low_sd": tier_values["low"].std(ddof=1),
        "medium_sd": tier_values["medium"].std(ddof=1),
        "high_sd": tier_values["high"].std(ddof=1),

        "kruskal_h": H,
        "kruskal_p": p_kw,

        "dunn_low_vs_medium_p": dunn.loc["low", "medium"],
        "dunn_low_vs_high_p": dunn.loc["low", "high"],
        "dunn_medium_vs_high_p": dunn.loc["medium", "high"],

        "cohens_d_high_vs_low": d_high_low,
        "cohens_d_high_vs_medium": d_high_medium,

        "high_above_both": high_above_both,
        "pattern_preserved": pattern_preserved,
    }


# ============================================================
# Load data
# ============================================================

if ACRONYM_FEATURES_PATH.exists():
    input_path = ACRONYM_FEATURES_PATH
    print(
        "Using acronym-enriched feature file:",
        input_path
    )

elif BASE_FEATURES_PATH.exists():
    input_path = BASE_FEATURES_PATH
    print(
        "Acronym-enriched file not found.",
        "Using base feature file:",
        input_path
    )

else:
    raise FileNotFoundError(
        "Could not find either:\n"
        f"{ACRONYM_FEATURES_PATH}\n"
        "or\n"
        f"{BASE_FEATURES_PATH}"
    )


df = pd.read_csv(input_path)


# ============================================================
# Validate essential columns
# ============================================================

required_columns = [
    "subreddit",
    "expertise_tier",
    "word_count",
]

missing_required = [
    c for c in required_columns
    if c not in df.columns
]

if missing_required:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_required)
    )


df["expertise_tier"] = (
    df["expertise_tier"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["tier_numeric"] = df["expertise_tier"].map(
    TIER_MAP
)

if df["tier_numeric"].isna().any():
    bad = sorted(
        df.loc[
            df["tier_numeric"].isna(),
            "expertise_tier"
        ].unique()
    )

    raise ValueError(
        f"Unrecognized expertise tiers: {bad}"
    )


# ============================================================
# Select metrics actually available
# ============================================================

candidate_metrics = (
    PRIMARY_METRICS
    + ACRONYM_METRICS
    + SENTIMENT_METRICS
    + EXPLORATORY_METRICS
)

available_metrics = [
    metric
    for metric in candidate_metrics
    if metric in df.columns
]

missing_metrics = [
    metric
    for metric in candidate_metrics
    if metric not in df.columns
]

if missing_metrics:
    print(
        "\nSkipping unavailable metrics:"
    )

    for metric in missing_metrics:
        print(" -", metric)


print(
    "\nMetrics included in LOSO analysis:"
)

for metric in available_metrics:
    print(" -", metric)


# ============================================================
# Full-data baseline + leave-one-subreddit-out analyses
# ============================================================

subreddits = sorted(
    df["subreddit"]
    .dropna()
    .astype(str)
    .unique()
)

analysis_iterations = [
    ("FULL_DATASET", df)
]

for subreddit in subreddits:
    subset = df[
        df["subreddit"] != subreddit
    ].copy()

    analysis_iterations.append(
        (subreddit, subset)
    )


all_results = []

for dropped_subreddit, subset in analysis_iterations:

    print(
        f"\nRunning iteration: {dropped_subreddit}"
    )

    for metric in available_metrics:

        result = run_metric_analysis(
            subset,
            metric,
            dropped_subreddit,
        )

        if result is not None:
            all_results.append(result)


results_df = pd.DataFrame(all_results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(
    "\nSaved full LOSO results to:"
)
print(OUTPUT_PATH)


# ============================================================
# Leave-one-subreddit-out partial correlations
# ============================================================

partial_rows = []

for dropped_subreddit, subset in analysis_iterations:

    # --------------------------------------------------------
    # Main syllable result controlling for post length
    # --------------------------------------------------------

    if "avg_syllables_per_word" in subset.columns:

        result = partial_correlation(
            subset,
            x_col="avg_syllables_per_word",
            y_col="tier_numeric",
            covariates=["word_count"],
        )

        partial_rows.append({
            "dropped_subreddit": dropped_subreddit,
            "metric": "avg_syllables_per_word",
            "covariates": "word_count",
            **result,
        })

    # --------------------------------------------------------
    # Main syllable result controlling for length + acronym density
    # --------------------------------------------------------

    if (
        "avg_syllables_per_word" in subset.columns
        and
        "acronym_density_per_100_words" in subset.columns
    ):

        result = partial_correlation(
            subset,
            x_col="avg_syllables_per_word",
            y_col="tier_numeric",
            covariates=[
                "word_count",
                "acronym_density_per_100_words",
            ],
        )

        partial_rows.append({
            "dropped_subreddit": dropped_subreddit,
            "metric": "avg_syllables_per_word",
            "covariates": (
                "word_count + "
                "acronym_density_per_100_words"
            ),
            **result,
        })

    # --------------------------------------------------------
    # Acronym-removed syllables
    # --------------------------------------------------------

    if (
        "avg_syllables_per_word_without_acronyms"
        in subset.columns
    ):

        result = partial_correlation(
            subset,
            x_col=(
                "avg_syllables_per_word_without_acronyms"
            ),
            y_col="tier_numeric",
            covariates=["word_count"],
        )

        partial_rows.append({
            "dropped_subreddit": dropped_subreddit,
            "metric": (
                "avg_syllables_per_word_without_acronyms"
            ),
            "covariates": "word_count",
            **result,
        })

    # --------------------------------------------------------
    # Acronym-normalized syllables
    # --------------------------------------------------------

    if (
        "avg_syllables_per_word_acronyms_normalized"
        in subset.columns
    ):

        result = partial_correlation(
            subset,
            x_col=(
                "avg_syllables_per_word_acronyms_normalized"
            ),
            y_col="tier_numeric",
            covariates=["word_count"],
        )

        partial_rows.append({
            "dropped_subreddit": dropped_subreddit,
            "metric": (
                "avg_syllables_per_word_acronyms_normalized"
            ),
            "covariates": "word_count",
            **result,
        })

    # --------------------------------------------------------
    # Sentiment robustness
    # --------------------------------------------------------

    for sentiment_metric in [
        "sentiment_compound",
        "sentiment_neg",
    ]:

        if sentiment_metric in subset.columns:

            result = partial_correlation(
                subset,
                x_col=sentiment_metric,
                y_col="tier_numeric",
                covariates=["word_count"],
            )

            partial_rows.append({
                "dropped_subreddit": dropped_subreddit,
                "metric": sentiment_metric,
                "covariates": "word_count",
                **result,
            })


partial_df = pd.DataFrame(partial_rows)

partial_df.to_csv(
    PARTIAL_OUTPUT_PATH,
    index=False,
)

print(
    "\nSaved LOSO partial correlations to:"
)
print(PARTIAL_OUTPUT_PATH)


# ============================================================
# Identify most influential subreddit for each metric
# ============================================================

influence_rows = []

if not results_df.empty:

    baseline = results_df[
        results_df["dropped_subreddit"]
        == "FULL_DATASET"
    ].copy()

    loso_only = results_df[
        results_df["dropped_subreddit"]
        != "FULL_DATASET"
    ].copy()

    for metric in available_metrics:

        baseline_metric = baseline[
            baseline["metric"] == metric
        ]

        metric_runs = loso_only[
            loso_only["metric"] == metric
        ].copy()

        if (
            baseline_metric.empty
            or metric_runs.empty
        ):
            continue

        baseline_h = (
            baseline_metric["kruskal_h"]
            .iloc[0]
        )

        baseline_d = (
            baseline_metric[
                "cohens_d_high_vs_low"
            ]
            .iloc[0]
        )

        metric_runs[
            "change_in_h"
        ] = (
            metric_runs["kruskal_h"]
            - baseline_h
        )

        metric_runs[
            "change_in_d_high_vs_low"
        ] = (
            metric_runs[
                "cohens_d_high_vs_low"
            ]
            - baseline_d
        )

        most_h = metric_runs.loc[
            metric_runs["change_in_h"].idxmin()
        ]

        most_d = metric_runs.loc[
            metric_runs[
                "change_in_d_high_vs_low"
            ].idxmin()
        ]

        influence_rows.append({
            "metric": metric,

            "baseline_kruskal_h": baseline_h,
            "baseline_d_high_vs_low": baseline_d,

            "largest_h_reduction_subreddit":
                most_h["dropped_subreddit"],

            "kruskal_h_after_removal":
                most_h["kruskal_h"],

            "change_in_h":
                most_h["change_in_h"],

            "largest_d_reduction_subreddit":
                most_d["dropped_subreddit"],

            "d_high_vs_low_after_removal":
                most_d[
                    "cohens_d_high_vs_low"
                ],

            "change_in_d_high_vs_low":
                most_d[
                    "change_in_d_high_vs_low"
                ],
        })


influence_df = pd.DataFrame(
    influence_rows
)

influence_df.to_csv(
    INFLUENCE_OUTPUT_PATH,
    index=False,
)

print(
    "\nSaved influential-subreddit summary to:"
)
print(INFLUENCE_OUTPUT_PATH)


# ============================================================
# Compact console summary for main lexical result
# ============================================================

main = results_df[
    results_df["metric"]
    == "avg_syllables_per_word"
].copy()

if not main.empty:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "AVERAGE SYLLABLES PER WORD — "
        "LOSO SUMMARY"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "dropped_subreddit",
        "low_mean",
        "medium_mean",
        "high_mean",
        "kruskal_h",
        "kruskal_p",
        "cohens_d_high_vs_low",
        "pattern_preserved",
    ]

    print(
        main[display_columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


print(
    "\nLOSO analysis complete."
)
