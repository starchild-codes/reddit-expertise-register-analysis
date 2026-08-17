from pathlib import Path

import pandas as pd
import pingouin as pg
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]

BASE_INPUT_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features.csv"
)

ACRONYM_INPUT_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features_acronyms.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "partial_correlations.csv"
)

TIER_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


# ============================================================
# Canonical analysis specification
# ============================================================

WORD_COUNT_FEATURES = [
    "fk_grade",
    "gunning_fog",
    "smog",
    "avg_syllables_per_word",
    "mattr",
    "hedge_per_100",
    "causal_per_100",
    "sentiment_compound",
    "sentiment_neg",
]

ACRONYM_CONTROL_FEATURE = (
    "avg_syllables_per_word"
)

ACRONYM_ROBUSTNESS_FEATURES = [
    "avg_syllables_per_word_without_acronyms",
    "avg_syllables_per_word_acronyms_normalized",
]


def extract_p_value(result):
    """
    Extract the p-value from a Pingouin partial_corr result
    while remaining compatible with minor column-name
    differences across Pingouin versions.
    """

    for column in result.columns:

        normalized = (
            column.lower()
            .replace("-", "")
            .replace("_", "")
        )

        if "pval" in normalized:
            return result[column].iloc[0]

    raise KeyError(
        "Could not identify the p-value column. "
        f"Returned columns: {result.columns.tolist()}"
    )


def run_partial_correlation(
    df,
    feature,
    covariates
):
    """
    Run a Pearson partial correlation between a linguistic
    feature and ordinal expertise tier while controlling
    for the specified covariates.
    """

    required_columns = [
        feature,
        "tier_numeric",
        *covariates,
    ]

    subset = (
        df[
            required_columns
        ]
        .dropna()
        .copy()
    )

    if len(subset) < 4:
        return None

    result = pg.partial_corr(
        data=subset,
        x=feature,
        y="tier_numeric",
        covar=covariates,
        method="pearson",
    )

    return {
        "feature": feature,
        "covariates": " + ".join(covariates),
        "n": len(subset),
        "r_partial": result["r"].iloc[0],
        "p_value": extract_p_value(result),
    }


def add_fdr_within_covariate_model(results):
    """
    Apply Benjamini-Hochberg FDR correction within each
    covariate specification.

    This reproduces the structure used in the committed
    partial_correlations.csv output.
    """

    results = results.copy()

    results[
        "fdr_q_within_covariate_model"
    ] = pd.NA

    for covariates, group in results.groupby(
        "covariates",
        dropna=False
    ):

        valid_mask = (
            group["p_value"]
            .notna()
        )

        if valid_mask.sum() == 0:
            continue

        _, q_values, _, _ = multipletests(
            group.loc[
                valid_mask,
                "p_value"
            ].astype(float),
            alpha=0.05,
            method="fdr_bh",
        )

        results.loc[
            group.loc[
                valid_mask
            ].index,
            "fdr_q_within_covariate_model"
        ] = q_values

    return results


def main():

    # ============================================================
    # Load acronym-enriched data when available
    # ============================================================

    if ACRONYM_INPUT_PATH.exists():

        input_path = (
            ACRONYM_INPUT_PATH
        )

    elif BASE_INPUT_PATH.exists():

        input_path = (
            BASE_INPUT_PATH
        )

    else:

        raise FileNotFoundError(
            "No feature dataset found.\n"
            "Run analysis/02_linguistic_features.py first."
        )


    df = pd.read_csv(
        input_path
    )


    # ============================================================
    # Validate tier labels
    # ============================================================

    if "expertise_tier" not in df.columns:

        raise ValueError(
            "Dataset is missing required column: "
            "expertise_tier"
        )


    df["expertise_tier"] = (
        df["expertise_tier"]
        .astype(str)
        .str.lower()
        .str.strip()
    )


    df["tier_numeric"] = (
        df["expertise_tier"]
        .map(
            TIER_MAP
        )
    )


    if (
        df["tier_numeric"]
        .isna()
        .any()
    ):

        unexpected = sorted(
            df.loc[
                df[
                    "tier_numeric"
                ].isna(),
                "expertise_tier"
            ]
            .dropna()
            .unique()
        )

        raise ValueError(
            "Unexpected expertise-tier labels: "
            f"{unexpected}"
        )


    if "word_count" not in df.columns:

        raise ValueError(
            "Dataset is missing required column: "
            "word_count"
        )


    rows = []


    print(
        f"Loaded {len(df):,} posts from "
        f"{input_path.name}"
    )


    # ============================================================
    # MODEL 1
    # Partial correlations controlling for word count
    # ============================================================

    print(
        "\nPartial correlations controlling "
        "for word count\n"
    )


    missing_word_count_features = [
        feature
        for feature
        in WORD_COUNT_FEATURES
        if feature
        not in df.columns
    ]


    if missing_word_count_features:

        raise ValueError(
            "Dataset is missing required word-count "
            "partial-correlation features: "
            f"{missing_word_count_features}"
        )


    for feature in WORD_COUNT_FEATURES:

        result = run_partial_correlation(
            df=df,
            feature=feature,
            covariates=[
                "word_count"
            ],
        )

        if result is None:
            continue

        rows.append(
            result
        )

        print(
            f"{feature:<40} "
            f"r={result['r_partial']:+.4f}  "
            f"p={result['p_value']:.6g}"
        )


    # ============================================================
    # MODEL 2
    # Main syllable result controlling for
    # word count + acronym density
    # ============================================================

    acronym_column = (
        "acronym_density_per_100_words"
    )


    if (
        ACRONYM_CONTROL_FEATURE
        in df.columns
        and
        acronym_column
        in df.columns
    ):

        print(
            "\nAverage syllables per word "
            "controlling for word count + "
            "acronym density\n"
        )

        result = run_partial_correlation(
            df=df,
            feature=ACRONYM_CONTROL_FEATURE,
            covariates=[
                "word_count",
                acronym_column,
            ],
        )

        if result is not None:

            rows.append(
                result
            )

            print(
                f"{ACRONYM_CONTROL_FEATURE:<40} "
                f"r={result['r_partial']:+.4f}  "
                f"p={result['p_value']:.6g}"
            )

    else:

        print(
            "\nAcronym-density columns not found. "
            "Skipping the word-count + acronym-density "
            "robustness model."
        )

        print(
            "Run analysis/07_acronym_robustness.py "
            "before rerunning this script."
        )


    # ============================================================
    # MODEL 3
    # Acronym-removal and acronym-normalization
    # partial correlations controlling for word count
    # ============================================================

    available_robustness_features = [
        feature
        for feature
        in ACRONYM_ROBUSTNESS_FEATURES
        if feature
        in df.columns
    ]


    if available_robustness_features:

        print(
            "\nAcronym-adjusted syllable measures "
            "controlling for word count\n"
        )


        for feature in available_robustness_features:

            result = run_partial_correlation(
                df=df,
                feature=feature,
                covariates=[
                    "word_count"
                ],
            )

            if result is None:
                continue

            rows.append(
                result
            )

            print(
                f"{feature:<40} "
                f"r={result['r_partial']:+.4f}  "
                f"p={result['p_value']:.6g}"
            )


    # ============================================================
    # Assemble output
    # ============================================================

    results = pd.DataFrame(
        rows
    )


    if results.empty:

        raise RuntimeError(
            "No partial correlations were generated."
        )


    results = add_fdr_within_covariate_model(
        results
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
        "\nSaved canonical partial-correlation results to:"
    )

    print(
        OUTPUT_PATH
    )


    print(
        f"\nRows written: {len(results)}"
    )


if __name__ == "__main__":
    main()
