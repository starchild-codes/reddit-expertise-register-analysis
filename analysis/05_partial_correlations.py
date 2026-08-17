from pathlib import Path

import pandas as pd
import pingouin as pg


ROOT = Path(__file__).resolve().parents[1]

BASE_INPUT_PATH = ROOT / "data" / "reddit_posts_features.csv"
ACRONYM_INPUT_PATH = ROOT / "data" / "reddit_posts_features_acronyms.csv"

OUTPUT_PATH = ROOT / "results" / "partial_correlations.csv"

TIER_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

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
]


def extract_p_value(result):
    """
    Extract the p-value from a Pingouin partial_corr result
    while remaining compatible with minor column-name differences
    across Pingouin versions.
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
    required_columns = [
        feature,
        "tier_numeric",
        *covariates,
    ]

    subset = df[
        required_columns
    ].dropna()

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


def main():
    # Prefer the acronym-enriched dataset when available.
    # This allows both the standard word-count control and the
    # word-count + acronym-density robustness model to be produced
    # from a single run.
    if ACRONYM_INPUT_PATH.exists():
        input_path = ACRONYM_INPUT_PATH
    elif BASE_INPUT_PATH.exists():
        input_path = BASE_INPUT_PATH
    else:
        raise FileNotFoundError(
            "No feature dataset found.\n"
            "Run analysis/02_linguistic_features.py first."
        )

    df = pd.read_csv(
        input_path
    )

    required = {
        "expertise_tier",
        "word_count",
    } | set(FEATURES)

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    df["tier_numeric"] = (
        df["expertise_tier"]
        .map(TIER_MAP)
    )

    if df["tier_numeric"].isna().any():
        unexpected = sorted(
            df.loc[
                df["tier_numeric"].isna(),
                "expertise_tier"
            ]
            .dropna()
            .unique()
        )

        raise ValueError(
            "Unexpected expertise-tier labels: "
            f"{unexpected}"
        )

    rows = []

    print(
        f"Loaded {len(df):,} posts from "
        f"{input_path.name}"
    )

    # ============================================================
    # MODEL 1 — CONTROL FOR WORD COUNT
    # ============================================================

    print(
        "\nPartial correlations controlling for word count\n"
    )

    for feature in FEATURES:
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
            f"{feature:<30} "
            f"r={result['r_partial']:+.4f}  "
            f"p={result['p_value']:.6g}"
        )

    # ============================================================
    # MODEL 2 — CONTROL FOR WORD COUNT + ACRONYM DENSITY
    # ============================================================

    acronym_column = (
        "acronym_density_per_100_words"
    )

    if acronym_column in df.columns:
        print(
            "\nPartial correlations controlling for "
            "word count + acronym density\n"
        )

        for feature in FEATURES:
            result = run_partial_correlation(
                df=df,
                feature=feature,
                covariates=[
                    "word_count",
                    acronym_column,
                ],
            )

            if result is None:
                continue

            rows.append(
                result
            )

            print(
                f"{feature:<30} "
                f"r={result['r_partial']:+.4f}  "
                f"p={result['p_value']:.6g}"
            )

    else:
        print(
            "\nAcronym-density column not found. "
            "Skipping the word-count + acronym-density model."
        )
        print(
            "Run analysis/07_acronym_robustness.py "
            "and rerun this script."
        )

    results = pd.DataFrame(
        rows
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
        f"\nSaved results to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
