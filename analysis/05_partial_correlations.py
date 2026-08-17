from pathlib import Path

import pandas as pd
import pingouin as pg


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "reddit_posts_features.csv"
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
    for column in result.columns:
        normalized = column.lower().replace("-", "").replace("_", "")
        if "pval" in normalized:
            return result[column].iloc[0]

    raise KeyError(
        f"Could not identify p-value column. "
        f"Returned columns: {result.columns.tolist()}"
    )


def run_partial_correlation(df, feature, covariates):
    columns = [feature, "tier_numeric"] + covariates

    subset = df[columns].dropna()

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
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}\n"
            "Run analysis/02_linguistic_features.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "expertise_tier",
        "word_count",
    } | set(FEATURES)

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df["tier_numeric"] = df["expertise_tier"].map(TIER_MAP)

    if df["tier_numeric"].isna().any():
        bad_labels = sorted(
            df.loc[
                df["tier_numeric"].isna(),
                "expertise_tier"
            ].dropna().unique()
        )

        raise ValueError(
            f"Unexpected expertise-tier labels found: {bad_labels}"
        )

    rows = []

    print(f"Loaded {len(df):,} posts")
    print("\nPartial correlations controlling for word count\n")

    for feature in FEATURES:
        result = run_partial_correlation(
            df=df,
            feature=feature,
            covariates=["word_count"],
        )

        if result is not None:
            rows.append(result)

            print(
                f"{feature:<28} "
                f"r={result['r_partial']:+.4f}  "
                f"p={result['p_value']:.6g}"
            )

    # Optional second robustness model.
    # This becomes active once acronym_density_per_100 exists.
    acronym_column = "acronym_density_per_100"

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

            if result is not None:
                rows.append(result)

                print(
                    f"{feature:<28} "
                    f"r={result['r_partial']:+.4f}  "
                    f"p={result['p_value']:.6g}"
                )
    else:
        print(
            "\nAcronym-density column not found. "
            "Skipping two-covariate robustness analysis."
        )

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
