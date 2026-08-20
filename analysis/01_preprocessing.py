from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_INPUT_PATH = ROOT / "data" / "reddit_posts_working.csv"
OUTPUT_PATH = ROOT / "data" / "reddit_posts_deidentified.csv"


COLUMN_RENAMES = {
    "text_clean": "clean_text",
    "expertise_tier": "expertise_tier",
    "subreddit": "subreddit",
    "created_utc": "created_utc",
    "score": "score",
    "num_comments": "num_comments",
    "word_count": "word_count",
}


DROP_COLUMNS = [
    # Raw text fields not needed in the public de-identified dataset
    "title",
    "text",

    # Historical duplicate readability columns
    "fk_grade.1",
    "gunning_fog.1",
    "smog.1",
    "avg_sentence_length.1",
    "avg_syllables_per_word.1",

    # Derived analysis fields that should be regenerated
    "fk_grade",
    "gunning_fog",
    "smog",
    "avg_sentence_length",
    "avg_syllables_per_word",
    "type_token_ratio",
    "mattr",
    "hedge_count",
    "causal_count",
    "hedge_per_100",
    "causal_per_100",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "sentiment_compound",
    "tier_numeric",
    "acronym_count",
    "acronym_density_per_100_words",
    "word_count_without_acronyms",
    "avg_syllables_per_word_without_acronyms",
    "avg_syllables_per_word_acronyms_normalized",
]


TIER_MAP = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


EXPECTED_SUBREDDITS = {
    "explainlikeimfive",
    "Futurology",
    "GenerativeAI",
    "ChatGPT",
    "learnmachinelearning",
    "OpenAI",
    "LocalLLaMA",
    "MachineLearning",
    "deeplearning",
}


def main():
    if not RAW_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Working dataset not found: {RAW_INPUT_PATH}\n"
            "Place the historical processed CSV at "
            "data/reddit_posts_working.csv before running this script."
        )

    df = pd.read_csv(RAW_INPUT_PATH)

    print(f"Loaded {len(df):,} rows")
    print(f"Loaded {len(df.columns)} columns")

    required_source_columns = {
        "subreddit",
        "expertise_tier",
        "created_utc",
        "score",
        "num_comments",
        "word_count",
        "text_clean",
    }

    missing = required_source_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Working dataset is missing required source columns: "
            f"{sorted(missing)}"
        )

    # Keep only canonical source variables.

    columns_to_drop = [
        column
        for column in DROP_COLUMNS
        if column in df.columns
    ]

    df = df.drop(
        columns=columns_to_drop
    )

    df = df.rename(
        columns=COLUMN_RENAMES
    )

    canonical_columns = [
        "subreddit",
        "expertise_tier",
        "created_utc",
        "score",
        "num_comments",
        "word_count",
        "clean_text",
    ]

    df = df[
        canonical_columns
    ].copy()

    # Basic cleaning.

    df["subreddit"] = (
        df["subreddit"]
        .astype(str)
        .str.strip()
    )

    df["expertise_tier"] = (
        df["expertise_tier"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["clean_text"] = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    for column in [
        "created_utc",
        "score",
        "num_comments",
        "word_count",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove invalid rows.

    before = len(df)

    df = df.dropna(
        subset=[
            "subreddit",
            "expertise_tier",
            "created_utc",
            "word_count",
        ]
    )

    df = df[
        df["clean_text"].str.len() > 0
    ]

    df = df[
        df["expertise_tier"].isin(
            TIER_MAP.keys()
        )
    ]

    removed = before - len(df)

    print(
        f"Removed {removed:,} rows with missing or invalid canonical data"
    )

    # Validation.

    unexpected_subreddits = (
        set(df["subreddit"].unique())
        - EXPECTED_SUBREDDITS
    )

    if unexpected_subreddits:
        raise ValueError(
            "Unexpected subreddit labels found: "
            f"{sorted(unexpected_subreddits)}"
        )

    if df.duplicated().any():
        duplicate_count = int(
            df.duplicated().sum()
        )

        print(
            f"Warning: {duplicate_count:,} fully duplicated rows found"
        )

    text_duplicates = int(
        df["clean_text"].duplicated().sum()
    )

    if text_duplicates > 0:
        print(
            f"Warning: {text_duplicates:,} duplicate clean_text values found"
        )

    # Add the public tier code.

    df["tier_code"] = (
        df["expertise_tier"]
        .map(TIER_MAP)
    )

    output_columns = [
        "subreddit",
        "expertise_tier",
        "tier_code",
        "created_utc",
        "score",
        "num_comments",
        "word_count",
        "clean_text",
    ]

    df = df[
        output_columns
    ]

    # Summary checks.

    print("\nTier counts:")
    print(
        df["expertise_tier"]
        .value_counts()
        .reindex(
            ["low", "medium", "high"]
        )
    )

    print("\nSubreddit counts:")
    print(
        df["subreddit"]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nFinal row count: {len(df):,}"
    )

    # These are the expected counts reported in the manuscript.
    expected_total = 1778

    expected_tier_counts = {
        "low": 598,
        "medium": 592,
        "high": 588,
    }

    if len(df) != expected_total:
        print(
            f"\nWARNING: Expected {expected_total:,} rows "
            f"but found {len(df):,}."
        )

    observed_tier_counts = (
        df["expertise_tier"]
        .value_counts()
        .to_dict()
    )

    for tier, expected_count in expected_tier_counts.items():
        observed = observed_tier_counts.get(
            tier,
            0
        )

        if observed != expected_count:
            print(
                f"WARNING: {tier} tier expected "
                f"{expected_count} rows but found {observed}."
            )

    # Save the cleaned data.

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved canonical dataset to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
