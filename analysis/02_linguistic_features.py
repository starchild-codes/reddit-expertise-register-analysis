from pathlib import Path
import re

import pandas as pd
import textstat
from lexicalrichness import LexicalRichness
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "reddit_posts_deidentified.csv"
OUTPUT_PATH = ROOT / "data" / "reddit_posts_features.csv"


HEDGE_WORDS = [
    "maybe", "perhaps", "might", "could", "possibly", "likely",
    "probably", "seemingly", "apparently", "presumably", "suggest",
    "appear", "seem", "tend", "generally", "usually", "sometimes",
    "often", "roughly", "approximately"
]

CAUSAL_WORDS = [
    "because", "therefore", "since", "thus", "hence", "consequently",
    "as a result", "due to", "which means", "this means", "so that",
    "in order to", "leads to", "caused by", "results in"
]


def compute_readability(text: str):
    text = str(text)

    try:
        return {
            "fk_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
            "smog": textstat.smog_index(text),
            "avg_sentence_length": textstat.words_per_sentence(text),
            "avg_syllables_per_word": textstat.avg_syllables_per_word(text),
        }
    except Exception:
        return {
            "fk_grade": None,
            "gunning_fog": None,
            "smog": None,
            "avg_sentence_length": None,
            "avg_syllables_per_word": None,
        }


def compute_lexical_and_discourse_features(text: str):
    text = str(text)
    text_lower = text.lower()

    words = text_lower.split()
    total = len(words)

    if total == 0:
        return {
            "type_token_ratio": None,
            "mattr": None,
            "hedge_count": None,
            "causal_count": None,
            "hedge_per_100": None,
            "causal_per_100": None,
        }

    ttr = len(set(words)) / total

    try:
        lex = LexicalRichness(text)
        mattr = lex.mattr(window_size=25) if total >= 25 else ttr
    except Exception:
        mattr = ttr

    hedge_count = sum(
        len(re.findall(r"\b" + re.escape(word) + r"\b", text_lower))
        for word in HEDGE_WORDS
    )

    causal_count = sum(
        len(re.findall(re.escape(word), text_lower))
        for word in CAUSAL_WORDS
    )

    return {
        "type_token_ratio": round(ttr, 4),
        "mattr": round(mattr, 4),
        "hedge_count": hedge_count,
        "causal_count": causal_count,
        "hedge_per_100": round((hedge_count / total) * 100, 4),
        "causal_per_100": round((causal_count / total) * 100, 4),
    }


def compute_sentiment(text: str, analyzer: SentimentIntensityAnalyzer):
    scores = analyzer.polarity_scores(str(text))

    return {
        "sentiment_pos": scores["pos"],
        "sentiment_neg": scores["neg"],
        "sentiment_neu": scores["neu"],
        "sentiment_compound": scores["compound"],
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}\n"
            "Add the canonical de-identified dataset before running this script."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"clean_text"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    print(f"Loaded {len(df):,} posts")
    print("Computing linguistic features...")

    analyzer = SentimentIntensityAnalyzer()

    readability = df["clean_text"].apply(
        lambda text: pd.Series(compute_readability(text))
    )

    lexical_discourse = df["clean_text"].apply(
        lambda text: pd.Series(
            compute_lexical_and_discourse_features(text)
        )
    )

    sentiment = df["clean_text"].apply(
        lambda text: pd.Series(
            compute_sentiment(text, analyzer)
        )
    )

    generated_columns = (
        list(readability.columns)
        + list(lexical_discourse.columns)
        + list(sentiment.columns)
    )

    existing_generated = [
        col for col in generated_columns
        if col in df.columns
    ]

    if existing_generated:
        df = df.drop(columns=existing_generated)

    df = pd.concat(
        [df, readability, lexical_discourse, sentiment],
        axis=1
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved feature-enriched dataset to:")
    print(OUTPUT_PATH)

    print("\nMean feature values by expertise tier:")

    summary_columns = [
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

    if "expertise_tier" in df.columns:
        print(
            df.groupby("expertise_tier")[summary_columns]
            .mean()
            .round(4)
        )


if __name__ == "__main__":
    main()
