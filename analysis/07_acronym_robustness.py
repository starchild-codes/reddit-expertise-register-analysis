from pathlib import Path
import re

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy import stats


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features.csv"
)

FEATURE_OUTPUT_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features_acronyms.csv"
)

CANONICAL_OUTPUT_PATH = (
    ROOT
    / "results"
    / "acronym_robustness.csv"
)

DUNN_OUTPUT_PATH = (
    ROOT
    / "results"
    / "acronym_robustness_dunn.csv"
)

PARTIAL_OUTPUT_PATH = (
    ROOT
    / "results"
    / "acronym_robustness_partial_correlations.csv"
)


# ============================================================
# Configuration
# ============================================================

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

ACRONYM_RE = re.compile(
    r"\b[A-Z]{2,}(?:[-]?[A-Z0-9]+)*\b"
)

VOWELS = set(
    "aeiouyAEIOUY"
)


# ============================================================
# Syllable-counting helpers
# ============================================================

def count_syllables_word(word):
    """
    Rule-based syllable counter used only for the
    acronym-removal and acronym-normalization analyses.

    The primary avg_syllables_per_word variable is taken
    from analysis/02_linguistic_features.py.
    """

    word = (
        str(word)
        .lower()
        .strip(
            ".,!?;:'\"()[]{}"
        )
    )

    if not word:
        return 0

    syllable_count = 0
    previous_was_vowel = False

    for character in word:

        is_vowel = (
            character in VOWELS
        )

        if (
            is_vowel
            and not previous_was_vowel
        ):
            syllable_count += 1

        previous_was_vowel = is_vowel


    if (
        word.endswith("e")
        and syllable_count > 1
    ):
        syllable_count -= 1


    return max(
        1,
        syllable_count
    )


def avg_syllables_per_word_custom(text):

    words = (
        str(text)
        .split()
    )

    if not words:
        return np.nan

    total_syllables = sum(
        count_syllables_word(
            word
        )
        for word in words
    )

    return (
        total_syllables
        / len(words)
    )


# ============================================================
# Acronym-feature computation
# ============================================================

def compute_acronym_metrics(text):

    text = str(text)

    acronyms = (
        ACRONYM_RE
        .findall(text)
    )

    acronym_count = len(
        acronyms
    )


    words = (
        text.split()
    )

    word_count = len(
        words
    )


    if word_count > 0:

        acronym_density = (
            acronym_count
            / word_count
            * 100
        )

    else:

        acronym_density = 0.0


    # --------------------------------------------------------
    # Acronyms removed
    # --------------------------------------------------------

    text_without_acronyms = (
        ACRONYM_RE.sub(
            "",
            text
        )
    )

    words_without_acronyms = (
        text_without_acronyms
        .split()
    )


    if words_without_acronyms:

        syllables_without_acronyms = (
            avg_syllables_per_word_custom(
                text_without_acronyms
            )
        )

    else:

        syllables_without_acronyms = (
            np.nan
        )


    # --------------------------------------------------------
    # Acronyms normalized
    # --------------------------------------------------------

    text_normalized = (
        ACRONYM_RE.sub(
            "term",
            text
        )
    )


    if text_normalized.split():

        syllables_normalized = (
            avg_syllables_per_word_custom(
                text_normalized
            )
        )

    else:

        syllables_normalized = (
            np.nan
        )


    return {
        "acronym_count":
            acronym_count,

        "acronym_density_per_100_words":
            round(
                acronym_density,
                4
            ),

        "word_count_without_acronyms":
            len(
                words_without_acronyms
            ),

        "avg_syllables_per_word_without_acronyms":
            (
                round(
                    syllables_without_acronyms,
                    4
                )
                if pd.notna(
                    syllables_without_acronyms
                )
                else np.nan
            ),

        "avg_syllables_per_word_acronyms_normalized":
            (
                round(
                    syllables_normalized,
                    4
                )
                if pd.notna(
                    syllables_normalized
                )
                else np.nan
            ),
    }


# ============================================================
# Statistical helpers
# ============================================================

def cohens_d_high_vs_low(
    df,
    metric
):

    low = (
        df.loc[
            df["expertise_tier"]
            == "low",
            metric
        ]
        .dropna()
        .astype(float)
    )

    high = (
        df.loc[
            df["expertise_tier"]
            == "high",
            metric
        ]
        .dropna()
        .astype(float)
    )


    if (
        len(low) < 2
        or len(high) < 2
    ):
        return np.nan


    pooled_sd = np.sqrt(
        (
            low.var(
                ddof=1
            )
            +
            high.var(
                ddof=1
            )
        )
        / 2
    )


    if (
        pooled_sd == 0
        or pd.isna(
            pooled_sd
        )
    ):
        return np.nan


    return (
        high.mean()
        - low.mean()
    ) / pooled_sd


def extract_p_value(result):

    for column in result.columns:

        normalized = (
            column.lower()
            .replace("-", "")
            .replace("_", "")
        )

        if "pval" in normalized:

            return (
                result[
                    column
                ]
                .iloc[0]
            )

    raise KeyError(
        "Could not identify p-value column "
        "from Pingouin output."
    )


def run_partial_correlation(
    df,
    feature,
    covariates
):

    required = [
        feature,
        "tier_numeric",
        *covariates,
    ]


    subset = (
        df[
            required
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
        "feature":
            feature,

        "covariates":
            " + ".join(
                covariates
            ),

        "n":
            len(subset),

        "r_partial":
            result["r"]
            .iloc[0],

        "p_value":
            extract_p_value(
                result
            ),
    }


# ============================================================
# Main
# ============================================================

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
        "clean_text",
        "expertise_tier",
        "word_count",
        "avg_syllables_per_word",
    }


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
            .unique()
        )

        raise ValueError(
            "Unexpected expertise-tier labels: "
            f"{unexpected}"
        )


    print(
        f"Loaded {len(df):,} posts"
    )

    print(
        "Computing acronym robustness features..."
    )


    # ========================================================
    # Compute acronym-derived variables
    # ========================================================

    acronym_features = (
        df["clean_text"]
        .apply(
            lambda text: pd.Series(
                compute_acronym_metrics(
                    text
                )
            )
        )
    )


    duplicate_columns = [
        column
        for column
        in acronym_features.columns
        if column
        in df.columns
    ]


    if duplicate_columns:

        df = df.drop(
            columns=
                duplicate_columns
        )


    df = pd.concat(
        [
            df,
            acronym_features
        ],
        axis=1
    )


    # ========================================================
    # Save acronym-enriched feature dataset
    # ========================================================

    FEATURE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False
    )


    print(
        "\nSaved acronym-enriched feature dataset:"
    )

    print(
        FEATURE_OUTPUT_PATH
    )


    # ========================================================
    # Canonical robustness metrics
    # ========================================================

    robustness_metrics = [
        "acronym_density_per_100_words",
        "avg_syllables_per_word",
        "avg_syllables_per_word_without_acronyms",
        "avg_syllables_per_word_acronyms_normalized",
    ]


    canonical_rows = []
    dunn_rows = []


    for metric in robustness_metrics:

        tier_values = {}

        for tier in TIER_ORDER:

            tier_values[tier] = (
                df.loc[
                    df["expertise_tier"]
                    == tier,
                    metric
                ]
                .dropna()
                .astype(float)
            )


        h_stat, p_value = (
            stats.kruskal(
                tier_values["low"],
                tier_values["medium"],
                tier_values["high"],
            )
        )


        d_high_low = (
            cohens_d_high_vs_low(
                df,
                metric
            )
        )


        canonical_rows.append({
            "metric":
                metric,

            "low_mean":
                tier_values[
                    "low"
                ].mean(),

            "medium_mean":
                tier_values[
                    "medium"
                ].mean(),

            "high_mean":
                tier_values[
                    "high"
                ].mean(),

            "kruskal_h":
                h_stat,

            "p_value":
                p_value,

            "cohens_d_high_vs_low":
                d_high_low,
        })


        # ----------------------------------------------------
        # Dunn pairwise tests
        # ----------------------------------------------------

        subset = (
            df[
                [
                    "expertise_tier",
                    metric
                ]
            ]
            .dropna()
            .copy()
        )


        matrix = sp.posthoc_dunn(
            subset,
            val_col=metric,
            group_col="expertise_tier",
            p_adjust="bonferroni",
        )


        pairs = [
            (
                "low",
                "medium"
            ),
            (
                "low",
                "high"
            ),
            (
                "medium",
                "high"
            ),
        ]


        for tier_a, tier_b in pairs:

            dunn_rows.append({
                "metric":
                    metric,

                "tier_a":
                    tier_a,

                "tier_b":
                    tier_b,

                "p_bonferroni":
                    matrix.loc[
                        tier_a,
                        tier_b
                    ],
            })


    canonical_df = pd.DataFrame(
        canonical_rows
    )


    dunn_df = pd.DataFrame(
        dunn_rows
    )


    # ========================================================
    # Save canonical robustness summary
    # ========================================================

    CANONICAL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    canonical_df.to_csv(
        CANONICAL_OUTPUT_PATH,
        index=False
    )


    print(
        "\nCanonical acronym robustness results:"
    )

    print(
        canonical_df
        .round(4)
        .to_string(
            index=False
        )
    )


    print(
        "\nSaved canonical robustness output:"
    )

    print(
        CANONICAL_OUTPUT_PATH
    )


    # ========================================================
    # Save Dunn robustness tests
    # ========================================================

    dunn_df.to_csv(
        DUNN_OUTPUT_PATH,
        index=False
    )


    print(
        "\nSaved acronym robustness Dunn tests:"
    )

    print(
        DUNN_OUTPUT_PATH
    )


    # ========================================================
    # Partial correlations
    # ========================================================

    partial_rows = []


    # Main syllable metric controlling for word count
    result = run_partial_correlation(
        df=df,
        feature=(
            "avg_syllables_per_word"
        ),
        covariates=[
            "word_count"
        ],
    )


    if result is not None:

        partial_rows.append(
            result
        )


    # Main syllable metric controlling for
    # word count + acronym density
    result = run_partial_correlation(
        df=df,
        feature=(
            "avg_syllables_per_word"
        ),
        covariates=[
            "word_count",
            "acronym_density_per_100_words",
        ],
    )


    if result is not None:

        partial_rows.append(
            result
        )


    # Acronym-removed syllable metric
    result = run_partial_correlation(
        df=df,
        feature=(
            "avg_syllables_per_word_without_acronyms"
        ),
        covariates=[
            "word_count"
        ],
    )


    if result is not None:

        partial_rows.append(
            result
        )


    # Acronym-normalized syllable metric
    result = run_partial_correlation(
        df=df,
        feature=(
            "avg_syllables_per_word_acronyms_normalized"
        ),
        covariates=[
            "word_count"
        ],
    )


    if result is not None:

        partial_rows.append(
            result
        )


    partial_df = pd.DataFrame(
        partial_rows
    )


    partial_df.to_csv(
        PARTIAL_OUTPUT_PATH,
        index=False
    )


    print(
        "\nSaved acronym robustness partial correlations:"
    )

    print(
        PARTIAL_OUTPUT_PATH
    )


    # ========================================================
    # Key-result summary
    # ========================================================

    print(
        "\nKey robustness results"
    )


    for metric in [
        "avg_syllables_per_word",
        "avg_syllables_per_word_without_acronyms",
        "avg_syllables_per_word_acronyms_normalized",
    ]:

        row = (
            canonical_df[
                canonical_df[
                    "metric"
                ]
                == metric
            ]
            .iloc[0]
        )

        print(
            f"{metric:<50} "
            f"H={row['kruskal_h']:.2f}  "
            f"p={row['p_value']:.6g}  "
            f"d={row['cohens_d_high_vs_low']:.3f}"
        )


if __name__ == "__main__":
    main()
