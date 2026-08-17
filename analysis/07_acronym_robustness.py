from pathlib import Path
import re

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy import stats


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

SUMMARY_OUTPUT_PATH = (
    ROOT
    / "results"
    / "acronym_robustness_summary.csv"
)

KW_OUTPUT_PATH = (
    ROOT
    / "results"
    / "acronym_robustness_kruskal.csv"
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


# Original regex used in the research analysis.
ACRONYM_RE = re.compile(
    r"\b[A-Z]{2,}(?:[-]?[A-Z0-9]+)*\b"
)

VOWELS = set(
    "aeiouyAEIOUY"
)


def count_syllables_word(word: str) -> int:
    """
    Rule-based syllable counter retained from the original
    acronym robustness analysis.

    Important:
    This implementation is used only for the acronym-removal and
    acronym-normalization sensitivity analyses.

    The primary avg_syllables_per_word variable is computed by
    textstat in analysis/02_linguistic_features.py.
    """

    word = (
        str(word)
        .lower()
        .strip(
            ".,!?;:'\"()"
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

        previous_was_vowel = (
            is_vowel
        )

    # Original silent-e adjustment.
    if (
        word.endswith("e")
        and syllable_count > 1
    ):
        syllable_count -= 1

    return max(
        1,
        syllable_count
    )


def avg_syllables_per_word_custom(
    text: str
) -> float:

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


def compute_acronym_metrics(
    text: str
):

    text = str(
        text
    )

    acronyms = (
        ACRONYM_RE.findall(
            text
        )
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

    acronym_density = (
        (
            acronym_count
            / word_count
        )
        * 100
        if word_count > 0
        else 0.0
    )

    # ============================================================
    # ACRONYMS REMOVED
    # ============================================================

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

    # ============================================================
    # ACRONYMS NORMALIZED
    # ============================================================

    # Original analysis replaced each detected acronym with
    # the neutral one-syllable token "term".
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


def cohens_d_high_vs_low(
    df,
    metric
):

    low = df.loc[
        df["expertise_tier"]
        == "low",
        metric
    ].dropna()

    high = df.loc[
        df["expertise_tier"]
        == "high",
        metric
    ].dropna()

    if (
        len(low) < 2
        or len(high) < 2
    ):
        return np.nan

    pooled_sd = np.sqrt(
        (
            low.std(
                ddof=1
            ) ** 2
            +
            high.std(
                ddof=1
            ) ** 2
        )
        / 2
    )

    if pooled_sd == 0:
        return 0.0

    return (
        high.mean()
        - low.mean()
    ) / pooled_sd


def extract_p_value(
    result
):
    for column in result.columns:

        normalized = (
            column.lower()
            .replace(
                "-",
                ""
            )
            .replace(
                "_",
                ""
            )
        )

        if "pval" in normalized:
            return (
                result[
                    column
                ]
                .iloc[0]
            )

    raise KeyError(
        "Could not identify the p-value column. "
        f"Returned columns: "
        f"{result.columns.tolist()}"
    )


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

    print(
        f"Loaded {len(df):,} posts"
    )

    print(
        "Computing acronym robustness features..."
    )

    # ============================================================
    # COMPUTE ACRONYM FEATURES
    # ============================================================

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

    # Safe reruns:
    # remove any old acronym-derived columns before concatenating.
    duplicate_columns = [
        column
        for column
        in acronym_features.columns
        if column in df.columns
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

    df["tier_numeric"] = (
        df["expertise_tier"]
        .map(
            TIER_MAP
        )
    )

    # ============================================================
    # SAVE ACRONYM-ENRICHED DATASET
    # ============================================================

    FEATURE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        FEATURE_OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved acronym-enriched dataset to:"
    )
    print(
        FEATURE_OUTPUT_PATH
    )

    # ============================================================
    # SUMMARY TABLE
    # ============================================================

    summary_metrics = [
        "acronym_density_per_100_words",
        "avg_syllables_per_word",
        "avg_syllables_per_word_without_acronyms",
        "avg_syllables_per_word_acronyms_normalized",
    ]

    summary = (
        df
        .groupby(
            "expertise_tier"
        )[
            summary_metrics
        ]
        .agg(
            [
                "mean",
                "std",
                "count"
            ]
        )
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH
    )

    print(
        "\nSummary by expertise tier:"
    )

    print(
        summary.round(
            4
        ).to_string()
    )

    print(
        f"\nSaved summary to:\n"
        f"{SUMMARY_OUTPUT_PATH}"
    )

    # ============================================================
    # KRUSKAL-WALLIS TESTS
    # ============================================================

    kw_metrics = [
        "acronym_density_per_100_words",
        "avg_syllables_per_word_without_acronyms",
        "avg_syllables_per_word_acronyms_normalized",
    ]

    kw_rows = []

    print(
        "\n"
        + "=" * 70
    )
    print(
        "Kruskal-Wallis acronym robustness tests"
    )
    print(
        "=" * 70
    )

    for metric in kw_metrics:

        groups = [
            df.loc[
                df[
                    "expertise_tier"
                ]
                == tier,
                metric
            ]
            .dropna()

            for tier
            in TIER_ORDER
        ]

        h_stat, p_value = (
            stats.kruskal(
                *groups
            )
        )

        d_value = (
            cohens_d_high_vs_low(
                df,
                metric
            )
        )

        kw_rows.append({
            "metric":
                metric,

            "kruskal_h":
                h_stat,

            "p_value":
                p_value,

            "cohens_d_high_vs_low":
                d_value,
        })

        print(
            f"{metric:<52} "
            f"H={h_stat:.3f}  "
            f"p={p_value:.6g}  "
            f"d={d_value:+.4f}"
        )

    kw_results = (
        pd.DataFrame(
            kw_rows
        )
    )

    kw_results.to_csv(
        KW_OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved Kruskal-Wallis results to:\n"
        f"{KW_OUTPUT_PATH}"
    )

    # ============================================================
    # DUNN POST-HOC TESTS
    # ============================================================

    dunn_rows = []

    print(
        "\n"
        + "=" * 70
    )
    print(
        "Dunn post-hoc tests with Bonferroni correction"
    )
    print(
        "=" * 70
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

    for metric in kw_metrics:

        subset = (
            df[
                [
                    "expertise_tier",
                    metric
                ]
            ]
            .dropna()
        )

        matrix = (
            sp.posthoc_dunn(
                subset,
                val_col=metric,
                group_col=
                    "expertise_tier",
                p_adjust=
                    "bonferroni",
            )
        )

        print(
            f"\n--- {metric} ---"
        )

        for (
            tier_a,
            tier_b
        ) in pairs:

            p_adjusted = (
                matrix.loc[
                    tier_a,
                    tier_b
                ]
            )

            dunn_rows.append({
                "metric":
                    metric,

                "tier_a":
                    tier_a,

                "tier_b":
                    tier_b,

                "p_bonferroni":
                    p_adjusted,

                "significant_0_05":
                    (
                        p_adjusted
                        < 0.05
                    ),
            })

            print(
                f"{tier_a:<7} "
                f"vs "
                f"{tier_b:<7} "
                f"p_adj="
                f"{p_adjusted:.6g}"
            )

    dunn_results = (
        pd.DataFrame(
            dunn_rows
        )
    )

    dunn_results.to_csv(
        DUNN_OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved Dunn results to:\n"
        f"{DUNN_OUTPUT_PATH}"
    )

    # ============================================================
    # PARTIAL CORRELATIONS
    # ============================================================

    partial_specs = [
        (
            "avg_syllables_per_word",
            [
                "word_count"
            ],
        ),
        (
            "avg_syllables_per_word",
            [
                "word_count",
                "acronym_density_per_100_words",
            ],
        ),
        (
            "avg_syllables_per_word_without_acronyms",
            [
                "word_count"
            ],
        ),
        (
            "avg_syllables_per_word_acronyms_normalized",
            [
                "word_count"
            ],
        ),
    ]

    partial_rows = []

    print(
        "\n"
        + "=" * 70
    )
    print(
        "Partial-correlation robustness tests"
    )
    print(
        "=" * 70
    )

    for (
        metric,
        covariates
    ) in partial_specs:

        required_columns = [
            metric,
            "tier_numeric",
            *covariates,
        ]

        subset = (
            df[
                required_columns
            ]
            .dropna()
        )

        result = (
            pg.partial_corr(
                data=subset,
                x=metric,
                y="tier_numeric",
                covar=covariates,
                method="pearson",
            )
        )

        r_value = (
            result[
                "r"
            ]
            .iloc[0]
        )

        p_value = (
            extract_p_value(
                result
            )
        )

        partial_rows.append({
            "metric":
                metric,

            "covariates":
                " + ".join(
                    covariates
                ),

            "n":
                len(
                    subset
                ),

            "r_partial":
                r_value,

            "p_value":
                p_value,
        })

        print(
            f"{metric:<48} "
            f"| "
            f"{' + '.join(covariates):<45} "
            f"r={r_value:+.4f}  "
            f"p={p_value:.6g}"
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

    print(
        f"\nSaved partial correlations to:\n"
        f"{PARTIAL_OUTPUT_PATH}"
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print(
        "\n"
        + "=" * 70
    )
    print(
        "ACRONYM ROBUSTNESS ANALYSIS COMPLETE"
    )
    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        FEATURE_OUTPUT_PATH
    )
    print(
        SUMMARY_OUTPUT_PATH
    )
    print(
        KW_OUTPUT_PATH
    )
    print(
        DUNN_OUTPUT_PATH
    )
    print(
        PARTIAL_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
