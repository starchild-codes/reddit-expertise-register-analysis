from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"

PRIMARY_PATH = (
    RESULTS_DIR
    / "primary_statistics.csv"
)

POSTHOC_PATH = (
    RESULTS_DIR
    / "posthoc_dunn_bonferroni.csv"
)

PARTIAL_PATH = (
    RESULTS_DIR
    / "partial_correlations.csv"
)

ACRONYM_PATH = (
    RESULTS_DIR
    / "acronym_robustness.csv"
)

ACRONYM_DUNN_PATH = (
    RESULTS_DIR
    / "acronym_robustness_dunn.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "fdr_results.csv"
)


# ============================================================
# Helpers
# ============================================================

def apply_bh(
    df,
    p_column,
    family_name
):
    """
    Apply Benjamini-Hochberg FDR correction to one
    explicitly defined family of hypothesis tests.
    """

    corrected = (
        df.copy()
        .reset_index(drop=True)
    )

    corrected[
        "q_value"
    ] = pd.NA

    corrected[
        "reject_fdr_0_05"
    ] = pd.NA

    valid_mask = (
        corrected[
            p_column
        ]
        .notna()
    )

    if valid_mask.sum() > 0:

        reject, q_values, _, _ = (
            multipletests(
                corrected.loc[
                    valid_mask,
                    p_column
                ].astype(float),
                alpha=0.05,
                method="fdr_bh",
            )
        )

        corrected.loc[
            valid_mask,
            "q_value"
        ] = q_values

        corrected.loc[
            valid_mask,
            "reject_fdr_0_05"
        ] = reject


    corrected[
        "test_family"
    ] = family_name

    corrected[
        "raw_p_column"
    ] = p_column

    return corrected


def normalize_feature_column(
    df
):
    """
    Robustness outputs use 'metric' while the primary
    analysis uses 'feature'. Standardize everything to
    the column name 'feature'.
    """

    result = df.copy()

    if (
        "feature"
        not in result.columns
        and
        "metric"
        in result.columns
    ):

        result = (
            result.rename(
                columns={
                    "metric":
                    "feature"
                }
            )
        )

    return result


# ============================================================
# Family 1
# Kruskal-Wallis omnibus tests
# ============================================================

def build_omnibus_family():

    frames = []


    # --------------------------------------------------------
    # Main linguistic measures
    # --------------------------------------------------------

    if not PRIMARY_PATH.exists():

        raise FileNotFoundError(
            f"Missing required file: "
            f"{PRIMARY_PATH}"
        )


    primary = pd.read_csv(
        PRIMARY_PATH
    )


    required = {
        "feature",
        "kruskal_h",
        "p_value",
    }


    missing = (
        required
        - set(
            primary.columns
        )
    )


    if missing:

        raise ValueError(
            f"{PRIMARY_PATH.name} is missing columns: "
            f"{sorted(missing)}"
        )


    primary_subset = (
        primary[
            [
                "feature",
                "kruskal_h",
                "p_value",
            ]
        ]
        .copy()
    )


    primary_subset[
        "analysis_source"
    ] = "primary_statistics"


    frames.append(
        primary_subset
    )


    # --------------------------------------------------------
    # Acronym robustness omnibus tests
    # --------------------------------------------------------

    if ACRONYM_PATH.exists():

        acronym = pd.read_csv(
            ACRONYM_PATH
        )

        acronym = (
            normalize_feature_column(
                acronym
            )
        )


        required = {
            "feature",
            "kruskal_h",
            "p_value",
        }


        missing = (
            required
            - set(
                acronym.columns
            )
        )


        if missing:

            raise ValueError(
                f"{ACRONYM_PATH.name} is missing columns: "
                f"{sorted(missing)}"
            )


        acronym_subset = (
            acronym[
                [
                    "feature",
                    "kruskal_h",
                    "p_value",
                ]
            ]
            .copy()
        )


        # avg_syllables_per_word already appears in the
        # primary family. Do not count the same hypothesis twice.
        acronym_subset = (
            acronym_subset[
                acronym_subset[
                    "feature"
                ]
                != "avg_syllables_per_word"
            ]
            .copy()
        )


        acronym_subset[
            "analysis_source"
        ] = "acronym_robustness"


        frames.append(
            acronym_subset
        )


    omnibus = pd.concat(
        frames,
        ignore_index=True,
    )


    # Defensive duplicate check.
    omnibus = (
        omnibus
        .drop_duplicates(
            subset=[
                "feature"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


    omnibus[
        "comparison"
    ] = "three-tier omnibus"


    omnibus = apply_bh(
        omnibus,
        p_column="p_value",
        family_name=(
            "kruskal_wallis_omnibus"
        ),
    )


    print(
        "Kruskal-Wallis FDR family: "
        f"{len(omnibus)} tests"
    )


    return omnibus


# ============================================================
# Family 2
# Dunn metric-by-tier-pair comparisons
# ============================================================

def build_dunn_family():

    frames = []


    # --------------------------------------------------------
    # Main Dunn tests
    # --------------------------------------------------------

    if not POSTHOC_PATH.exists():

        raise FileNotFoundError(
            f"Missing required file: "
            f"{POSTHOC_PATH}"
        )


    posthoc = pd.read_csv(
        POSTHOC_PATH
    )


    required = {
        "feature",
        "tier_a",
        "tier_b",
        "p_bonferroni",
    }


    missing = (
        required
        - set(
            posthoc.columns
        )
    )


    if missing:

        raise ValueError(
            f"{POSTHOC_PATH.name} is missing columns: "
            f"{sorted(missing)}"
        )


    primary_dunn = (
        posthoc[
            [
                "feature",
                "tier_a",
                "tier_b",
                "p_bonferroni",
            ]
        ]
        .copy()
    )


    primary_dunn[
        "analysis_source"
    ] = "primary_posthoc"


    frames.append(
        primary_dunn
    )


    # --------------------------------------------------------
    # Acronym robustness Dunn tests
    # --------------------------------------------------------

    if ACRONYM_DUNN_PATH.exists():

        acronym_dunn = pd.read_csv(
            ACRONYM_DUNN_PATH
        )


        acronym_dunn = (
            normalize_feature_column(
                acronym_dunn
            )
        )


        required = {
            "feature",
            "tier_a",
            "tier_b",
            "p_bonferroni",
        }


        missing = (
            required
            - set(
                acronym_dunn.columns
            )
        )


        if missing:

            raise ValueError(
                f"{ACRONYM_DUNN_PATH.name} "
                "is missing columns: "
                f"{sorted(missing)}"
            )


        acronym_dunn = (
            acronym_dunn[
                [
                    "feature",
                    "tier_a",
                    "tier_b",
                    "p_bonferroni",
                ]
            ]
            .copy()
        )


        # The original syllable metric is already included in
        # the ordinary post-hoc results.
        acronym_dunn = (
            acronym_dunn[
                acronym_dunn[
                    "feature"
                ]
                != "avg_syllables_per_word"
            ]
            .copy()
        )


        acronym_dunn[
            "analysis_source"
        ] = "acronym_robustness"


        frames.append(
            acronym_dunn
        )


    dunn = pd.concat(
        frames,
        ignore_index=True,
    )


    # One row = one metric-by-tier-pair hypothesis.
    dunn = (
        dunn
        .drop_duplicates(
            subset=[
                "feature",
                "tier_a",
                "tier_b",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


    dunn[
        "comparison"
    ] = (
        dunn[
            "tier_a"
        ].astype(str)
        + " vs "
        + dunn[
            "tier_b"
        ].astype(str)
    )


    # Dunn values have already received the within-metric
    # Bonferroni adjustment. BH here corresponds to the
    # manuscript's separate FDR family across the resulting
    # metric-by-tier-pair comparisons.
    dunn = apply_bh(
        dunn,
        p_column="p_bonferroni",
        family_name=(
            "dunn_metric_by_tier_pair"
        ),
    )


    print(
        "Dunn FDR family: "
        f"{len(dunn)} metric-by-tier-pair tests"
    )


    return dunn


# ============================================================
# Family 3
# Partial correlations
# ============================================================

def build_partial_families():

    if not PARTIAL_PATH.exists():

        raise FileNotFoundError(
            f"Missing required file: "
            f"{PARTIAL_PATH}"
        )


    partial = pd.read_csv(
        PARTIAL_PATH
    )


    required = {
        "feature",
        "covariates",
        "r_partial",
        "p_value",
    }


    missing = (
        required
        - set(
            partial.columns
        )
    )


    if missing:

        raise ValueError(
            f"{PARTIAL_PATH.name} is missing columns: "
            f"{sorted(missing)}"
        )


    frames = []


    # --------------------------------------------------------
    # Primary partial-correlation family:
    # control for post length / word count
    # --------------------------------------------------------

    word_count = (
        partial[
            partial[
                "covariates"
            ]
            == "word_count"
        ]
        .copy()
    )


    if not word_count.empty:

        word_count[
            "comparison"
        ] = (
            "expertise tier | word_count"
        )


        word_count[
            "analysis_source"
        ] = (
            "partial_correlations"
        )


        word_count = apply_bh(
            word_count,
            p_column="p_value",
            family_name=(
                "partial_correlation_word_count"
            ),
        )


        frames.append(
            word_count
        )


        print(
            "Word-count partial-correlation "
            "FDR family: "
            f"{len(word_count)} tests"
        )


    # --------------------------------------------------------
    # Acronym-density robustness partial correlation
    # --------------------------------------------------------

    acronym_model = (
        partial[
            partial[
                "covariates"
            ]
            == (
                "word_count + "
                "acronym_density_per_100_words"
            )
        ]
        .copy()
    )


    if not acronym_model.empty:

        acronym_model[
            "comparison"
        ] = (
            "expertise tier | "
            "word_count + acronym density"
        )


        acronym_model[
            "analysis_source"
        ] = (
            "partial_correlations"
        )


        # This is a separately specified robustness model.
        # If it contains one test, its BH q-value is naturally
        # identical to its p-value.
        acronym_model = apply_bh(
            acronym_model,
            p_column="p_value",
            family_name=(
                "partial_correlation_"
                "word_count_plus_acronym_density"
            ),
        )


        frames.append(
            acronym_model
        )


        print(
            "Acronym-density partial-correlation "
            "FDR family: "
            f"{len(acronym_model)} test(s)"
        )


    if not frames:

        raise RuntimeError(
            "No recognized partial-correlation "
            "families were found."
        )


    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "Applying Benjamini-Hochberg "
        "false-discovery-rate correction...\n"
    )


    omnibus = (
        build_omnibus_family()
    )


    dunn = (
        build_dunn_family()
    )


    partial = (
        build_partial_families()
    )


    # ========================================================
    # Combine canonical FDR output
    # ========================================================

    combined = pd.concat(
        [
            omnibus,
            dunn,
            partial,
        ],
        ignore_index=True,
        sort=False,
    )


    preferred_columns = [
        "test_family",
        "analysis_source",
        "feature",
        "comparison",
        "covariates",
        "kruskal_h",
        "r_partial",
        "tier_a",
        "tier_b",
        "p_value",
        "p_bonferroni",
        "raw_p_column",
        "q_value",
        "reject_fdr_0_05",
    ]


    ordered_columns = [
        column
        for column
        in preferred_columns
        if column
        in combined.columns
    ]


    remaining_columns = [
        column
        for column
        in combined.columns
        if column
        not in ordered_columns
    ]


    combined = combined[
        ordered_columns
        + remaining_columns
    ]


    combined.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "\nSaved canonical FDR results to:"
    )

    print(
        OUTPUT_PATH
    )


    print(
        "\nRows written:",
        len(combined)
    )


    print(
        "\nFDR families:"
    )


    for (
        family,
        count
    ) in (
        combined[
            "test_family"
        ]
        .value_counts()
        .items()
    ):

        print(
            f" - {family}: {count}"
        )


if __name__ == "__main__":
    main()
