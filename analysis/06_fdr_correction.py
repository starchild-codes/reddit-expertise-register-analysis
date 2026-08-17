from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]

PRIMARY_PATH = ROOT / "results" / "primary_statistics.csv"
POSTHOC_PATH = ROOT / "results" / "posthoc_dunn_bonferroni.csv"
PARTIAL_PATH = ROOT / "results" / "partial_correlations.csv"

OUTPUT_PATH = ROOT / "results" / "fdr_results.csv"


def apply_bh(df, p_column, family_name):
    """
    Apply Benjamini-Hochberg FDR correction to one family of tests.
    """

    subset = df.copy()

    valid_mask = subset[p_column].notna()

    subset["q_value"] = pd.NA
    subset["reject_fdr_0_05"] = pd.NA

    if valid_mask.sum() == 0:
        subset["test_family"] = family_name
        return subset

    reject, q_values, _, _ = multipletests(
        subset.loc[valid_mask, p_column].astype(float),
        alpha=0.05,
        method="fdr_bh",
    )

    subset.loc[valid_mask, "q_value"] = q_values
    subset.loc[valid_mask, "reject_fdr_0_05"] = reject

    subset["test_family"] = family_name

    return subset


def main():
    output_frames = []

    # ============================================================
    # FAMILY 1 — Omnibus Kruskal-Wallis tests
    # ============================================================

    if PRIMARY_PATH.exists():
        primary = pd.read_csv(PRIMARY_PATH)

        required = {"feature", "p_value"}

        if not required.issubset(primary.columns):
            raise ValueError(
                f"{PRIMARY_PATH.name} is missing required columns: "
                f"{sorted(required - set(primary.columns))}"
            )

        omnibus = primary[
            [
                "feature",
                "kruskal_h",
                "p_value",
                "cohens_d_high_vs_low",
            ]
        ].copy()

        omnibus["comparison"] = "three-tier omnibus"

        omnibus = apply_bh(
            omnibus,
            p_column="p_value",
            family_name="kruskal_wallis",
        )

        output_frames.append(omnibus)

        print(
            f"Applied BH-FDR to "
            f"{len(omnibus)} Kruskal-Wallis tests"
        )

    else:
        print(
            f"Skipping omnibus FDR: "
            f"{PRIMARY_PATH.name} not found"
        )

    # ============================================================
    # FAMILY 2 — Dunn pairwise comparisons
    # ============================================================

    if POSTHOC_PATH.exists():
        posthoc = pd.read_csv(POSTHOC_PATH)

        required = {
            "feature",
            "tier_a",
            "tier_b",
            "p_bonferroni",
        }

        if not required.issubset(posthoc.columns):
            raise ValueError(
                f"{POSTHOC_PATH.name} is missing required columns: "
                f"{sorted(required - set(posthoc.columns))}"
            )

        dunn = posthoc.copy()

        dunn["comparison"] = (
            dunn["tier_a"].astype(str)
            + " vs "
            + dunn["tier_b"].astype(str)
        )

        # Note:
        # The p-values here are already Bonferroni-adjusted within
        # each Dunn analysis. This additional BH step reproduces the
        # manuscript's cross-metric multiple-testing correction family.
        dunn = apply_bh(
            dunn,
            p_column="p_bonferroni",
            family_name="dunn_pairwise",
        )

        output_frames.append(dunn)

        print(
            f"Applied BH-FDR to "
            f"{len(dunn)} Dunn pairwise tests"
        )

    else:
        print(
            f"Skipping Dunn FDR: "
            f"{POSTHOC_PATH.name} not found"
        )

    # ============================================================
    # FAMILY 3 — Partial correlations
    # ============================================================

    if PARTIAL_PATH.exists():
        partial = pd.read_csv(PARTIAL_PATH)

        required = {
            "feature",
            "covariates",
            "r_partial",
            "p_value",
        }

        if not required.issubset(partial.columns):
            raise ValueError(
                f"{PARTIAL_PATH.name} is missing required columns: "
                f"{sorted(required - set(partial.columns))}"
            )

        # Important:
        # Treat each covariate specification as a separate family.
        # Example:
        #   word_count
        #   word_count + acronym_density_per_100

        for covariate_model, group in partial.groupby(
            "covariates",
            dropna=False
        ):
            corrected = apply_bh(
                group.copy(),
                p_column="p_value",
                family_name=(
                    "partial_correlation__"
                    + str(covariate_model)
                    .replace(" ", "_")
                    .replace("+", "plus")
                ),
            )

            corrected["comparison"] = (
                "expertise tier | "
                + str(covariate_model)
            )

            output_frames.append(corrected)

            print(
                f"Applied BH-FDR to "
                f"{len(corrected)} partial correlations "
                f"for covariates: {covariate_model}"
            )

    else:
        print(
            f"Skipping partial-correlation FDR: "
            f"{PARTIAL_PATH.name} not found"
        )

    # ============================================================
    # COMBINE RESULTS
    # ============================================================

    if not output_frames:
        raise FileNotFoundError(
            "No result files were found. Run the preceding analysis "
            "scripts before applying FDR correction."
        )

    combined = pd.concat(
        output_frames,
        ignore_index=True,
        sort=False,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved FDR-corrected results to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
