# Community Technical Orientation and Linguistic Register in AI Reddit Communities: A Cluster-Aware Exploratory Reanalysis

This repository accompanies the final submitted manuscript, *Community Technical Orientation and Linguistic Register in AI Reddit Communities: A Cluster-Aware Exploratory Reanalysis*.

## Current study

The paper reanalyzes 1,778 historically collected, highly scored posts from nine Reddit communities. Communities are grouped into three exploratory community technical-orientation tiers (three communities per tier). These are subreddit-level intended-audience labels, not measures of individual author expertise.

The primary inferential unit is subreddit. A conservative AI/ML screen retains 1,299 posts. In exact cluster-level permutation tests across nine subreddit means, none of ten planned outcomes survives Benjamini-Hochberg FDR correction. Topic- and log-length-adjusted analyses likewise yield no FDR-significant result. Grouped validation does not show reliable generalization to unseen communities.

The design therefore does not support causal audience-adaptation claims, individual-expertise claims, or robust general tier-level linguistic separation.

## External-rater tier sensitivity

Two external raters independently classified the nine communities. They agreed on 8 of 9 communities (88.9%; Cohen's kappa = .82), but neither reproduced the author's original mapping exactly. Both placed r/Futurology and r/GenerativeAI in the medium tier rather than the original low tier, and they differed from one another only for r/ChatGPT. Exact fixed-size sensitivity analyses under their 2/4/3 and 1/5/3 mappings again found no outcome surviving BH FDR correction. This robustness check does not establish a uniquely correct community taxonomy.

## Repository structure

- `analysis/current/` - scripts for preparing non-text release metadata, constructing manuscript-facing tables, validating preserved outputs, rendering figures, and computing the external-rater sensitivity analysis.
- `results/current/` - canonical current aggregate results, topic outputs, grouped-validation outputs, de-identified AI-screen decisions, and external-rater sensitivity outputs.
- `figures/current/` - the six manuscript figures.
- `protocols/prospective_replication/` - frozen future 24-community replication protocol; it contributed no observations to this paper.
- `provenance/failed_fresh_collection/` - documentation of later failed compliant fresh-collection attempts; not study data.
- `archive/original_analysis/` - superseded post-level analysis and its outputs. Never treat these as current evidence.

## Reproduction and validation

The complete original post bodies and post-level linguistic feature matrix are not distributed. Consequently, the historical feature pipeline cannot be regenerated from a fresh clone. Aggregate outputs are preserved and can be validated:

```bash
python analysis/current/01_validate_preserved_outputs.py
```

To re-render the six figures from those preserved aggregate outputs, install
the pinned plotting dependency and run:

```bash
python -m pip install -r requirements.txt
python analysis/current/03_render_current_figures.py
```

To regenerate and validate the external-rater sensitivity outputs from the
released subreddit means:

```bash
python analysis/current/04_external_rater_sensitivity.py
python analysis/current/05_validate_rater_sensitivity.py
```

`analysis/current/00_prepare_release_outputs.py` can recreate the de-identified AI-screen decision file only if an authorized private historical audit CSV is supplied. It deliberately strips text before writing public output.

```bash
python analysis/current/00_prepare_release_outputs.py --source /secure/path/post_level_ai_screen.csv
python analysis/current/02_build_manuscript_tables.py
python analysis/current/01_validate_preserved_outputs.py
```

## Main outputs

- [Table 1: corpus composition and AI-relevance audit](results/current/table1_subreddit_composition_ai_audit.csv)
- [Table 2: cluster permutation and adjusted results](results/current/table2_cluster_permutation_results.csv)
- [Table 3: external-rater tier sensitivity](results/current/rater_sensitivity/tier_mapping_sensitivity_summary.csv) and [all ten outcomes per mapping](results/current/rater_sensitivity/tier_mapping_sensitivity_all_outcomes.csv)
- [Table 4: topic labels and overlap](results/current/topic_terms_labels.csv) and [topic-overlap matrix](results/current/topic_overlap_by_tier.csv)
- [Table 5: grouped classifier performance](results/current/table5_grouped_classifier_performance_permutation.csv)
- [Figure 1](figures/current/figure1_ai_relevance_retention.png), [Figure 2](figures/current/figure2_average_syllables_subreddit_means.png), [Figure 3](figures/current/figure3_cluster_permutation_fdr.png), [Figure 4](figures/current/figure4_topic_distribution_by_tier.png), [Figure 5](figures/current/figure5_logistic_grouped_confusion_matrix.png), [Figure 6](figures/current/figure6_leave_one_subreddit_out.png)

## Data availability

See [DATA_CODE_AVAILABILITY.md](DATA_CODE_AVAILABILITY.md). Reddit text, titles, usernames, credentials, and browser/session artifacts are not distributed.

## Interpretation guardrail

The results are exploratory and community-level. Descriptive heterogeneity across subreddits does not establish an individual-expertise difference or causal audience adaptation.
