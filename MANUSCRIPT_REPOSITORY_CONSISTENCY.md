# Manuscript Repository Consistency

The strengthened submitted manuscript, `docs/Community_Technical_Orientation_Convergence_Strengthened.docx`, is the controlling source. All values below were checked against its abstract, Methods, Results, tables, and figure captions.

| Manuscript section | Claim or value | Supporting repository file | Exact field | Status |
|---|---|---|---|---|
| Abstract, Methods | 1,778 posts; 9 subreddits; 3 orientation tiers | `results/current/corpus_integrity_audit.csv` | `posts`, `subreddits`, `tiers` | MATCH |
| Results 3.1, Table 1 | 1,299 AI-screened posts | `results/current/corpus_integrity_audit.csv` | `ai_relevant_posts` | MATCH |
| Results 3.1, Table 1 | Per-subreddit retention and counts | `results/current/table1_subreddit_composition_ai_audit.csv` | all rows | MATCH |
| Results 3.2, Table 2 | Ten tier means, exact p-values, BH q-values | `results/current/table2_cluster_permutation_results.csv` | all rows | MATCH |
| Results 3.2 | Syllables/word p=.0536, q=.5357 | `results/current/table2_cluster_permutation_results.csv` | `avg_syllables_per_word` | MATCH |
| Results 3.3, Table 2 | Adjusted p=.4286, q=.8482 for syllables/word | `results/current/table2_cluster_permutation_results.csv` | `avg_syllables_per_word` | MATCH |
| Results 3.3 | Adjusted q-values .8482 or .8893 | `results/current/table4_topic_length_adjusted_cluster_results.csv` | `fdr_p` | MATCH |
| Methods 2.4, Results 3.3 | Seven of eight topics overlap all tiers; Topic 6 fails | `results/current/topic_overlap_by_tier.csv` | `overlaps_all_tiers` | MATCH |
| Methods 2.2.1 | Author, Rater 1, and Rater 2 assignments | `results/current/rater_sensitivity/tier_assignments_three_raters.csv` | all 9 rows | MATCH |
| Methods 2.2.1, Results 3.2.5 | Rater 1 vs Rater 2: 8/9, 88.9%, kappa=.82 | `results/current/rater_sensitivity/pairwise_agreement.csv` | `rater1_vs_rater2` | MATCH |
| Supplementary rater audit | Author vs Rater 1: 6/9, kappa=.50; Author vs Rater 2: 7/9, kappa=.6667 | `results/current/rater_sensitivity/pairwise_agreement.csv` | `author_original_vs_rater1`, `author_original_vs_rater2` | MATCH |
| Methods 2.2.1, Results 3.2.5 | Three-rater Fleiss' kappa=.654 | `results/current/rater_sensitivity/fleiss_kappa_three_raters.csv` | `fleiss_kappa` | MATCH |
| Results 3.2.5, Table 3 | Rater 1 (2/4/3): causal terms smallest p=.0206, q=.2063; no q<.05 | `results/current/rater_sensitivity/tier_mapping_sensitivity_all_outcomes.csv` | `rater1`, `causal_per_100` | MATCH |
| Results 3.2.5, Table 3 | Rater 2 (1/5/3): causal terms smallest p=.0496, q=.3836; no q<.05 | `results/current/rater_sensitivity/tier_mapping_sensitivity_all_outcomes.csv` | `rater2`, `causal_per_100` | MATCH |
| Results 3.2.5, Table 3 | Adjusted external-mapping sensitivity values | no released subreddit-level residual means | public re-computation unavailable | NOT REPRODUCIBLE FROM PUBLIC INPUTS; documented limitation, not a mismatch |
| Results 3.4, Table 5 | Logistic accuracy=.3549, balanced accuracy=.3336, macro-F1=.3355 | `results/current/table5_grouped_classifier_performance_permutation.csv` | `logistic_regression` row | MATCH |
| Results 3.4, Table 5 | Random forest accuracy=.2394, balanced accuracy=.1998, macro-F1=.1880 | `results/current/table5_grouped_classifier_performance_permutation.csv` | `random_forest` row | MATCH |
| Results 3.4, Table 5 | Subreddit-label permutation p-values | `results/current/table5_grouped_classifier_performance_permutation.csv` | permutation p columns | MATCH |
| Figures 1-6 | Values, axes, and captioned content | `figures/current/` plus corresponding CSV above | named figure and CSV | MATCH |

No unexplained numerical mismatch was found. Historical output recomputation is not claimed because the repository does not contain original post bodies or the post-level feature matrix. The adjusted external-mapping results are reported by the manuscript as historical working-data analyses but cannot be independently rerun from the public release because its adjusted subreddit residual means are unavailable.
