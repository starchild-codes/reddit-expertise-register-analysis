# Manuscript Repository Consistency

The final submitted manuscript is the controlling source. All values below were checked against its abstract, Methods, Results, tables, and figure captions.

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
| Results 3.4, Table 3 | Logistic accuracy=.3549, balanced accuracy=.3336, macro-F1=.3355 | `results/current/table5_grouped_classifier_performance_permutation.csv` | `logistic_regression` row | MATCH |
| Results 3.4, Table 3 | Random forest accuracy=.2394, balanced accuracy=.1998, macro-F1=.1880 | `results/current/table5_grouped_classifier_performance_permutation.csv` | `random_forest` row | MATCH |
| Results 3.4, Table 3 | Subreddit-label permutation p-values | `results/current/table5_grouped_classifier_performance_permutation.csv` | permutation p columns | MATCH |
| Figures 1-6 | Values, axes, and captioned content | `figures/current/` plus corresponding CSV above | named figure and CSV | MATCH |

No unexplained numerical mismatch was found. Historical output recomputation is not claimed because the repository does not contain original post bodies or the post-level feature matrix.
