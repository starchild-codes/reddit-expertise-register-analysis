# Current publication figures

The six PNG files in this directory are the current, publication-quality
figures for the exploratory reanalysis. They are rendered by
`analysis/current/03_render_current_figures.py` from the released aggregate
CSV files in `results/current/`; no post text, feature extraction, model
fitting, or inferential recomputation occurs during rendering.

| Figure | File | Source data |
| --- | --- | --- |
| 1 | `figure1_ai_relevance_retention.png` | `table1_subreddit_composition_ai_audit.csv` |
| 2 | `figure2_average_syllables_subreddit_means.png` | `table2_subreddit_linguistic_descriptives.csv` |
| 3 | `figure3_cluster_permutation_fdr.png` | `table3_exact_cluster_permutation_fdr.csv` |
| 4 | `figure4_topic_distribution_by_tier.png` | `topic_distribution_by_tier.csv` |
| 5 | `figure5_logistic_grouped_confusion_matrix.png` | `classifier_grouped_predictions.csv` |
| 6 | `figure6_leave_one_subreddit_out.png` | `classifier_logistic_leave_one_subreddit_out.csv` |

The figures are descriptive displays of a historical exploratory analysis.
They do not warrant causal or individual-level expertise claims.
