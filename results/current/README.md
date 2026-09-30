# Current results

This directory is the canonical, manuscript-matched release of aggregate
outputs for the exploratory existing-corpus reanalysis. The controlling
manuscript is `docs/Final_Revised_Reddit_Manuscript_Convergence.docx`.

The current results are preserved outputs rather than a rerun from raw post
bodies: the released package intentionally excludes post text, titles,
usernames, and other potentially identifying content. See
`DATA_CODE_AVAILABILITY.md` and `environment_and_provenance.json` for limits.

Start with:

- `table1_subreddit_composition_ai_audit.csv` — corpus and AI-screen counts.
- `table2_cluster_permutation_results.csv` — exact permutation and
  topic/length-adjusted results joined by outcome.
- `table5_grouped_classifier_performance_permutation.csv` — grouped held-out
  classifier metrics and subreddit-label permutation tests.
- `topic_overlap_by_tier.csv` and `topic_terms_labels.csv` — topic-control
  evidence and labels.
