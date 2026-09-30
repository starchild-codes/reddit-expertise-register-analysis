# Current analysis utilities

These utilities operate on preserved exploratory outputs. They do not perform
a new corpus collection, recompute linguistic features, or refit the original
post-level models. The external-rater script reproducibly runs a new
subreddit-level, fixed-size permutation sensitivity analysis from released
subreddit means.

| Script | Purpose | Runnable from public clone? |
| --- | --- | --- |
| `00_prepare_release_outputs.py` | Creates the de-identified AI-screen decision file from an authorized private audit file, stripping text before output. | No — requires the private source file. |
| `01_validate_preserved_outputs.py` | Validates corpus counts and manuscript-critical values in preserved CSVs. | Yes — standard library only. |
| `02_build_manuscript_tables.py` | Joins primary and adjusted preserved results into the manuscript-facing Table 2 CSV. | Yes — standard library only. |
| `03_render_current_figures.py` | Renders six current figures from the aggregate CSVs. | Yes — install `requirements.txt`. |
| `04_external_rater_sensitivity.py` | Computes agreement and exact fixed-size external-tier sensitivity results from released subreddit means. | Yes — standard library only. |
| `05_validate_rater_sensitivity.py` | Validates external assignments, agreement, permutation counts, and FDR results. | Yes — standard library only. |

The inability to regenerate the historical feature and inferential pipeline is
a data-availability constraint, not a claim of full raw-data reproducibility.
