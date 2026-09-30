# External rater tier sensitivity

This directory is the canonical transcription and reproducible sensitivity
analysis for two external ratings of the nine historical communities.

The assignments were supplied by the author as previously recorded independent
ratings. Screenshots or other rater-source artifacts are not included in the
public repository. `tier_assignments_three_raters.csv` is the canonical
transcription used here.

The raters did not fully reproduce the author's original three-by-three-by-three
mapping: both moved r/Futurology and r/GenerativeAI from low to medium, and they
differed only on r/ChatGPT. This analysis asks whether the corrected cluster-level
inference depends on one exact tier mapping. It is a robustness check, not proof
of a uniquely correct community taxonomy or construct validity.

`04_external_rater_sensitivity.py` computes agreement statistics and enumerates
all fixed-size tier allocations using the released nine-subreddit descriptive
means. It produces 1,260 allocations for Rater 1's 2/4/3 split and 504 for Rater
2's 1/5/3 split, then applies BH correction across the ten planned outcomes.

The public release does not include the subreddit-level residual means from the
topic- and length-adjusted analysis. Consequently, the manuscript's adjusted
external-mapping sensitivity values cannot be newly recomputed from this clone;
they are documented as a public reproducibility boundary rather than recreated
from insufficient inputs.
