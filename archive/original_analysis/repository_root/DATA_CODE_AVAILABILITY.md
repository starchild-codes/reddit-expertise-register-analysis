# Data and code availability

This repository contains the analysis code, public subreddit-to-tier mapping,
numerical results, figures, dependency specification, and methodological
documentation for the study *Expertise-Tiered AI Discourse on Reddit*.

## Data included in the repository

The repository includes the tier mapping and the derived statistical outputs
needed to inspect the reported analyses. It does not include the 1,778
text-bearing Reddit posts or the text-bearing intermediate datasets generated
from them.

## Why the posts are not redistributed

Reddit posts are user-generated text and can remain searchable or identifying
even after usernames and other direct identifiers are removed. The repository
therefore does not redistribute the full post-text dataset merely to make the
repository appear self-contained. The local file named
`reddit_posts_deidentified.csv` is also not treated as automatically anonymous;
it still contains cleaned post text.

## Reproducibility boundary

The analysis is reproducible conditional on access to the historical working
dataset. A fresh clone can be used to inspect the released results, read the
full pipeline, install the pinned dependencies, and rerun the pipeline once a
compatible `data/reddit_posts_working.csv` is supplied. A fresh clone alone
cannot recreate all 1,778 observations because the text-bearing input is not
included.

The required input schema, preprocessing steps, feature definitions, analysis
order, random seeds, statistical procedures, and output locations are
documented in `README.md`, `data/README.md`, and
`docs/methodology_notes.md`.

## Access and reuse

The source dataset may be requested from the study authors for legitimate
research purposes, subject to applicable platform terms, privacy considerations,
and the authors' ability to share it. The repository makes no promise that
full-text access will be granted. Researchers who cannot obtain the source
dataset can still use the released code, mapping, numerical outputs, figures,
and documentation to audit the reported analyses.

The analysis code is released under the MIT License. The data and derived files
remain subject to the terms and permissions described above and should not be
redistributed without checking those constraints.

## Versioning

For a paper submission, cite an immutable release or commit rather than the
moving `main` branch. If the repository is archived with a DOI service, add
that DOI to `CITATION.cff` and to the manuscript's data and code availability
statement.
