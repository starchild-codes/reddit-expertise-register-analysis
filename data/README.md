# Data

This directory contains the de-identified data used for the analyses reported in:

**“Expertise-Tiered AI Discourse on Reddit: A Computational Linguistic Analysis of Community-Rewarded Register Variation.”**

## Files

### `subreddit_tier_mapping.csv`

Maps each of the nine sampled subreddits to its expected-audience expertise tier.

The three tiers are:

- `low`
- `medium`
- `high`

Tier assignments are community-level labels and should not be interpreted as verified expertise labels for individual Reddit users.

### `reddit_posts_deidentified.csv`

This file will contain the canonical post-level dataset used for the analysis.

Expected fields include:

- `post_id`
- `subreddit`
- `tier`
- `tier_code`
- `created_utc`
- `score`
- `num_comments`
- `clean_text`
- `word_count`

The final dataset contains **1,778 posts** from **nine subreddits**.

## De-identification

Usernames and other unnecessary personally identifying information are not included in the released dataset.

Only variables necessary for reproducing the reported analyses are retained.

Post titles are retained as metadata only if required for reproducibility, but they are not included in the linguistic analysis.

## Sampling

Posts were collected from approximately April 2025 to April 2026.

The sampling strategy prioritized highly scored posts using overlapping Reddit retrieval windows:

- one year
- one month
- one week

Shorter retrieval windows were used to supplement the one-year results when additional eligible unique posts were required.

Duplicate post IDs and duplicate cleaned texts were removed.

Posts were also excluded if they were:

- shorter than 50 words after preprocessing;
- deleted or removed;
- identified as likely bot-generated using the study's rule-based filtering procedure.

The dataset should therefore be interpreted as a sample of **community-rewarded discourse**, rather than a representative sample of all posts made in each subreddit.

## Subreddit Counts

The final tier totals are:

- Low expertise tier: **598**
- Medium expertise tier: **592**
- High expertise tier: **588**

Each subreddit contributes approximately **192–200 posts**.

## Important Design Boundary

The post-level observations are nested within only nine subreddit communities.

The study therefore distinguishes between the large number of post-level observations and the smaller number of community-level units.

Leave-one-subreddit-out robustness analyses are included elsewhere in the repository to assess whether the central findings depend strongly on any single community.

## Ethical Use

This dataset is provided for research reproducibility and methodological inspection.

It should not be used to identify, profile, or evaluate individual Reddit users.
