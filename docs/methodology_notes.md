# Methodology Notes

This document provides implementation-level methodological notes for the study:

**“Expertise-Tiered AI Discourse on Reddit: A Computational Linguistic Analysis of Community-Rewarded Register Variation.”**

It is intended to complement the manuscript and make the computational workflow easier to inspect and reproduce.

## Study Design

The study analyzes linguistic variation across nine Reddit communities grouped into three expected-audience expertise tiers.

The tiers are assigned at the **subreddit level**, not the individual-user level.

The analysis therefore concerns **community-rewarded register variation associated with expected audience expertise**. It does not establish verified author expertise or causal within-author audience adaptation.

## Sample

The final dataset contains:

- 1,778 posts
- 9 subreddits
- 3 expertise tiers

Final tier counts:

- Low: 598
- Medium: 592
- High: 588

Each subreddit contributes approximately 192–200 posts.

## Sampling Strategy

Posts were collected from approximately April 2025 to April 2026.

The scraper prioritized highly scored posts from overlapping Reddit retrieval windows:

1. one year;
2. one month;
3. one week.

Shorter windows were used to supplement the one-year results when additional eligible unique posts were required.

Duplicate post IDs and duplicate cleaned texts were removed.

Posts were excluded if they were:

- shorter than 50 words after preprocessing;
- deleted or removed;
- identified as likely bot-generated using a rule-based heuristic.

Because the study prioritizes highly scored posts, the sample is interpreted as **community-rewarded discourse** rather than a representative sample of all subreddit activity.

## Expertise-Tier Assignment

The nine communities were grouped as follows:

### Low

- r/explainlikeimfive
- r/Futurology
- r/GenerativeAI

### Medium

- r/ChatGPT
- r/learnmachinelearning
- r/OpenAI

### High

- r/LocalLLaMA
- r/MachineLearning
- r/deeplearning

Tiering was based on community-level expectations, including technical vocabulary, discussion norms, and the level of assumed domain familiarity.

These labels are operational categories used for the study and should not be interpreted as direct measures of the expertise of individual authors.

## Text Processing

Post titles were retained as metadata but excluded from the linguistic analysis.

The analysis operates on cleaned post bodies.

The exact preprocessing implementation will be documented in the corresponding analysis script once the canonical source code is finalized.

## Linguistic Features

The primary feature set includes:

### Readability and Surface Structure

- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG Index
- average sentence length
- average syllables per word

### Lexical Diversity

- Type-Token Ratio
- Moving Average Type-Token Ratio with a 25-word window

### Discourse Markers

- hedge words per 100 words
- causal connectives per 100 words

### Sentiment

VADER sentiment outputs:

- positive
- negative
- neutral
- compound

Sentiment analyses are treated as secondary and exploratory.

## Primary Statistical Tests

For each linguistic metric, the primary omnibus comparison across the three tiers uses the Kruskal-Wallis test.

Where appropriate, pairwise comparisons are performed using Dunn post hoc tests with Bonferroni correction.

High-versus-low standardized differences are summarized using Cohen's d.

## Multiple-Testing Correction

Benjamini-Hochberg false discovery rate correction is applied separately to families of related tests, including:

- omnibus Kruskal-Wallis tests;
- Dunn pairwise comparisons;
- partial correlations.

The interpretation prioritizes findings that remain robust after correction.

## Length-Controlled Analysis

Because post length can influence several linguistic measures, partial correlations are used to evaluate associations between expertise tier and linguistic features while controlling for word count.

Additional robustness analyses also control simultaneously for:

- word count;
- acronym density.

## Acronym Robustness

Technical AI communities may use acronyms more frequently, which could influence syllable-based metrics.

To test this possibility, the analysis includes:

- acronym density per 100 words;
- recalculation of syllables per word after detected acronyms are removed;
- recalculation after detected acronyms are normalized to a one-syllable placeholder;
- partial correlations controlling for acronym density.

The acronym detector is rule-based and does not capture every possible technical abbreviation, particularly mixed-case names.

## Leave-One-Subreddit-Out Analysis

The 1,778 observations are nested within only nine subreddit communities.

To assess whether the main findings depend excessively on a single subreddit, the analysis repeats the central comparisons nine times, excluding one subreddit in each run.

This procedure is used as a sensitivity analysis.

It does not eliminate the underlying nested structure or convert the nine communities into statistically independent post-level experimental units.

## Classification Analysis

Machine-learning models are used as supplementary evidence that the measured linguistic features contain information associated with expertise tier.

Models include:

- Logistic Regression
- Random Forest

The models use computed linguistic features only.

The following are excluded from the feature matrix:

- subreddit name
- raw text
- post title
- score
- number of comments
- timestamp
- other direct metadata

The reported analyses include:

- three-class low / medium / high classification;
- binary high-versus-low classification;
- shuffled-label testing.

The classification analysis is not interpreted as a tool for inferring the expertise of individual Reddit users.

## Important Statistical Boundary

The largest methodological limitation is the nested sampling structure.

Although there are 1,778 posts, the expertise-tier variable is assigned at the level of only nine communities.

Accordingly, post-level statistical significance should not be interpreted as equivalent to having 1,778 independent community-level observations.

The leave-one-subreddit-out analysis is included to assess stability across communities, but it does not fully resolve this issue.

## Reproducibility

The repository is being organized so that:

1. canonical de-identified data are stored in `data/`;
2. analysis scripts are stored in `analysis/`;
3. machine-readable numerical outputs are stored in `results/`;
4. manuscript figures are generated into `figures/`.

Where possible, each reported number in the manuscript should be traceable to a saved result file or reproducible analysis script.

## Current Repository Status

The repository structure and documentation are being finalized prior to public release.

Any placeholder or incomplete files should be replaced with canonical source data and analysis code before the repository is cited as fully reproducible.
