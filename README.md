# Expertise-Tiered AI Discourse on Reddit

A computational linguistic analysis of community-rewarded register variation across Reddit communities with different expected audience expertise levels.

This repository contains the analysis code, reproducibility outputs, and documentation for the research paper:

**Expertise-Tiered AI Discourse on Reddit: A Computational Linguistic Analysis of Community-Rewarded Register Variation**

## Overview

This study examines how linguistic register differs across Reddit communities that expect different levels of background knowledge from their audiences.

The analysis uses **1,778 highly scored Reddit posts** from **nine subreddits**, grouped into three community-level expertise tiers:

### Low expertise
- r/explainlikeimfive
- r/Futurology
- r/GenerativeAI

### Medium expertise
- r/ChatGPT
- r/learnmachinelearning
- r/OpenAI

### High expertise
- r/LocalLLaMA
- r/MachineLearning
- r/deeplearning

The tier labels describe the **expected audience and communication norms of each subreddit**, not the verified expertise of individual Reddit users.

Two low-tier communities, r/explainlikeimfive and r/Futurology, cover broader subject matter rather than AI alone. Their retained posts were not independently validated for topic equivalence. The paper therefore treats topic mix and general-audience writing as important limitations and potential confounds.

---

## Research Question

**How do community-rewarded linguistic patterns differ across Reddit communities with different expected audience expertise levels, and which features distinguish those tiers most reliably?**

The study focuses on whether differences appear primarily in:

- readability
- lexical complexity
- sentence length
- lexical diversity
- hedging
- causal connectives
- sentiment
- post length

It also evaluates whether these features collectively contain enough information to distinguish expertise tiers using machine-learning classifiers.

---

## Main Finding

The clearest linguistic difference across the expertise tiers was **average syllables per word**, used as a proxy for lexical complexity and technical-register density.

Across the full dataset:

- Low tier: **1.534 syllables/word**
- Medium tier: **1.536 syllables/word**
- High tier: **1.626 syllables/word**
- Kruskal-Wallis: **H = 89.90, p < .001**
- High vs. low Cohen's d: **0.521**

The result remained after:

- controlling for post length
- controlling jointly for post length and acronym density
- removing detected acronyms
- replacing acronyms with one-syllable placeholders
- Benjamini-Hochberg false-discovery-rate correction
- leave-one-subreddit-out analysis

After controlling for word count and acronym density, average syllables per word remained associated with expertise tier:

**partial r = .168, p < .001**

When detected acronyms were removed, the high-versus-low effect remained:

**d = .558**

The leave-one-subreddit-out analysis preserved the central pattern after removal of every subreddit. Removing r/explainlikeimfive produced the largest reduction in the high-versus-low effect, lowering Cohen's d from **0.521 to approximately 0.237**, showing that its explicit simplification norms strengthen the contrast without fully creating it.

---

## Dataset

The final dataset contains:

- **1,778 posts**
- **598 low-tier posts**
- **592 medium-tier posts**
- **588 high-tier posts**
- **9 subreddits**
- approximately 192–200 posts per subreddit
- posts collected between **April 2025 and April 2026**

The sampling strategy prioritized highly scored posts because the study investigates **community-rewarded discourse**, rather than average posting behavior.

Posts shorter than 50 words were excluded.

Titles were retained as metadata during collection but excluded from linguistic feature calculations.

### Important dataset note

The original working dataset contains Reddit post text and should not automatically be treated as suitable for public redistribution.

The raw working file is therefore intended to remain local/private unless redistribution and privacy considerations have been reviewed.

Generated analysis outputs can be reproduced from the working dataset using the scripts in this repository.

---

## Linguistic Measures

The analysis includes the following measures.

### Readability

- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG
- average sentence length
- average syllables per word

### Lexical diversity

- Type-Token Ratio (TTR)
- Moving Average Type-Token Ratio (MATTR), window size 25

### Discourse features

- hedge terms per 100 words
- causal connectives per 100 words

### Sentiment

VADER sentiment scores:

- positive
- negative
- neutral
- compound

### Additional variables

- word count
- acronym count
- acronym density per 100 words
- syllables per word after acronym removal
- syllables per word after acronym normalization

---

## Statistical Analysis

### Kruskal-Wallis tests

Each linguistic feature is compared across the low, medium, and high expertise tiers using a Kruskal-Wallis test.

### Dunn post hoc tests

Significant omnibus results are followed by Dunn pairwise comparisons with Bonferroni correction.

The three comparisons are:

- low vs. medium
- low vs. high
- medium vs. high

### Effect sizes

Cohen's d is calculated for the high-versus-low contrast.

### Partial correlations

Partial correlations examine the relationship between expertise tier and linguistic features after controlling for post length.

The main lexical-complexity robustness analysis also controls simultaneously for:

- word count
- acronym density

### False Discovery Rate

Benjamini-Hochberg false-discovery-rate correction is used to account for multiple testing.

Correction families are handled separately for:

- Kruskal-Wallis omnibus tests
- Dunn metric-by-tier comparisons
- partial correlations

### Acronym robustness

Technical communities use more abbreviations, so the main lexical-complexity result is tested under three conditions:

1. original average syllables per word
2. detected acronyms removed
3. detected acronyms replaced with a one-syllable placeholder

Acronym density is also included as a statistical control.

### Leave-One-Subreddit-Out Analysis

The complete analysis is repeated after removing each subreddit individually.

This tests whether a single community is responsible for the observed tier differences.

Two LOSO output files are provided:

- `leave_one_subreddit_out_primary_syllables.csv`  
  Compact output focused on the main average-syllables-per-word result.

- `leave_one_subreddit_out_all_metrics.csv`  
  Extended LOSO output covering lexical, readability, sentiment, acronym, hedging, and causal-connective measures.

---

## Machine-Learning Analysis

Two models are used:

- Logistic Regression
- Random Forest

The models use only computed linguistic features.

They do **not** use:

- subreddit names
- titles
- raw post text
- Reddit scores
- comment counts
- timestamps
- user information

The primary task predicts three expertise tiers:

- low
- medium
- high

A supplementary binary task compares only:

- low
- high

The three-class Logistic Regression model achieves approximately **45% accuracy**, compared with a shuffled-label baseline of approximately **34%**.

The binary high-versus-low classifier reaches approximately **65% accuracy**.

These models are included as supporting evidence that the linguistic variables contain non-random tier information. They are **not intended as systems for inferring the expertise of individual Reddit users or posts**.

---

## Repository Structure

```text
reddit-expertise-register-analysis/
│
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
│
├── data/
│   ├── README.md
│   ├── subreddit_tier_mapping.csv
│   ├── reddit_posts_working.csv
│   ├── reddit_posts_deidentified.csv
│   ├── reddit_posts_features.csv
│   └── reddit_posts_features_acronyms.csv
│
├── analysis/
│   ├── 01_preprocessing.py
│   ├── 02_linguistic_features.py
│   ├── 03_primary_statistics.py
│   ├── 04_posthoc_tests.py
│   ├── 05_partial_correlations.py
│   ├── 06_fdr_correction.py
│   ├── 07_acronym_robustness.py
│   ├── 08_leave_one_subreddit_out.py
│   ├── 09_classification.py
│   └── 10_generate_figures.py
│
├── results/
│   ├── primary_statistics.csv
│   ├── posthoc_dunn_bonferroni.csv
│   ├── partial_correlations.csv
│   ├── fdr_results.csv
│   ├── acronym_robustness.csv
│   ├── leave_one_subreddit_out_primary_syllables.csv
│   ├── leave_one_subreddit_out_all_metrics.csv
│   ├── classifier_results.csv
│   ├── classifier_shuffle_baseline.csv
│   ├── random_forest_feature_importance.csv
│   └── data_topicality_audit.csv
│
├── figures/
│
└── docs/
    └── methodology_notes.md
