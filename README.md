# Expertise-Tiered AI Discourse on Reddit

A computational linguistic analysis of community-rewarded register variation across Reddit communities with different expected audience expertise levels.

This repository contains the analysis code, numerical outputs, figures, and methodological documentation for the research paper:

**Expertise-Tiered AI Discourse on Reddit: A Computational Linguistic Analysis of Community-Rewarded Register Variation**

## Overview

This study examines how linguistic register differs across Reddit communities that expect different levels of background knowledge from their audiences.

The final sample contains **1,778 highly scored Reddit posts** from **nine subreddits**, grouped into three community-level expected-audience expertise tiers.

### Low expected-audience expertise

- r/explainlikeimfive
- r/Futurology
- r/GenerativeAI

### Medium expected-audience expertise

- r/ChatGPT
- r/learnmachinelearning
- r/OpenAI

### High expected-audience expertise

- r/LocalLLaMA
- r/MachineLearning
- r/deeplearning

The tier labels describe the **expected audience and communication norms of each subreddit**. They are not verified expertise labels for individual authors.

Two low-tier communities, r/explainlikeimfive and r/Futurology, cover broader subject matter rather than AI alone. The retained posts were not independently hand-validated for topical equivalence across subreddits. Topic mix and general-audience writing are therefore treated as important limitations and potential confounds.

---

## Research Question

**How do community-rewarded linguistic patterns differ across Reddit communities with different expected audience expertise levels, and which features distinguish those tiers most reliably?**

The study examines:

- readability;
- lexical complexity;
- sentence length;
- lexical diversity;
- hedging;
- causal connectives;
- sentiment;
- post length;
- acronym use.

A supplementary machine-learning analysis tests whether the computed linguistic features collectively contain non-random information about subreddit expertise tier.

---

## Main Result

The clearest tier difference occurs in **average syllables per word**, used as a proxy for lexical complexity and technical-register density.

Across the full dataset:

- Low tier: **1.534 syllables/word**
- Medium tier: **1.536 syllables/word**
- High tier: **1.626 syllables/word**
- Kruskal-Wallis: **H = 89.90, p < .001**
- High-vs-low Cohen's d: **0.521**

The association remains after controlling for post length:

- **partial r = .197, p < .001**

It also remains after controlling jointly for post length and acronym density:

- **partial r = .168, p < .001**

### Acronym robustness

Because technical communities use more acronyms, average syllables per word was recalculated after acronym transformation.

After detected acronyms were removed:

- Low: **1.528**
- Medium: **1.534**
- High: **1.631**
- **H = 99.10, p < .001**
- High-vs-low **d = .558**

After detected acronyms were normalized to the one-syllable placeholder `term`:

- Low: **1.526**
- Medium: **1.531**
- High: **1.628**
- **H = 99.88, p < .001**
- High-vs-low **d = .558**

The implementation retains full-precision post-level transformed values before statistical testing; values are rounded only for presentation.

### Leave-one-subreddit-out robustness

The central pattern remains after excluding each subreddit individually.

Removing r/explainlikeimfive produces the largest reduction in the high-vs-low syllable effect, showing that its explicit simplification norm strengthens the overall contrast, while the broader tier pattern remains detectable without it.

---

## Dataset

The final analysis sample contains:

- **1,778 posts**
- **598 low-tier posts**
- **592 medium-tier posts**
- **588 high-tier posts**
- **9 subreddits**
- approximately **192–200 posts per subreddit**
- posts collected from approximately **April 2025 to April 2026**

The sampling strategy prioritized highly scored posts because the study investigates **community-rewarded discourse**, not representative average posting behavior.

Posts shorter than 50 words after preprocessing were excluded.

Post titles were retained during the historical collection workflow but are excluded from linguistic feature calculations.

### Data availability and privacy

The historical working dataset contains Reddit post text and is intentionally excluded from version control.

The following local/generated text-bearing files are ignored by Git:

```text
data/reddit_posts_working.csv
data/reddit_posts_deidentified.csv
data/reddit_posts_features.csv
data/reddit_posts_features_acronyms.csv
