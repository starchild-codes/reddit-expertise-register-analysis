# Expertise-Tiered AI Discourse on Reddit

A computational linguistic analysis of community-rewarded register variation across Reddit communities with different expected audience expertise levels.

This repository contains the analysis code, numerical outputs, figures, and methodological documentation for the research paper:

**Expertise-Tiered AI Discourse on Reddit: A Computational Linguistic Analysis of Community-Rewarded Register Variation**

---

## Overview

This study examines whether linguistic register varies systematically across Reddit communities that expect different levels of background knowledge from their audiences.

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

The tier labels describe the **expected audience and communication norms of each subreddit**. They are not verified expertise labels for individual Reddit users.

Two low-tier communities, r/explainlikeimfive and r/Futurology, cover broader subject matter rather than AI alone. The retained posts were not independently hand-validated for topical equivalence across subreddits. Topic composition and general-audience writing therefore remain important potential confounds.

---

## Research Question

**How do community-rewarded linguistic patterns differ across Reddit communities with different expected audience expertise levels, and which features distinguish those tiers most reliably?**

The study evaluates differences in:

- readability
- lexical complexity
- sentence length
- lexical diversity
- hedging
- causal connectives
- sentiment
- post length
- acronym use

A supplementary machine-learning analysis tests whether the computed linguistic features collectively contain non-random information about subreddit expertise tier.

---

## Main Finding

The clearest linguistic difference across expertise tiers is **average syllables per word**, used as a proxy for lexical complexity and technical-register density.

Across the full dataset:

| Tier | Mean syllables/word |
|---|---:|
| Low | 1.534 |
| Medium | 1.536 |
| High | 1.626 |

**Kruskal-Wallis H = 89.90, p < .001**

**High-vs-low Cohen's d = 0.521**

The relationship remains after controlling for post length:

**partial r = .197, p < .001**

It also remains after controlling jointly for word count and acronym density:

**partial r = .168, p < .001**

These results indicate that the higher lexical-complexity pattern is not explained solely by longer posts or greater acronym use.

---

## Acronym Robustness

Technical AI communities use more abbreviations and acronyms, which could artificially increase syllable-based complexity measures.

The analysis therefore recalculates average syllables per word under two additional conditions.

### Acronyms removed

| Tier | Mean syllables/word |
|---|---:|
| Low | 1.528 |
| Medium | 1.534 |
| High | 1.631 |

**H = 99.10, p < .001**

**High-vs-low d = 0.558**

### Acronyms normalized

Detected acronyms are replaced with the one-syllable placeholder `term`.

| Tier | Mean syllables/word |
|---|---:|
| Low | 1.526 |
| Medium | 1.531 |
| High | 1.628 |

**H = 99.88, p < .001**

**High-vs-low d = 0.558**

The final implementation retains full-precision post-level transformed values before statistical testing; rounding is used only for presentation.

The acronym detector used in the final robustness analysis is:

```python
r"\b[A-Z]{2,}(?:-?[A-Z0-9]+)*\b"
