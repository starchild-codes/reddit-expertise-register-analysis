# Expertise-Tiered AI Discourse on Reddit

Computational linguistic analysis of **1,778 highly scored Reddit posts** from nine communities grouped by expected-audience expertise.

The repository contains the full analysis pipeline, reproducibility outputs, robustness checks, classification experiments, and generated figures. The text-bearing input data are intentionally not redistributed; see [DATA_CODE_AVAILABILITY.md](DATA_CODE_AVAILABILITY.md).

The analyses are reproducible conditional on access to the historical working dataset. A fresh clone can inspect the released results and rerun the pipeline after a compatible local input file is supplied, but cannot recreate all 1,778 observations from the repository alone.

---

## Overview

The dataset contains posts from nine subreddits grouped into three community-level expertise tiers.

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

The tiers represent **subreddit-level audience and communication expectations**, not verified expertise of individual users.

Final tier counts:

| Tier | Posts |
|---|---:|
| Low | 598 |
| Medium | 592 |
| High | 588 |
| **Total** | **1,778** |

---

## Main Results

Average syllables per word showed the strongest tier difference.

| Tier | Mean syllables/word |
|---|---:|
| Low | 1.534 |
| Medium | 1.536 |
| High | 1.626 |

- Kruskal-Wallis: **H = 89.90, p < .001**
- High vs. low Cohen's d: **0.521**
- Controlling for word count: **partial r = .197, p < .001**
- Controlling for word count + acronym density: **partial r = .168, p < .001**

### Acronym robustness

After removing detected acronyms:

- **H = 99.10**
- **d = 0.558**

After replacing detected acronyms with the one-syllable token `term`:

- **H = 99.88**
- **d = 0.558**

The final acronym detector is:

```python
r"\b[A-Z]{2,}(?:-?[A-Z0-9]+)*\b"
```

Post-level transformed values are retained at full precision before statistical testing.

---

## Linguistic Features

The analysis includes:

### Readability

- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG
- average sentence length
- average syllables per word

### Lexical diversity

- Type-Token Ratio
- MATTR with a 25-word window

### Discourse features

- hedge terms per 100 words
- causal connectives per 100 words

### Sentiment

VADER:

- positive
- negative
- neutral
- compound

### Additional variables

- word count
- acronym count
- acronym density
- syllables per word after acronym removal
- syllables per word after acronym normalization

---

## Statistical Analysis

The repository implements:

- Kruskal-Wallis omnibus tests
- Dunn pairwise tests with Bonferroni correction
- high-vs-low Cohen's d
- partial Pearson correlations
- Benjamini-Hochberg FDR correction
- acronym-removal robustness analysis
- acronym-normalization robustness analysis
- leave-one-subreddit-out sensitivity analysis

The high-vs-low effect-size convention used throughout the repository is:

```text
(mean_high - mean_low) /
sqrt((variance_high + variance_low) / 2)
```

---

## Leave-One-Subreddit-Out Analysis

The central analyses are repeated after removing each subreddit individually.

The current LOSO pipeline generates:

```text
results/leave_one_subreddit_out_all_metrics.csv
results/leave_one_subreddit_out_partial_correlations.csv
results/leave_one_subreddit_out_influential_subreddits.csv
```

---

## Classification

The repository includes supplementary classification experiments using computed linguistic features.

Models:

- Logistic Regression
- Random Forest

The models do not use raw post text, subreddit identity, titles, scores, timestamps, or user information.

### Current results

| Task | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Three-class | Logistic Regression | 0.458 | 0.449 |
| Three-class | Random Forest | 0.404 | 0.405 |
| High vs. Low | Logistic Regression | 0.647 | 0.644 |

Mean accuracy across 100 shuffled-label Logistic Regression runs:

**0.335**

---

## Repository Structure

```text
reddit-expertise-register-analysis/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── DATA_CODE_AVAILABILITY.md
├── requirements.txt
├── .gitignore
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
├── data/
│   ├── README.md
│   └── subreddit_tier_mapping.csv
│
├── results/
│   ├── primary_statistics.csv
│   ├── posthoc_dunn_bonferroni.csv
│   ├── partial_correlations.csv
│   ├── fdr_results.csv
│   ├── acronym_robustness.csv
│   ├── acronym_robustness_dunn.csv
│   ├── acronym_robustness_partial_correlations.csv
│   ├── leave_one_subreddit_out_all_metrics.csv
│   ├── leave_one_subreddit_out_partial_correlations.csv
│   ├── leave_one_subreddit_out_influential_subreddits.csv
│   ├── classifier_results.csv
│   ├── classifier_confusion_matrix.csv
│   ├── classifier_shuffle_baseline.csv
│   ├── random_forest_feature_importance.csv
│
├── figures/
│   ├── figure_01_word_count_by_tier.png
│   ├── figure_02_syllables_by_tier.png
│   ├── figure_03_fk_grade_distribution.png
│   ├── figure_04_acronym_robustness.png
│   ├── figure_05_loso_cohens_d.png
│   ├── figure_06_hedging_by_tier.png
│   ├── figure_07_sentiment_by_subreddit.png
│   ├── figure_08_subreddit_feature_heatmap.png
│   ├── figure_09_partial_correlations.png
│   ├── figure_10_random_forest_importance.png
│   └── supplementary_loso_kruskal_h.png
│
└── docs/
    └── methodology_notes.md
```

---

## Data Files

The text-bearing working and intermediate datasets are intentionally not committed to the repository.

The following files are ignored through `.gitignore`:

```text
data/reddit_posts_working.csv
data/reddit_posts_deidentified.csv
data/reddit_posts_features.csv
data/reddit_posts_features_acronyms.csv
```

The committed repository contains the numerical outputs required to inspect the analyses without including the full post-text dataset. The absence of the 1,778 posts is intentional: cleaned Reddit text can remain searchable or identifying even after direct identifiers are removed.

---

## Reproducing the Analysis

The finalized pipeline was successfully rerun using:

```text
Python 3.12.3
```

### 1. Create an environment

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Add the local working dataset

Place the processed dataset at:

```text
data/reddit_posts_working.csv
```

Required source columns:

```text
subreddit
expertise_tier
created_utc
score
num_comments
word_count
text_clean
```

The repository does not include this file. See [DATA_CODE_AVAILABILITY.md](DATA_CODE_AVAILABILITY.md) for the reproducibility boundary and the conditions under which the historical dataset may be requested.

### 3. Run the pipeline

Use this order:

```bash
python analysis/01_preprocessing.py
python analysis/02_linguistic_features.py
python analysis/03_primary_statistics.py
python analysis/04_posthoc_tests.py
python analysis/07_acronym_robustness.py
python analysis/05_partial_correlations.py
python analysis/06_fdr_correction.py
python analysis/08_leave_one_subreddit_out.py
python analysis/09_classification.py
python analysis/10_generate_figures.py
```

`07_acronym_robustness.py` runs before `05_partial_correlations.py` because the latter uses acronym-derived variables generated by `07`.

Generated outputs are written to:

```text
results/
figures/
```

---

## Main Output Files

### Primary statistics

```text
results/primary_statistics.csv
results/posthoc_dunn_bonferroni.csv
results/partial_correlations.csv
results/fdr_results.csv
```

### Acronym robustness

```text
results/acronym_robustness.csv
results/acronym_robustness_dunn.csv
results/acronym_robustness_partial_correlations.csv
```

### Leave-one-subreddit-out

```text
results/leave_one_subreddit_out_all_metrics.csv
results/leave_one_subreddit_out_partial_correlations.csv
results/leave_one_subreddit_out_influential_subreddits.csv
```

### Classification

```text
results/classifier_results.csv
results/classifier_confusion_matrix.csv
results/classifier_shuffle_baseline.csv
results/random_forest_feature_importance.csv
```

---

## Dependencies

The reproducibility environment is pinned in:

```text
requirements.txt
```

Key packages include:

- pandas
- NumPy
- SciPy
- scikit-learn
- statsmodels
- matplotlib
- textstat
- scikit-posthocs
- vaderSentiment
- lexicalrichness
- pingouin

---

## Additional Documentation

Implementation-level details are available in:

```text
docs/methodology_notes.md
```

The subreddit-to-tier mapping is stored in:

```text
data/subreddit_tier_mapping.csv
```

Citation metadata are stored in:

```text
CITATION.cff
```

Data and code availability details are documented in:

```text
DATA_CODE_AVAILABILITY.md
```

---

## Notes

The 1,778 posts are nested within nine subreddit communities, and expertise tier is assigned at the subreddit level. Two low-tier communities, r/explainlikeimfive and r/Futurology, are broader than AI-specific discussion. These design limitations are documented in the repository notes.

---

## License

See `LICENSE`.
