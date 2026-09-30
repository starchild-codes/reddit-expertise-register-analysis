"""Render the six publication figures from preserved aggregate CSV outputs.

This is a rendering-only script: it does not recalculate linguistic features,
fit models, or conduct inference.  It makes the current figures traceable to
the released, manuscript-matched tables.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "current"
FIGURES = ROOT / "figures" / "current"
TIERS = ("low", "medium", "high")
COLORS = {"low": "#4C78A8", "medium": "#F58518", "high": "#54A24B"}
LABELS = {
    "fk_grade": "Flesch–Kincaid grade",
    "gunning_fog": "Gunning Fog",
    "smog": "SMOG",
    "avg_sentence_length": "Mean sentence length",
    "avg_syllables_per_word": "Mean syllables per word",
    "mattr": "MATTR",
    "hedge_per_100": "Hedges per 100 words",
    "causal_per_100": "Causal markers per 100 words",
    "sentiment_compound": "Sentiment compound",
    "sentiment_neg": "Negative sentiment",
}


def rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def save(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=400, bbox_inches="tight")
    plt.close()


def fig1_retention() -> None:
    data = rows("table1_subreddit_composition_ai_audit.csv")
    data.sort(key=lambda row: (TIERS.index(row["expertise_tier"]), row["subreddit"].lower()))
    names = [row["subreddit"] for row in data]
    rates = [100 * float(row["ai_relevance_rate"]) for row in data]
    colors = [COLORS[row["expertise_tier"]] for row in data]
    fig, ax = plt.subplots(figsize=(10, 5.3))
    ax.bar(names, rates, color=colors, edgecolor="white", linewidth=.7)
    ax.set_ylabel("AI-relevant posts retained (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Conservative AI/ML content-screen retention by community")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[t]) for t in TIERS]
    ax.legend(handles, [t.title() + " orientation" for t in TIERS], frameon=False, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    save("figure1_ai_relevance_retention.png")


def fig2_syllables() -> None:
    data = rows("table2_subreddit_linguistic_descriptives.csv")
    data.sort(key=lambda row: (TIERS.index(row["expertise_tier"]), row["subreddit"].lower()))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for tier in TIERS:
        subset = [r for r in data if r["expertise_tier"] == tier]
        x = [data.index(r) for r in subset]
        y = [float(r["avg_syllables_per_word"]) for r in subset]
        ax.scatter(x, y, s=70, color=COLORS[tier], label=tier.title(), zorder=3)
        for xpos, row, ypos in zip(x, subset, y):
            ax.annotate(row["subreddit"], (xpos, ypos), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(data)), [str(i + 1) for i in range(len(data))])
    ax.set_xlabel("Community (ordered by orientation tier)")
    ax.set_ylabel("Subreddit mean syllables per word")
    ax.set_title("Subreddit means for the closest descriptive contrast")
    ax.legend(title="Orientation", frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    save("figure2_average_syllables_subreddit_means.png")


def fig3_permutation() -> None:
    data = rows("table3_exact_cluster_permutation_fdr.csv")
    data.sort(key=lambda row: float(row["exact_p"]))
    y = np.arange(len(data))
    exact = [float(row["exact_p"]) for row in data]
    q = [float(row["fdr_p"]) for row in data]
    labels = [LABELS[row["outcome"]] for row in data]
    fig, ax = plt.subplots(figsize=(9.3, 6.0))
    ax.hlines(y, exact, q, color="#AAB7C4", lw=2, zorder=1)
    ax.scatter(exact, y, color="#2A6FBB", s=60, zorder=3, label="Exact cluster permutation p")
    ax.scatter(q, y, color="#D1495B", s=60, marker="s", zorder=3, label="Benjamini–Hochberg q")
    ax.axvline(.05, color="#555555", lw=1, ls="--", label="0.05 reference")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, .95)
    ax.set_xlabel("Probability")
    ax.set_title("Exact subreddit-level permutation results and FDR adjustment")
    ax.grid(axis="x", alpha=.2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -0.13), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    save("figure3_cluster_permutation_fdr.png")


def fig4_topics() -> None:
    data = rows("topic_distribution_by_tier.csv")
    topic_ids = list(range(8))
    width = .24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, tier in zip((-width, 0, width), TIERS):
        vals = []
        for topic in topic_ids:
            match = next((r for r in data if r["expertise_tier"] == tier and int(r["topic_id"]) == topic), None)
            vals.append(0 if match is None else 100 * float(match["share_within_tier"]))
        ax.bar(np.array(topic_ids) + offset, vals, width=width, color=COLORS[tier], label=tier.title())
    ax.set_xticks(topic_ids, [f"Topic {i + 1}" for i in topic_ids])
    ax.set_ylabel("Share within orientation tier (%)")
    ax.set_title("Broad-topic composition of the AI-screened corpus")
    ax.legend(title="Orientation", frameon=False, ncol=3)
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    save("figure4_topic_distribution_by_tier.png")


def fig5_confusion() -> None:
    data = [r for r in rows("classifier_grouped_predictions.csv") if r["model"] == "logistic_regression"]
    matrix = np.zeros((3, 3), dtype=int)
    for row in data:
        matrix[TIERS.index(row["actual"]), TIERS.index(row["predicted"])] += 1
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if matrix[i, j] > matrix.max() * .55 else "#1F2933")
    ax.set_xticks(range(3), [t.title() for t in TIERS])
    ax.set_yticks(range(3), [t.title() for t in TIERS])
    ax.set_xlabel("Predicted orientation")
    ax.set_ylabel("Observed orientation")
    ax.set_title("Logistic regression: grouped held-out predictions")
    fig.colorbar(image, ax=ax, label="Posts")
    save("figure5_logistic_grouped_confusion_matrix.png")


def fig6_logo() -> None:
    data = [r for r in rows("classifier_logistic_leave_one_subreddit_out.csv") if r["model"] == "logistic_regression"]
    data.sort(key=lambda r: float(r["accuracy"]), reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar([r["held_out_subreddit"] for r in data], [100 * float(r["accuracy"]) for r in data], color="#2A6FBB")
    ax.axhline(100 / 3, color="#555555", ls="--", lw=1, label="Three-class chance reference")
    ax.set_ylim(0, 60)
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("Leave-one-subreddit-out logistic-regression accuracy")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save("figure6_leave_one_subreddit_out.png")


if __name__ == "__main__":
    fig1_retention()
    fig2_syllables()
    fig3_permutation()
    fig4_topics()
    fig5_confusion()
    fig6_logo()
    print("Rendered six current figures from preserved aggregate outputs.")
