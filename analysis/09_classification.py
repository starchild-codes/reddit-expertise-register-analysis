from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "reddit_posts_features.csv"
)

SUMMARY_OUTPUT_PATH = (
    ROOT
    / "results"
    / "classifier_results.csv"
)

IMPORTANCE_OUTPUT_PATH = (
    ROOT
    / "results"
    / "random_forest_feature_importance.csv"
)

CONFUSION_OUTPUT_PATH = (
    ROOT
    / "results"
    / "classifier_confusion_matrix.csv"
)

SHUFFLE_OUTPUT_PATH = (
    ROOT
    / "results"
    / "classifier_shuffle_baseline.csv"
)


FEATURES = [
    "fk_grade",
    "gunning_fog",
    "smog",
    "avg_sentence_length",
    "avg_syllables_per_word",
    "type_token_ratio",
    "mattr",
    "hedge_per_100",
    "causal_per_100",
    "word_count",
]


def macro_f1_from_report(report_dict):
    return (
        report_dict
        .get("macro avg", {})
        .get("f1-score", np.nan)
    )


def run_three_class_models(df):
    X = df[
        FEATURES
    ].dropna()

    y = df.loc[
        X.index,
        "expertise_tier"
    ]

    label_encoder = LabelEncoder()

    y_encoded = (
        label_encoder
        .fit_transform(y)
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y_encoded,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded,
        )
    )

    rows = []

    # ============================================================
    # LOGISTIC REGRESSION
    # ============================================================

    logistic = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    logistic.fit(
        X_train,
        y_train
    )

    pred_lr = logistic.predict(
        X_test
    )

    lr_accuracy = accuracy_score(
        y_test,
        pred_lr
    )

    lr_report = classification_report(
        y_test,
        pred_lr,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    lr_macro_f1 = macro_f1_from_report(
        lr_report
    )

    rows.append({
        "task":
            "three_class_expertise_tier",

        "model":
            "logistic_regression",

        "n_total":
            len(X),

        "n_train":
            len(X_train),

        "n_test":
            len(X_test),

        "accuracy":
            lr_accuracy,

        "macro_f1":
            lr_macro_f1,

        "random_state":
            42,
    })

    print(
        "\nThree-class Logistic Regression"
    )
    print(
        f"Accuracy: {lr_accuracy:.4f}"
    )
    print(
        f"Macro-F1: {lr_macro_f1:.4f}"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        pred_lr
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"actual_{label}"
            for label
            in label_encoder.classes_
        ],
        columns=[
            f"pred_{label}"
            for label
            in label_encoder.classes_
        ],
    )

    cm_df.to_csv(
        CONFUSION_OUTPUT_PATH
    )

    # ============================================================
    # RANDOM FOREST
    # ============================================================

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )

    random_forest.fit(
        X_train,
        y_train
    )

    pred_rf = random_forest.predict(
        X_test
    )

    rf_accuracy = accuracy_score(
        y_test,
        pred_rf
    )

    rf_report = classification_report(
        y_test,
        pred_rf,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    rf_macro_f1 = macro_f1_from_report(
        rf_report
    )

    rows.append({
        "task":
            "three_class_expertise_tier",

        "model":
            "random_forest",

        "n_total":
            len(X),

        "n_train":
            len(X_train),

        "n_test":
            len(X_test),

        "accuracy":
            rf_accuracy,

        "macro_f1":
            rf_macro_f1,

        "random_state":
            42,
    })

    print(
        "\nThree-class Random Forest"
    )
    print(
        f"Accuracy: {rf_accuracy:.4f}"
    )
    print(
        f"Macro-F1: {rf_macro_f1:.4f}"
    )

    importance_df = (
        pd.DataFrame({
            "feature":
                FEATURES,

            "importance":
                random_forest
                .feature_importances_,
        })
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance_df.to_csv(
        IMPORTANCE_OUTPUT_PATH,
        index=False
    )

    # ============================================================
    # SHUFFLED-LABEL BASELINE
    # ============================================================

    rng = np.random.default_rng(
        42
    )

    shuffle_rows = []

    for run in range(100):
        y_shuffled = (
            y_encoded.copy()
        )

        rng.shuffle(
            y_shuffled
        )

        X_train_s, X_test_s, y_train_s, y_test_s = (
            train_test_split(
                X,
                y_shuffled,
                test_size=0.20,
                random_state=run,
                stratify=y_shuffled,
            )
        )

        shuffled_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

        shuffled_model.fit(
            X_train_s,
            y_train_s
        )

        shuffled_pred = (
            shuffled_model
            .predict(
                X_test_s
            )
        )

        shuffled_accuracy = (
            accuracy_score(
                y_test_s,
                shuffled_pred
            )
        )

        shuffle_rows.append({
            "run":
                run,

            "accuracy":
                shuffled_accuracy,
        })

    shuffle_df = pd.DataFrame(
        shuffle_rows
    )

    shuffle_df.to_csv(
        SHUFFLE_OUTPUT_PATH,
        index=False
    )

    shuffle_mean = (
        shuffle_df[
            "accuracy"
        ]
        .mean()
    )

    shuffle_sd = (
        shuffle_df[
            "accuracy"
        ]
        .std(
            ddof=0
        )
    )

    rows.append({
        "task":
            "three_class_expertise_tier",

        "model":
            "shuffled_label_logistic_regression",

        "n_total":
            len(X),

        "n_train":
            np.nan,

        "n_test":
            np.nan,

        "accuracy":
            shuffle_mean,

        "macro_f1":
            np.nan,

        "random_state":
            "runs_0_to_99",
    })

    print(
        "\nShuffled-label baseline"
    )
    print(
        f"Mean accuracy: {shuffle_mean:.4f}"
    )
    print(
        f"SD: {shuffle_sd:.4f}"
    )

    return rows


def run_binary_model(df):
    binary_df = (
        df[
            df[
                "expertise_tier"
            ]
            .isin(
                [
                    "low",
                    "high"
                ]
            )
        ]
        .copy()
    )

    X = (
        binary_df[
            FEATURES
        ]
        .dropna()
    )

    y = binary_df.loc[
        X.index,
        "expertise_tier"
    ]

    label_encoder = LabelEncoder()

    y_encoded = (
        label_encoder
        .fit_transform(y)
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y_encoded,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded,
        )
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = macro_f1_from_report(
        report
    )

    print(
        "\nBinary Logistic Regression: high vs low"
    )
    print(
        f"Accuracy: {accuracy:.4f}"
    )
    print(
        f"Macro-F1: {macro_f1:.4f}"
    )

    return {
        "task":
            "binary_high_vs_low",

        "model":
            "logistic_regression",

        "n_total":
            len(X),

        "n_train":
            len(X_train),

        "n_test":
            len(X_test),

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "random_state":
            42,
    }


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{INPUT_PATH}\n"
            "Run analysis/02_linguistic_features.py first."
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required = {
        "expertise_tier"
    } | set(FEATURES)

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    print(
        f"Loaded {len(df):,} posts"
    )

    SUMMARY_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    rows.extend(
        run_three_class_models(
            df
        )
    )

    rows.append(
        run_binary_model(
            df
        )
    )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved classifier summary to:"
    )
    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        "\nSaved Random Forest feature importance to:"
    )
    print(
        IMPORTANCE_OUTPUT_PATH
    )

    print(
        "\nSaved confusion matrix to:"
    )
    print(
        CONFUSION_OUTPUT_PATH
    )

    print(
        "\nSaved shuffled-label runs to:"
    )
    print(
        SHUFFLE_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
