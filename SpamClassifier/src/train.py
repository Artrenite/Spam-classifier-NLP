from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .preprocessing import preprocess_corpus


RANDOM_STATE = 42


@dataclass
class ExperimentResult:
    name: str
    vectorizer: str
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def load_data(data_path: Path) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(data_path)
    if set(["label", "text"]).issubset(df.columns):
        y = df["label"]
        X = df["text"]
    elif set(["CATEGORY", "MESSAGE"]).issubset(df.columns):
        y = df["CATEGORY"]
        X = df["MESSAGE"]
    else:
        raise ValueError(f"Unexpected columns in CSV: {list(df.columns)}")

    if y.dtype == object:
        y = y.map({"ham": 0, "spam": 1}).astype(int)

    return X.astype(str), y.astype(int)


def make_pipeline(vectorizer: BaseEstimator, model: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", FunctionTransformer(preprocess_corpus, validate=False)),
            ("vectorizer", vectorizer),
            ("model", model),
        ]
    )


def evaluate_model(name: str, pipe: Pipeline, X_test, y_test) -> ExperimentResult:
    y_pred = pipe.predict(X_test)
    return ExperimentResult(
        name=name,
        vectorizer=pipe.named_steps["vectorizer"].__class__.__name__,
        model=pipe.named_steps["model"].__class__.__name__,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
    )


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, labels: List[str]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metrics_bar(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(metrics_df["name"], metrics_df["f1"])
    ax.set_ylabel("F1-score")
    ax.set_xlabel("Model")
    ax.set_xticks(range(len(metrics_df["name"])))
    ax.set_xticklabels(metrics_df["name"], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _unwrap_estimator(model: BaseEstimator) -> BaseEstimator:
    """
    If model is CalibratedClassifierCV, return fitted underlying estimator (LinearSVC).
    Otherwise return model itself.
    """
    if isinstance(model, CalibratedClassifierCV):
        # after fit(), estimator_ is available
        if hasattr(model, "estimator_") and model.estimator_ is not None:
            return model.estimator_
    return model


def extract_top_features(pipeline: Pipeline, top_n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame] | None:
    vec = pipeline.named_steps["vectorizer"]
    model = pipeline.named_steps["model"]
    inner_model = _unwrap_estimator(model)

    try:
        feature_names = np.array(vec.get_feature_names_out())
    except Exception:
        return None

    if hasattr(inner_model, "coef_"):
        w = inner_model.coef_.ravel()
        top_spam_idx = np.argsort(w)[-top_n:][::-1]
        top_ham_idx = np.argsort(w)[:top_n]
        top_spam = pd.DataFrame({"feature": feature_names[top_spam_idx], "weight": w[top_spam_idx]})
        top_ham = pd.DataFrame({"feature": feature_names[top_ham_idx], "weight": w[top_ham_idx]})
        return top_spam, top_ham

    if hasattr(inner_model, "feature_log_prob_"):
        logp = inner_model.feature_log_prob_
        top_spam_idx = np.argsort(logp[1])[-top_n:][::-1]
        top_ham_idx = np.argsort(logp[0])[-top_n:][::-1]
        top_spam = pd.DataFrame({"feature": feature_names[top_spam_idx], "log_prob": logp[1][top_spam_idx]})
        top_ham = pd.DataFrame({"feature": feature_names[top_ham_idx], "log_prob": logp[0][top_ham_idx]})
        return top_spam, top_ham

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="spam_NLP.csv",
        help="Path to spam_NLP.csv (not included in ZIP).",
    )
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    vectorizers = {
        "count": CountVectorizer(max_features=10000, ngram_range=(1, 2)),
        "tfidf": TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95),
    }

    svm_calibrated = CalibratedClassifierCV(
        estimator=LinearSVC(random_state=RANDOM_STATE),
        method="sigmoid",
        cv=3,
    )

    models = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="liblinear"),
        "svm": svm_calibrated,
    }

    results: List[ExperimentResult] = []
    fitted: Dict[str, Pipeline] = {}

    for v_name, vec in vectorizers.items():
        for m_name, model in models.items():
            exp_name = f"{v_name}_{m_name}"
            pipe = make_pipeline(vec, model)
            pipe.fit(X_train, y_train)
            fitted[exp_name] = pipe
            results.append(evaluate_model(exp_name, pipe, X_test, y_test))

    metrics_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("f1", ascending=False)
    metrics_df.to_csv(out_dir / "metrics_models.csv", index=False)

    best_name = metrics_df.iloc[0]["name"]  # sorted by f1 desc
    best_pipe = fitted[best_name]

    joblib.dump(best_pipe, out_dir / "spam_pipeline.joblib")
    joblib.dump(best_pipe.named_steps["model"], out_dir / "model.joblib")
    joblib.dump(best_pipe.named_steps["vectorizer"], out_dir / "vectorizer.joblib")

    y_pred_best = best_pipe.predict(X_test)
    report = classification_report(y_test, y_pred_best, target_names=["HAM", "SPAM"])
    (out_dir / "best_model_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred_best)
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", labels=["HAM", "SPAM"])

    tops = extract_top_features(best_pipe, top_n=20)
    if tops is not None:
        top_spam, top_ham = tops
        top_spam.to_csv(out_dir / "top_features_spam.csv", index=False)
        top_ham.to_csv(out_dir / "top_features_ham.csv", index=False)

    tune_pipe = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, solver="liblinear"),
    )
    param_grid = {
        "vectorizer__max_features": [5000],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [1, 2],
        "vectorizer__max_df": [0.9, 0.95],
        "model__C": [0.5, 1.0, 2.0],
    }
    grid = RandomizedSearchCV(
        tune_pipe,
        param_distributions=param_grid,
        n_iter=6,
        scoring="f1",
        cv=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv(out_dir / "gridsearch_results.csv", index=False)

    if "param_model__C" in cv_df.columns:
        tmp = cv_df[["param_model__C", "mean_test_score"]].copy()
        tmp["param_model__C"] = tmp["param_model__C"].astype(float)
        tmp = tmp.sort_values("param_model__C")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(tmp["param_model__C"], tmp["mean_test_score"], marker="o")
        ax.set_xlabel("Parametr C (LogisticRegression)")
        ax.set_ylabel("Średni F1-score (CV)")
        ax.set_title("Wpływ parametru C na jakość modelu (TF-IDF + LR)")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / "hyperparameter_C_vs_f1.png", dpi=160)
        plt.close(fig)

    tuned_lr_pipe: Pipeline = grid.best_estimator_
    tuned_lr_res = evaluate_model("tuned_tfidf_lr", tuned_lr_pipe, X_test, y_test)

    metrics_plus = pd.concat([metrics_df, pd.DataFrame([tuned_lr_res.__dict__])], ignore_index=True)
    metrics_plus = metrics_plus.sort_values("f1", ascending=False).reset_index(drop=True)
    plot_metrics_bar(metrics_plus, out_dir / "metrics_comparison.png")

    print("Done.")
    print("Best baseline model:", best_name)
    print("Best baseline metrics (test):", metrics_df.iloc[0].to_dict())
    print("Tuned LR best params:", grid.best_params_)
    print("Tuned LR test metrics:", tuned_lr_res)


if __name__ == "__main__":
    main()
