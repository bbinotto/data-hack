"""Compare classifiers/encodings/scalers for predicting IS_DELAYED, then
retrain the winner on train+val and report a final test-set score.

Note: IS_DELAYED is imbalanced (~81.5% on-time / 18.5% delayed), so the
"always predict on-time" baseline already scores ~0.815 accuracy. Accuracy
alone is reported for comparability with the original script, but
balanced accuracy / F1 / ROC-AUC are tracked too since they're more
informative for this class distribution.

ORIGIN_AIRPORT/DESTINATION_AIRPORT (322 categories each) exceed
HistGradientBoostingClassifier's native categorical cardinality limit
(255), so those two columns get smoothed target-encoding (mean historical
delay rate, fit on train only) for the boosting candidates; AIRLINE (14
categories) stays a native pandas category.
"""

import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

DATA_DIR = Path(__file__).resolve().parent.parent

NUM_FEATURES = ["MONTH", "DAY_OF_WEEK", "DISTANCE", "DEPARTURE_HOUR"]
CAT_FEATURES = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
HIGH_CARD_FEATURES = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
TARGET_ENCODE_SMOOTHING = 20

DTYPES = {
    "MONTH": "int8",
    "DAY_OF_WEEK": "int8",
    "AIRLINE": "category",
    "ORIGIN_AIRPORT": "category",
    "DESTINATION_AIRPORT": "category",
    "SCHEDULED_DEPARTURE": "int32",
    "DISTANCE": "int32",
    "IS_DELAYED": "int8",
}


def load(filename, frac=None, random_state=42):
    df = pd.read_csv(DATA_DIR / filename, usecols=list(DTYPES), dtype=DTYPES)
    if frac is not None:
        df = df.sample(frac=frac, random_state=random_state)
    df["DEPARTURE_HOUR"] = (df["SCHEDULED_DEPARTURE"] // 100).astype("int8")
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df["IS_DELAYED"]
    return X, y


def fit_target_encoding(X, y, columns, smoothing=TARGET_ENCODE_SMOOTHING):
    """Smoothed mean-target encoding, fit on training data only."""
    global_mean = y.mean()
    mappings = {}
    for col in columns:
        stats = (
            pd.DataFrame({"cat": X[col].to_numpy(), "y": y.to_numpy()})
            .groupby("cat")["y"]
            .agg(["mean", "count"])
        )
        stats["smoothed"] = (
            stats["mean"] * stats["count"] + global_mean * smoothing
        ) / (stats["count"] + smoothing)
        mappings[col] = stats["smoothed"]
    return mappings, global_mean


def apply_target_encoding(X, mappings, global_mean):
    """Build the boosting feature frame: target-encoded high-cardinality columns
    plus the low-cardinality AIRLINE category and the numeric features."""
    out = X[NUM_FEATURES + ["AIRLINE"]].copy()
    for col, mapping in mappings.items():
        out[f"{col}_DELAY_RATE"] = (
            X[col].map(mapping).fillna(global_mean).astype("float32")
        )
    return out


def evaluate(name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )
    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else float("nan"),
        "seconds": time.time() - t0,
    }
    return metrics


def main():
    # Sample for speed while comparing candidates; the winner gets refit on
    # the full train+val data at the end.
    X_train, y_train = load("flights_train.csv", frac=0.3)
    X_test, y_test = load("flights_test.csv")

    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"Majority-class baseline accuracy: {baseline:.4f}\n")

    mappings, global_mean = fit_target_encoding(X_train, y_train, HIGH_CARD_FEATURES)
    X_train_hgb = apply_target_encoding(X_train, mappings, global_mean)
    X_test_hgb = apply_target_encoding(X_test, mappings, global_mean)

    onehot = ColumnTransformer(
        [
            ("num", "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    ordinal = ColumnTransformer(
        [
            ("num", "passthrough", NUM_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT_FEATURES,
            ),
        ]
    )

    # (name, model, feature_kind) -- feature_kind picks which (X_train, X_test) pair to use.
    candidates = []

    for scaler_name, scaler in [
        ("StandardScaler", StandardScaler(with_mean=False)),
        ("MaxAbsScaler", MaxAbsScaler()),
        ("NoScaling", None),
    ]:
        for C in (0.1, 1.0, 10.0):
            steps = [("encode", onehot)]
            if scaler is not None:
                steps.append(("scale", scaler))
            steps.append(("clf", LogisticRegression(max_iter=1000, C=C)))
            name = f"LogisticRegression(C={C}, scaler={scaler_name})"
            candidates.append((name, Pipeline(steps), "raw"))

    for n_estimators, max_depth in [(200, 10), (300, 16), (300, None)]:
        name = f"RandomForest(n_estimators={n_estimators}, max_depth={max_depth})"
        model = Pipeline(
            [
                ("encode", ordinal),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        candidates.append((name, model, "raw"))

    for learning_rate, max_depth, max_iter in [
        (0.1, None, 200),
        (0.05, 8, 400),
        (0.1, 6, 300),
    ]:
        name = f"HistGradientBoosting(lr={learning_rate}, max_depth={max_depth}, max_iter={max_iter})"
        model = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            random_state=42,
        )
        candidates.append((name, model, "hgb"))

    feature_sets = {"raw": (X_train, X_test), "hgb": (X_train_hgb, X_test_hgb)}

    results = []
    for name, model, kind in candidates:
        X_tr, X_va = feature_sets[kind]
        metrics = evaluate(name, model, X_tr, y_train, X_va, y_test)
        results.append((metrics, model, kind))

    results.sort(key=lambda r: r[0]["accuracy"], reverse=True)

    results_df = pd.DataFrame([metrics for metrics, _, _ in results]).set_index("name")
    print("\nModel comparison (sorted by accuracy):")
    print(results_df.round(4).to_string())

    best_metrics, best_model, best_kind = results[0]
    print(f"\nBest by accuracy: {best_metrics['name']} (feature kind: {best_kind})")

    # Refit the winner on train+val, evaluate once on the held-out test set.
    X_full_raw = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    X_val_raw, y_val = load("flights_val.csv")

    if best_kind == "hgb":
        full_mappings, full_global_mean = fit_target_encoding(
            X_full_raw, y_full, HIGH_CARD_FEATURES
        )
        X_full = apply_target_encoding(X_full_raw, full_mappings, full_global_mean)
        X_val = apply_target_encoding(X_val_raw, full_mappings, full_global_mean)
    else:
        X_full, X_val = X_full_raw, X_val_raw

    best_model.fit(X_full, y_full)
    y_pred = best_model.predict(X_val)
    print(f"\nFinal test accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred, target_names=["On-time", "Delayed"]))


if __name__ == "__main__":
    main()
