#!/usr/bin/env python3
import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in")
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    X = df.drop(["row_id", "ts_end", "best_strategy"], axis=1)
    y = df["best_strategy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, shuffle=False)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42))
    ])
    pipe.fit(X_train, y_train)

    print("=== Hold‑out ===")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))

    # opcional: cross‑val “rolling window”
    tscv = TimeSeriesSplit(n_splits=4)
    cv_scores = cross_val_score(pipe, X, y, cv=tscv, n_jobs=-1)
    print("CV scores:", cv_scores.round(3), "→ mean", cv_scores.mean().round(3))

    joblib.dump(pipe, args.model_out)
    print(f"Modelo salvo em {args.model_out}")
