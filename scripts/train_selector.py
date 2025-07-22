#!/usr/bin/env python3
"""Treina um modelo de seleção de estratégia (logreg ou árvore)."""

import argparse, joblib, pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="dataset gerado pelo build_training_set.py")
    p.add_argument("--model", default="models/selector.pkl")
    p.add_argument("--algo", choices=["logreg", "tree"], default="logreg")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()

def main() -> None:
    args = parse()
    df = pd.read_csv(args.csv)

    X = df.drop(columns=["row_id", "ts_end", "best_strategy"])
    y = df["best_strategy"]

    # ------- split estratificado -------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )

    num_cols = X.select_dtypes("number").columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols)
    ])

    if args.algo == "logreg":
        clf = LogisticRegression(max_iter=200, multi_class="multinomial")
    else:
        clf = DecisionTreeClassifier(max_depth=5)

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"Accuracy: {acc:.3f}")

    pathlib.Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model)
    print(f"Modelo salvo em {args.model}")

if __name__ == "__main__":
    main()
