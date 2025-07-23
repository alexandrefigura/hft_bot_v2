#!/usr/bin/env python3
"""
Treina modelo ML para selecionar a MELHOR estratégia de trading prevista
pelo dataset gerado em build_training_set_v3.py.

Principais features:
• Split temporal + gap para evitar leakage
• Cross‑validation TimeSeriesSplit e walk‑forward
• Modelos RF / XGB / LGB (instalação opcional)
• GridSearchCV com parâmetros customizáveis via JSON
• Seleção de KMelhores features e validação de features duplicadas/constantes
• Relatórios em CSV/PNG + metadados JSON
• Logging detalhado com controle de log‑level

Uso básico:
    python train_model_v2.py datasets/selector.csv \
           --output-dir models/rf_model \
           --model rf --test-ratio 0.25 --gap-days 7

Para usar XGBoost ou LightGBM:
    pip install xgboost lightgbm
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Dependências opcionais
try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ---------------------------------------------------------------------------
# Logging global
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transformador para validar e limpar features
# ---------------------------------------------------------------------------


class FeatureValidator(BaseEstimator, TransformerMixin):
    """Remove features constantes ou duplicadas."""

    def __init__(self, remove_constant: bool = True, remove_duplicate: bool = True):
        self.remove_constant = remove_constant
        self.remove_duplicate = remove_duplicate
        self.features_to_keep_: list[int] = []

    def fit(self, X, y=None):  # noqa: N803
        self.features_to_keep_ = list(range(X.shape[1]))
        if self.remove_constant:
            constant_mask = X.std(axis=0) != 0
            self.features_to_keep_ = [i for i, keep in enumerate(constant_mask) if keep]
        if self.remove_duplicate:
            _, unique_idx = np.unique(X.T, axis=0, return_index=True)
            self.features_to_keep_ = sorted(set(self.features_to_keep_) & set(unique_idx))
        return self

    def transform(self, X):  # noqa: N803
        return X[:, self.features_to_keep_]

# ---------------------------------------------------------------------------
# Split temporal helper
# ---------------------------------------------------------------------------

def temporal_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2, gap_days: int = 0):
    """Separa treino/teste respeitando tempo com gap opcional."""

    df = df.sort_values("ts_end").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    if gap_days > 0:
        gap_end = train_df["ts_end"].max() + pd.Timedelta(days=gap_days)
        test_df = df[df["ts_end"] > gap_end]
    else:
        test_df = df.iloc[split_idx:]
    logger.info("Treino: %s ➜ %s (%d) | Teste: %s ➜ %s (%d)",
                train_df["ts_end"].min().date(), train_df["ts_end"].max().date(), len(train_df),
                test_df["ts_end"].min().date(), test_df["ts_end"].max().date(), len(test_df))
    return train_df, test_df

# ---------------------------------------------------------------------------
# Walk‑forward validation
# ---------------------------------------------------------------------------

def walk_forward_validation(X, y, ts, model, n_splits: int = 5, train_months: int = 6, test_months: int = 1):
    """Validação walk‑forward usando janelas fixas em meses."""
    rows = []
    for i in range(n_splits):
        train_start = ts.min() + pd.DateOffset(months=i * test_months)
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start, test_end = train_end, train_end + pd.DateOffset(months=test_months)

        train_mask = (ts >= train_start) & (ts < train_end)
        test_mask = (ts >= test_start) & (ts < test_end)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model.fit(X[train_mask], y[train_mask])
        acc = accuracy_score(y[test_mask], model.predict(X[test_mask]))
        rows.append({
            "split": i,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "accuracy": acc,
            "n_train": train_mask.sum(),
            "n_test": test_mask.sum(),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Visualizações helper
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes, path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predito"); plt.ylabel("Real"); plt.tight_layout(); plt.savefig(path); plt.close()


def plot_feature_importance(model, feature_names, path: Path, top_n: int = 20):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[idx], imp[idx])
    plt.tight_layout(); plt.savefig(path); plt.close()

# ---------------------------------------------------------------------------
# Pipeline factory - CORRIGIDO
# ---------------------------------------------------------------------------

def get_model_pipeline(model_type: str, n_feats: int, param_grid: Optional[Dict[str, Any]] = None) -> Tuple[Pipeline, Dict[str, Any]]:
    steps = [
        ("validator", FeatureValidator()),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k="all")),
    ]
    if model_type == "rf":
        steps.append(("model", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)))
        grid = {
            "selector__k": [min(n_feats, k) for k in [10, 20, 30]] + ["all"],  # CORRIGIDO
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
        }
    elif model_type == "xgb" and HAS_XGB:
        steps.append(("model", xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)))
        grid = {
            "selector__k": [min(n_feats, k) for k in [10, 20, 30]] + ["all"],  # CORRIGIDO
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1, 0.3],
        }
    elif model_type == "lgb" and HAS_LGB:
        steps.append(("model", lgb.LGBMClassifier(random_state=42, verbosity=-1)))
        grid = {
            "selector__k": [min(n_feats, k) for k in [10, 20, 30]] + ["all"],  # CORRIGIDO
            "model__n_estimators": [200, 400],
            "model__num_leaves": [31, 63, 127],
            "model__learning_rate": [0.05, 0.1, 0.3],
        }
    else:
        raise ValueError(f"Modelo {model_type} não suportado ou dependência ausente.")

    if param_grid:
        grid.update(param_grid)
    return Pipeline(steps), grid

# ---------------------------------------------------------------------------
# Função principal de treinamento
# ---------------------------------------------------------------------------

def train_selector(csv_in: str, out_dir: str, model_type: str, test_ratio: float, gap_days: int, cv_splits: int, optimize: bool, param_grid: Optional[Dict[str, Any]] = None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_in, parse_dates=["ts_end"])
    feats_cols = [c for c in df.columns if c not in ("row_id", "ts_end", "best_strategy")]
    X, y, ts = df[feats_cols].values, df["best_strategy"].values, df["ts_end"]

    train_df, test_df = temporal_train_test_split(df, test_ratio, gap_days)
    X_train, y_train = train_df[feats_cols].values, train_df["best_strategy"].values
    X_test, y_test = test_df[feats_cols].values, test_df["best_strategy"].values

    pipe, grid = get_model_pipeline(model_type, X_train.shape[1], param_grid)

    if optimize:
        logger.info("GridSearchCV iniciando ...")
        gs = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(cv_splits), scoring="accuracy", n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_; best_cv = float(gs.best_score_)
        pd.DataFrame(gs.cv_results_).to_csv(out / "grid_search_results.csv", index=False)
    else:
        model = pipe.fit(X_train, y_train)
        best_params = {}; best_cv = None

    # Avaliação
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Accuracy teste: %.3f", acc)

    # Relatórios
    plot_confusion_matrix(y_test, y_pred, classes=np.unique(y), path=out / "confusion_matrix.png")
    plot_feature_importance(model.named_steps["model"], feats_cols, path=out / "feature_importance.png")

    pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.to_csv(out / "classification_report.csv")

    wf = walk_forward_validation(X, y, ts, model, n_splits=5, train_months=6, test_months=1)
    if not wf.empty:
        wf.to_csv(out / "walk_forward_results.csv", index=False)

    joblib.dump(model, out / "model.joblib")

    meta = {
        "model_type": model_type,
        "train_date": datetime.now().isoformat(),
        "test_accuracy": float(acc),
        "best_params": best_params,
        "best_cv": best_cv,
        "classes": sorted(set(y)),
        "n_features": len(feats_cols),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }
    with open(out / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Modelo e artefatos salvos em %s", out)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treina modelo para selecionar estratégia de trading")
    ap.add_argument("csv_in", help="Dataset gerado pelo builder")
    ap.add_argument("--output-dir", required=True, help="Diretório de saída")
    ap.add_argument("--model", choices=["rf", "xgb", "lgb"], default="rf")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--gap-days", type=int, default=7)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--no-optimize", action="store_true", help="Pular GridSearch")
    ap.add_argument("--param-grid", help="Arquivo JSON com grid customizado")
    ap.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = ap.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    grid = None
    if args.param_grid:
        grid = json.loads(Path(args.param_grid).read_text(encoding="utf-8"))

    if args.model == "xgb" and not HAS_XGB:
        logger.error("XGBoost não instalado -> pip install xgboost"); raise SystemExit(1)
    if args.model == "lgb" and not HAS_LGB:
        logger.error("LightGBM não instalado -> pip install lightgbm"); raise SystemExit(1)

    train_selector(
        csv_in=args.csv_in,
        out_dir=args.output_dir,
        model_type=args.model,
        test_ratio=args.test_ratio,
        gap_days=args.gap_days,
        cv_splits=args.cv_splits,
        optimize=not args.no_optimize,
        param_grid=grid,
    )