import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar dados
df = pd.read_csv('datasets/selector_sharpe_full.csv', parse_dates=['ts_end'])

# Separar features e target
feature_cols = [c for c in df.columns if c not in ('row_id', 'ts_end', 'best_strategy')]
X = df[feature_cols].values
y = df['best_strategy'].values

# Converter labels para números
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split temporal com gap
df = df.sort_values('ts_end').reset_index(drop=True)
split_idx = int(len(df) * 0.75)
train_idx = df.index[:split_idx]
test_idx = df.index[split_idx:]

# Gap de 7 dias
gap_end = df.loc[train_idx, 'ts_end'].max() + pd.Timedelta(days=7)
test_idx = df[df['ts_end'] > gap_end].index

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"Treino: {len(X_train)} amostras")
print(f"Teste: {len(X_test)} amostras")

# Treinar XGBoost
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.3f}")

# Salvar resultados
output_dir = Path('models/xgb_sharpe_fixed')
output_dir.mkdir(parents=True, exist_ok=True)

# Salvar modelo e encoder
joblib.dump(model, output_dir / 'model.joblib')
joblib.dump(le, output_dir / 'label_encoder.joblib')

# Matriz de confusão
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png')
plt.close()

# Feature importance
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(feature_imp['feature'][:20], feature_imp['importance'][:20])
plt.xlabel('Importância')
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png')
plt.close()

# Salvar metadados
metadata = {
    'model_type': 'xgb',
    'test_accuracy': float(accuracy),
    'classes': le.classes_.tolist(),
    'n_features': len(feature_cols),
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}

with open(output_dir / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Relatório de classificação
report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
pd.DataFrame(report).T.to_csv(output_dir / 'classification_report.csv')

print(f"\nModelo salvo em: {output_dir}")
print("\nPrincipais features:")
print(feature_imp.head(10))