# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os

# ── 1. Load Data ──────────────────────────────────────────
df = pd.read_csv('data/heart.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:", df.isnull().sum().sum())

# ── 2. EDA ────────────────────────────────────────────────
print("\nTarget distribution:")
print(df['target'].value_counts())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('model/correlation.png')
print("Saved correlation heatmap!")

# ── 3. Prepare Data ───────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'model/scaler.pkl')

# ── 4. Train 3 Models ─────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name} Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds))

# ── 5. Save Best Model ────────────────────────────────────
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, 'model/best_model.pkl')
print(f"\n✅ Best Model: {best_model_name} ({results[best_model_name]*100:.2f}%)")
print("Model saved!")