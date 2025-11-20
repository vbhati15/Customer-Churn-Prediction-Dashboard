# churn_model.py
# Simple end-to-end churn modelling script using a tiny sample dataset.
# Outputs: outputs/predictions_sample.csv, outputs/feature_importance.csv, outputs/feature_importance.png

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# Paths
DATA_PATH = 'data_sample.csv'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load data
print('Loading data...')
df = pd.read_csv(DATA_PATH)
print('Data shape:', df.shape)

# 2. Basic EDA
print('\n--- Head ---')
print(df.head())
print('\n--- Info ---')
print(df.info())
print('\n--- Nulls ---')
print(df.isnull().sum())

# 3. Feature engineering (simple)
TARGET = 'Churn'
# Drop any ID columns if present
if 'customerID' in df.columns:
    df = df.copy()
else:
    df = df.copy()

features = [c for c in df.columns if c != TARGET and c != 'customerID']

# Fill missing values
for c in features:
    if df[c].dtype == 'object':
        df[c] = df[c].fillna('Unknown')
    else:
        df[c] = df[c].fillna(df[c].median())

# One-hot encode categorical features (basic)
cat_cols = [c for c in features if df[c].dtype == 'object']
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Recompute features list after encoding
features = [c for c in df.columns if c != TARGET and c != 'customerID']

# 4. Train/test split
X = df[features]
y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Scaling for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Baseline model: Logistic Regression
print('\nTraining Logistic Regression...')
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train_scaled, y_train)
y_pred_lr = clf_lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
roc_lr = roc_auc_score(y_test, clf_lr.predict_proba(X_test_scaled)[:, 1])
print('Logistic Regression Accuracy:', round(acc_lr, 4))
print('Logistic Regression ROC AUC:', round(roc_lr, 4))

# 7. Tree model: Random Forest
print('\nTraining Random Forest...')
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, clf_rf.predict_proba(X_test)[:, 1])
print('Random Forest Accuracy:', round(acc_rf, 4))
print('Random Forest ROC AUC:', round(roc_rf, 4))

# 8. Save predictions sample
sample_out = X_test.copy()
sample_out = sample_out.reset_index(drop=True)
sample_out['true_churn'] = y_test.reset_index(drop=True)
sample_out['pred_churn'] = y_pred_rf
sample_out['pred_proba'] = clf_rf.predict_proba(X_test)[:, 1]
sample_out.head(10).to_csv(os.path.join(OUT_DIR, 'predictions_sample.csv'), index=False)
print('\nSaved sample predictions to', os.path.join(OUT_DIR, 'predictions_sample.csv'))

# 9. Feature importance
importances = clf_rf.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(
    by='importance', ascending=False
)
feat_imp.to_csv(os.path.join(OUT_DIR, 'feature_importance.csv'), index=False)
print('Saved feature importance to', os.path.join(OUT_DIR, 'feature_importance.csv'))

# 10. Feature importance plot
plt.figure(figsize=(8, 6))
top_n = min(15, feat_imp.shape[0])
plt.barh(feat_imp['feature'].head(top_n)[::-1], feat_imp['importance'].head(top_n)[::-1])
plt.xlabel('Importance')
plt.title('Top feature importances')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance.png'))
print('Saved feature importance plot to', os.path.join(OUT_DIR, 'feature_importance.png'))

# 11. Classification report
print('\nClassification report (Random Forest):')
print(classification_report(y_test, y_pred_rf))

print('\nDone.')
