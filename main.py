# fraud_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Basic info
print("Dataset shape:", df.shape)
print("Fraudulent transactions:", df[df['Class'] == 1].shape[0])

# Normalize 'Amount'
scaler = StandardScaler()
df['normAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df.drop(['Time','Amount'], axis=1, inplace=True)

# Separate features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# -------------------
# Anomaly Detection - Isolation Forest
# -------------------
print("\n Isolation Forest:")

isof = IsolationForest(n_estimators=100, contamination=0.001)
y_pred_iforest = isof.fit_predict(X)

# Convert -1 to 1 (fraud), 1 to 0 (normal)
y_pred_iforest = np.where(y_pred_iforest == -1, 1, 0)

print("F1 Score (IF):", f1_score(y, y_pred_iforest))
print("AUC-ROC (IF):", roc_auc_score(y, y_pred_iforest))

# -------------------
# Classification with SMOTE and Logistic Regression
# -------------------
print("\n Logistic Regression with SMOTE:")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance training data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res)

# Predict
y_pred_lr = lr.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred_lr))
print("F1 Score (LR):", f1_score(y_test, y_pred_lr))
print("AUC-ROC (LR):", roc_auc_score(y_test, y_pred_lr))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label='LogReg AUC: %.2f' % roc_auc_score(y_test, y_pred_lr))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Save the trained logistic regression model
joblib.dump(lr, 'logistic_model.pkl')

# Save the StandardScaler used for 'Amount'
joblib.dump(scaler, 'scaler.pkl')
