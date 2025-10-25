import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pickle
import os

# Create pkl_files directory if it doesn't exist
os.makedirs('pkl_files', exist_ok=True)

# Load data
data = pd.read_csv("cleaned_water_potability_2.csv")
X = data.drop('Potability', axis=1)
y = data['Potability']

# Scale features
scaler = RobustScaler()
x_scaled = scaler.fit_transform(X)
X = pd.DataFrame(x_scaled, columns=X.columns)

# Save scaler
with open('pkl_files/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Oversample
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Save test data for accuracy calculation
X_test.to_csv('pkl_files/X_test.csv', index=False)
pd.DataFrame(y_test, columns=['Potability']).to_csv('pkl_files/y_test.csv', index=False)

print("Training models...")

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
model = LogisticRegression(max_iter=100, random_state=0, n_jobs=20)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
with open('pkl_files/logistic_regression.pkl', 'wb') as f:
    pickle.dump(model, f)

# 2. Decision Tree
print("\n2. Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
with open('pkl_files/decision_tree.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

# 3. Random Forest
print("\n3. Training Random Forest...")
rf_model = RandomForestClassifier(min_samples_leaf=0.16, random_state=42, n_estimators=300)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
with open('pkl_files/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# 4. XGBoost
print("\n4. Training XGBoost...")
xgb=XGBClassifier( colsample_bytree=0.811, subsample=0.8344206263318037,objective='binary:logistic',eval_metric='auc',max_depth=9,n_estimators=200,random_state=42,learning_rate=0.109,n_jobs=5)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"Confusion matrix:\n{ConfusionMatrixDisplay.from_predictions(y_test, y_pred_xgb)} ")
sns.heatmap(pd.DataFrame(ConfusionMatrixDisplay.from_predictions(y_test, y_pred_xgb).confusion_matrix), annot=True, fmt='d', cmap='Blues')
print(f"Classification report:\n{classification_report(y_test, y_pred_xgb)}")
with open('pkl_files/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)

# 5. AdaBoost
print("\n5. Training AdaBoost...")
ada = AdaBoostClassifier(learning_rate=0.03, n_estimators=250, random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
ada_accuracy = accuracy_score(y_test, y_pred_ada)
print(f"AdaBoost Accuracy: {ada_accuracy:.4f}")
with open('pkl_files/adaboost_model.pkl', 'wb') as f:
    pickle.dump(ada, f)

# 6. K-Nearest Neighbors
print("\n6. Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {knn_accuracy:.4f}")
with open('pkl_files/knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# 7. Support Vector Machine
print("\n7. Training SVM...")
svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)  # Added probability=True for predict_proba
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
with open('pkl_files/svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Summary
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Logistic Regression: {lr_accuracy*100:.2f}%")
print(f"Decision Tree:       {dt_accuracy*100:.2f}%")
print(f"Random Forest:       {rf_accuracy*100:.2f}%")
print(f"XGBoost:             {xgb_accuracy*100:.2f}%")
print(f"AdaBoost:            {ada_accuracy*100:.2f}%")
print(f"KNN:                 {knn_accuracy*100:.2f}%")
print(f"SVM:                 {svm_accuracy*100:.2f}%")
print("="*50)
print("\nAll models saved successfully in 'pkl_files' folder!")
print("Test data saved for Flask app accuracy calculation.")