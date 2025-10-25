from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load all models
with open('pkl_files/logistic_regression.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('pkl_files/decision_tree.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('pkl_files/random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('pkl_files/xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('pkl_files/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('pkl_files/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Load scaler
with open('pkl_files/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Calculate accuracies (you need to save test data or calculate during training)
# For now, we'll use placeholder values - you should calculate these during training
model_accuracies = {
    'Logistic Regression': 50.68,
    'Decision Tree': 70.08,
    'Random Forest': 63.16,
    'XGBoost': 83.29,
    'K-Nearest Neighbors': 63.31,
    'Support Vector Machine': 61.95
}



@app.route('/')
def home():
    return render_template('index.html', accuracies=model_accuracies)

@app.route('/about')
def about():
    return render_template('about.html')    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        features = [
            float(request.form['ph']),
            float(request.form['Hardness']),
            float(request.form['Solids']),
            float(request.form['Chloramines']),
            float(request.form['Sulfate']),
            float(request.form['Conductivity']),
            float(request.form['Organic_carbon']),
            float(request.form['Trihalomethanes']),
            float(request.form['Turbidity'])
        ]
        
        # Create DataFrame with feature names
        feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction using XGBoost
        prediction = xgb_model.predict(input_scaled)[0]
        prediction_proba = xgb_model.predict_proba(input_scaled)[0]
        if prediction == 1:
            result = "Potable (Safe to Drink)" 
        else:
            result = "Not Potable (Unsafe to Drink)"
        confidence = round(max(prediction_proba) * 100, 2)
        
        return render_template('index.html', 
                             prediction=result,
                             confidence=confidence,
                             accuracies=model_accuracies,
                             input_values=dict(zip(feature_names, features)))
    
    except Exception as e:
        return render_template('index.html', 
                             error=f"Error: {str(e)}",
                             accuracies=model_accuracies)

if __name__ == '__main__':
    app.run(debug=True)