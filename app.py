from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('model/hypo.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    features = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 
               'on_antithyroid_medication', 'sick', 'pregnant', 
               'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
               'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 
               'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    
    input_data = []
    for feature in features:
        value = float(request.form[feature])
        input_data.append(value)
    
    # Convert to numpy array and reshape
    final_features = np.array(input_data).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Map prediction to class name
    class_names = {
        0: 'Negative',
        1: 'Compensated Hypothyroid',
        2: 'Primary Hypothyroid',
        3: 'Secondary Hypothyroid'
    }
    
    result = class_names[prediction[0]]
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)