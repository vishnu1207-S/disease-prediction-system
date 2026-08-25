from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import traceback

app = Flask(__name__)

MODEL_DIR = 'models'
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'disease_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    model_columns = joblib.load(os.path.join(MODEL_DIR, 'model_columns.pkl'))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please run train.py first.'}), 500

    try:
        data = request.json
        df_input = pd.DataFrame(0, index=[0], columns=model_columns)
        
        df_input['Age'] = float(data.get('age', 0))
        df_input['Blood_Pressure'] = float(data.get('blood_pressure', 0))
        df_input['Sugar_Level'] = float(data.get('sugar_level', 0))
        df_input['Cholesterol'] = float(data.get('cholesterol', 0))
        
        numerical_cols = ['Age', 'Blood_Pressure', 'Sugar_Level', 'Cholesterol']
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
        
        try:
            encoded_gender = label_encoder.transform([data.get('gender')])[0]
        except:
            encoded_gender = 0
        if 'Gender' in df_input.columns: 
            df_input['Gender'] = encoded_gender
            
        history_val = data.get('medical_history', 'None')
        history_col = f"Medical_History_{history_val}"
        if history_col in df_input.columns:
            df_input[history_col] = 1
            
        symptoms_list = data.get('symptoms', [])
        for symptom in symptoms_list:
            if symptom in df_input.columns:
                df_input[symptom] = 1
                
        prediction = model.predict(df_input)[0]
        return jsonify({'prediction': prediction})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
