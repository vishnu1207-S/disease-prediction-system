import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

warnings.filterwarnings('ignore')

def generate_healthcare_data(num_samples=2500):
    print("--- Step 1: Generating Synthetic Healthcare Data ---")
    data = []
    diseases = ['Flu', 'Heart Disease', 'Diabetes', 'Hypertension', 'Common Cold', 'Asthma', 'Healthy']
    
    np.random.seed(42)
    random.seed(42)
    
    for i in range(1, num_samples + 1):
        patient_id = f"P{i:04d}"
        disease = random.choice(diseases)
        
        if disease == 'Flu':
            age = random.randint(5, 80)
            gender = random.choice(['Male', 'Female'])
            symptoms = random.sample(['Fever', 'Cough', 'Fatigue', 'Headache', 'Sore Throat', 'Muscle Ache'], random.randint(2, 4))
            bp = random.randint(100, 130)
            sugar = random.randint(80, 110)
            cholesterol = random.randint(150, 220)
            history = random.choice(['None', 'Asthma'])
            
        elif disease == 'Heart Disease':
            age = random.randint(45, 90)
            gender = random.choice(['Male', 'Female', 'Male'])
            symptoms = random.sample(['Chest Pain', 'Shortness of Breath', 'Fatigue', 'Dizziness', 'Palpitations'], random.randint(2, 4))
            bp = random.randint(130, 180)
            sugar = random.randint(90, 150)
            cholesterol = random.randint(200, 300)
            history = random.choice(['Hypertension', 'Diabetes', 'None', 'Hypertension'])
            
        elif disease == 'Diabetes':
            age = random.randint(30, 85)
            gender = random.choice(['Male', 'Female'])
            symptoms = random.sample(['Increased Thirst', 'Frequent Urination', 'Fatigue', 'Blurred Vision', 'Weight Loss'], random.randint(2, 4))
            bp = random.randint(110, 150)
            sugar = random.randint(150, 350)
            cholesterol = random.randint(180, 260)
            history = random.choice(['None', 'Hypertension', 'Obesity'])
            
        elif disease == 'Hypertension':
            age = random.randint(35, 90)
            gender = random.choice(['Male', 'Female'])
            symptoms = random.sample(['Headache', 'Shortness of Breath', 'Nosebleeds', 'Dizziness', 'None'], random.randint(1, 3))
            bp = random.randint(140, 200)
            sugar = random.randint(80, 130)
            cholesterol = random.randint(180, 250)
            history = random.choice(['None', 'Diabetes', 'Obesity'])
            
        elif disease == 'Common Cold':
            age = random.randint(2, 80)
            gender = random.choice(['Male', 'Female'])
            symptoms = random.sample(['Runny Nose', 'Sore Throat', 'Cough', 'Congestion', 'Sneezing'], random.randint(2, 4))
            bp = random.randint(90, 130)
            sugar = random.randint(70, 110)
            cholesterol = random.randint(140, 220)
            history = random.choice(['None', 'Asthma'])
            
        elif disease == 'Asthma':
            age = random.randint(5, 70)
            gender = random.choice(['Male', 'Female'])
            symptoms = random.sample(['Shortness of Breath', 'Wheezing', 'Chest Tightness', 'Cough'], random.randint(2, 3))
            bp = random.randint(100, 130)
            sugar = random.randint(80, 120)
            cholesterol = random.randint(150, 230)
            history = random.choice(['None', 'Allergies'])
            
        else: # Healthy
            age = random.randint(18, 65)
            gender = random.choice(['Male', 'Female'])
            symptoms = ['None'] if random.random() > 0.1 else [random.choice(['Headache', 'Fatigue'])]
            bp = random.randint(100, 120)
            sugar = random.randint(70, 100)
            cholesterol = random.randint(120, 200)
            history = 'None'

        symptoms_str = ", ".join(symptoms)
        
        # Introduce missing values (~3%)
        if random.random() < 0.03: bp = np.nan
        if random.random() < 0.03: sugar = np.nan
        if random.random() < 0.03: cholesterol = np.nan
            
        data.append({
            'Patient_ID': patient_id,
            'Age': age,
            'Gender': gender,
            'Symptoms': symptoms_str,
            'Blood_Pressure': bp,
            'Sugar_Level': sugar,
            'Cholesterol': cholesterol,
            'Medical_History': history,
            'Disease': disease
        })
        
    return pd.DataFrame(data)

def predict_disease(patient_data, model, scaler, X_columns, label_encoder):
    df_input = pd.DataFrame(0, index=[0], columns=X_columns)
    
    df_input['Age'] = patient_data['Age']
    df_input['Blood_Pressure'] = patient_data['Blood_Pressure']
    df_input['Sugar_Level'] = patient_data['Sugar_Level']
    df_input['Cholesterol'] = patient_data['Cholesterol']
    
    numerical_cols = ['Age', 'Blood_Pressure', 'Sugar_Level', 'Cholesterol']
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
    
    try:
        encoded_gender = label_encoder.transform([patient_data['Gender']])[0]
    except:
        encoded_gender = 0
    if 'Gender' in df_input.columns: df_input['Gender'] = encoded_gender
        
    history_col = f"Medical_History_{patient_data['Medical_History']}"
    if history_col in df_input.columns:
        df_input[history_col] = 1
        
    for symptom in patient_data['Symptoms']:
        if symptom in df_input.columns:
            df_input[symptom] = 1
            
    return model.predict(df_input)[0]

def main():
    # 1. Data Generation
    df = generate_healthcare_data(3000)
    df.to_csv('healthcare_dataset.csv', index=False)
    print(">> Dataset generated and saved to 'healthcare_dataset.csv'")
    
    # 2. Data Preprocessing
    print("\n--- Step 2: Data Preprocessing ---")
    print("Handling missing values...")
    df['Blood_Pressure'] = df['Blood_Pressure'].fillna(df['Blood_Pressure'].median())
    df['Sugar_Level'] = df['Sugar_Level'].fillna(df['Sugar_Level'].median())
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())

    print("Encoding categorical variables...")
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df = pd.get_dummies(df, columns=['Medical_History'], drop_first=True)

    print("Feature Engineering (One-Hot Encoding Symptoms)...")
    symptoms_dummies = df['Symptoms'].str.get_dummies(sep=', ')
    df_processed = pd.concat([df, symptoms_dummies], axis=1)
    df_processed.drop(['Symptoms', 'Patient_ID'], axis=1, inplace=True)
    
    print(f">> Processed dataset shape: {df_processed.shape}")
    
    # 3. Model Training
    print("\n--- Step 3: Model Training ---")
    X = df_processed.drop('Disease', axis=1)
    y = df_processed['Disease']

    scaler = StandardScaler()
    numerical_cols = ['Age', 'Blood_Pressure', 'Sugar_Level', 'Cholesterol']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(kernel='linear', random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"Trained {name:25s} | Accuracy: {accuracy:.4f}")

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\n>> Best Model Selected: {best_model_name} with Accuracy {results[best_model_name]:.4f}")

    # 4. Model Evaluation
    print("\n--- Step 4: Model Evaluation ---")
    y_pred_best = best_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_best))

    # 5. Prediction Inference
    print("\n--- Step 5: Real-Time Prediction Test ---")
    new_patient = {
        'Age': 55,
        'Gender': 'Male',
        'Symptoms': ['Chest Pain', 'Shortness of Breath'],
        'Blood_Pressure': 160,
        'Sugar_Level': 110,
        'Cholesterol': 260,
        'Medical_History': 'Hypertension'
    }
    
    print("Testing new patient data:")
    for k, v in new_patient.items():
        print(f"  {k}: {v}")
        
    predicted_disease = predict_disease(new_patient, best_model, scaler, X.columns, label_encoder)
    print(f"\n>> PREDICTED DISEASE: === {predicted_disease.upper()} ===")

if __name__ == "__main__":
    main()
