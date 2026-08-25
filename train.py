import pandas as pd
import numpy as np
import random
import warnings
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

def main():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # 1. Data Generation
    df = generate_healthcare_data(3000)
    print(">> Dataset generated.")
    
    # 2. Data Preprocessing
    print("\n--- Step 2: Data Preprocessing ---")
    df['Blood_Pressure'] = df['Blood_Pressure'].fillna(df['Blood_Pressure'].median())
    df['Sugar_Level'] = df['Sugar_Level'].fillna(df['Sugar_Level'].median())
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())

    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    df = pd.get_dummies(df, columns=['Medical_History'], drop_first=True)

    symptoms_dummies = df['Symptoms'].str.get_dummies(sep=', ')
    df_processed = pd.concat([df, symptoms_dummies], axis=1)
    df_processed.drop(['Symptoms', 'Patient_ID'], axis=1, inplace=True)
    
    # 3. Model Training
    print("\n--- Step 3: Model Training ---")
    X = df_processed.drop('Disease', axis=1)
    y = df_processed['Disease']

    scaler = StandardScaler()
    numerical_cols = ['Age', 'Blood_Pressure', 'Sugar_Level', 'Cholesterol']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f">> Model Trained (Random Forest) | Accuracy: {accuracy:.4f}")

    # 4. Save Models for Web App
    print("\n--- Step 4: Saving Models ---")
    joblib.dump(model, 'models/disease_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(list(X.columns), 'models/model_columns.pkl')
    print(">> Models saved successfully to the 'models/' directory!")

if __name__ == "__main__":
    main()
