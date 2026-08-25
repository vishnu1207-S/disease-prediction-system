import pandas as pd
import numpy as np
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_data(num_samples=2000):
    data = []
    
    diseases = ['Flu', 'Heart Disease', 'Diabetes', 'Hypertension', 'Common Cold', 'Asthma', 'Healthy']
    
    for i in range(1, num_samples + 1):
        patient_id = f"P{i:04d}"
        disease = random.choice(diseases)
        
        # Base probabilities or ranges depending on disease
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
            gender = random.choice(['Male', 'Female', 'Male']) # slightly higher in males in this synthetic set
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
        
        # Add some noise (missing values) to simulate real-world data
        if random.random() < 0.05: bp = np.nan
        if random.random() < 0.05: sugar = np.nan
        if random.random() < 0.05: cholesterol = np.nan
            
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
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("Generating synthetic healthcare data...")
    df = generate_data(2000)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'healthcare_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data generated and saved to {output_path}")
    
    print("\nDataset Info:")
    print(df.info())
    print("\nDisease Distribution:")
    print(df['Disease'].value_counts())
