# Disease Prediction System (Healthcare Analytics) 🩺

An AI-powered web application that predicts probable diseases based on a patient's vitals, medical history, and symptoms. Built with Python, Scikit-learn, and Flask, featuring a modern glassmorphism web interface.

## Features
- **Machine Learning Models**: Uses a trained Random Forest Classifier to diagnose diseases based on synthetic patient data.
- **Dynamic Web Interface**: A sleek, dark-mode glassmorphism UI built with HTML and Vanilla CSS.
- **Real-time Inference**: Instantly processes form data and returns an AI prediction.

## Tech Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3

## Installation and Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/disease-prediction-system.git
   cd disease-prediction-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the AI model:**
   This will generate a synthetic dataset and save the trained models in the `models/` directory.
   ```bash
   python train.py
   ```

4. **Start the Web Server:**
   ```bash
   python app.py
   ```

5. **Open the App:**
   Visit `http://127.0.0.1:5000` in your web browser!

## Project Structure
- `train.py`: Data generation, preprocessing, and model training script.
- `app.py`: Flask backend server.
- `templates/index.html`: The web interface.
- `static/style.css`: UI styling and animations.
- `requirements.txt`: Python package dependencies.
- `models/`: Directory containing exported `.pkl` files (generated after running `train.py`).

## Disclaimer
This project uses synthetic data for educational purposes and is **not** a substitute for professional medical advice or diagnosis. Always consult a healthcare professional.
