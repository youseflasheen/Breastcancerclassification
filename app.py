import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('finalized_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the feature names in the correct order
feature_names = [
    'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
    'smoothness_se', 'compactness_se', 'concave points_se', 'symmetry_se',
    'symmetry_worst'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map the form input values to the features
        features = [float(request.form.get(name)) for name in feature_names]
        final_features = [np.array(features)]
        
        # Scale the features
        scaled_features = scaler.transform(final_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        output = "Malignant (cancerous)" if prediction == 'M' else "Benign (non-cancerous)"
        
        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception:
        return render_template('index.html', prediction_text='Invalid input. Please enter numerical values.')

if __name__ == '__main__':
    app.run(debug=True)
