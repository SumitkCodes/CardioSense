from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd # To create a DataFrame for the model

# Initialize Flask app
app = Flask(__name__)

# --- Model Loading ---
# Define the path to your saved model
MODEL_PATH = 'heart_disease_random_forest_model.pkl'
model = None

# Define the expected feature names in the correct order
# These must match the columns your model was trained on
EXPECTED_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

def load_model_on_startup():
    """Load the machine learning model from disk when the app starts."""
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}. "
              "Please ensure 'heart_disease_random_forest_model.pkl' is in the same directory as app.py.")
        model = None
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        model = None

# --- API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500

    try:
        # Get data from POST request (JSON payload)
        data = request.get_json(force=True)

        # Validate and extract features in the correct order
        input_features_list = []
        for feature_name in EXPECTED_FEATURES:
            if feature_name not in data:
                return jsonify({'error': f'Missing feature in input: {feature_name}'}), 400
            input_features_list.append(data[feature_name])

        # Convert features to a Pandas DataFrame (as the model was likely trained on one)
        # with column names, for a single prediction
        input_df = pd.DataFrame([input_features_list], columns=EXPECTED_FEATURES)

        # Make prediction
        prediction_class = model.predict(input_df)
        prediction_probabilities = model.predict_proba(input_df)

        # Format the response
        result = {
            'prediction': int(prediction_class[0]),
            'prediction_label': 'Heart Disease Present' if prediction_class[0] == 1 else 'No Heart Disease',
            'confidence': {
                'no_heart_disease (class 0)': float(prediction_probabilities[0][0]),
                'heart_disease (class 1)': float(prediction_probabilities[0][1])
            },
            'input_features': data # Optionally include the input features in the response
        }
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': f'Value error in input data: {str(ve)}. Ensure all features are numerical and valid.'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# --- Main ---
if __name__ == '__main__':
    # Load the model when the Flask app starts
    load_model_on_startup()
    # Run the Flask app
    # Set debug=False for a production environment
    app.run(host='0.0.0.0', port=5000, debug=True)