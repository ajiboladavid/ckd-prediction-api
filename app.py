# IMPORTS
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

# INITIALIZE FLASK APP
app = Flask(__name__)
CORS(app)

# LOAD SAVED MODELS AND PREPROCESSING
print("Loading models and preprocessing objects...")

# Load Random Forest model
with open('ckdp_random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded")

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded")

# Load feature names (to ensure correct order)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
print("Feature names loaded")
print(f"Expected features: {len(feature_names)}")

# Load label mappings (for text to number conversion)
with open('label_mappings.pkl', 'rb') as f:
    label_mappings = pickle.load(f)
print("Label mappings loaded")

# Load metadata (optional - for info endpoint)
with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
print("Metadata loaded")

print("\nAll components loaded successfully!\n")

# CREATE TEST ENDPOINT

@app.route('/')
def home():
    """Home endpoint - Test if API is running"""
    return jsonify({
        'message': 'CKD Prediction API is running!',
        'model': metadata['model_type'],
        'accuracy': f"{metadata['accuracy']*100:.2f}%",
        'endpoints': {
            '/': 'API info (you are here)',
            '/predict': 'Make CKD prediction (POST)',
            '/features': 'Get required features list (GET)'
        }
    })


# FEATURES ENDPOINT
@app.route('/features', methods=['GET'])
def get_features():
    """Return list of required features and their details"""

    # Separate numeric and categorical features
    numeric_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu',
                        'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

    categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn',
                            'dm', 'cad', 'appet', 'pe', 'ane']

    # Build response with feature details
    feature_details = {
        'total_features': len(feature_names),
        'feature_order': feature_names,
        'numeric_features': {
            'count': len(numeric_features),
            'names': numeric_features
        },
        'categorical_features': {
            'count': len(categorical_features),
            'names': categorical_features,
            'valid_values': {
                'rbc': list(label_mappings['rbc'].keys()),
                'pc': list(label_mappings['pc'].keys()),
                'pcc': list(label_mappings['pcc'].keys()),
                'ba': list(label_mappings['ba'].keys()),
                'htn': list(label_mappings['htn'].keys()),
                'dm': list(label_mappings['dm'].keys()),
                'cad': list(label_mappings['cad'].keys()),
                'appet': list(label_mappings['appet'].keys()),
                'pe': list(label_mappings['pe'].keys()),
                'ane': list(label_mappings['ane'].keys())
            }
        }
    }

    return jsonify(feature_details)


# PREDICTION ENDPOINT
@app.route('/predict', methods=['POST'])
def predict():
    """
    Make CKD prediction based on input features

    Expected input (JSON):
    {
        "age": 48,
        "bp": 80,
        "sg": 1.020,
        ... (all 24 features)
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate that data was sent
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please send JSON data with patient features'
            }), 400

        # Check if all required features are present
        missing_features = [f for f in feature_names if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing features',
                'missing': missing_features,
                'message': f'Please provide all {len(feature_names)} required features'
            }), 400

        # Encode categorical features (convert text to numbers)
        encoded_data = data.copy()

        for feature in ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
            if feature in encoded_data:
                value = encoded_data[feature]

                # Check if value is valid
                if value not in label_mappings[feature]:
                    return jsonify({
                        'error': f'Invalid value for {feature}',
                        'provided': value,
                        'valid_values': list(label_mappings[feature].keys())
                    }), 400

                # Encode the value
                encoded_data[feature] = label_mappings[feature][value]

        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([encoded_data])
        input_df = input_df[feature_names]  # Ensure correct order!

        # Scale the features (CRITICAL!)
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Interpret results
        result = 'CKD' if prediction == 0 else 'Not CKD'
        confidence_ckd = prediction_proba[0] * 100
        confidence_not_ckd = prediction_proba[1] * 100

        # Build response
        response = {
            'prediction': result,
            'prediction_code': int(prediction),
            'confidence': {
                'CKD': round(confidence_ckd, 2),
                'Not_CKD': round(confidence_not_ckd, 2)
            },
            'risk_level': 'High' if confidence_ckd > 80 else 'Medium' if confidence_ckd > 50 else 'Low',
            'recommendation': get_recommendation(result, confidence_ckd)
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


# HELPER FUNCTION: GET RECOMMENDATION
def get_recommendation(result, confidence):
    """Provide medical recommendation based on prediction"""
    if result == 'CKD':
        if confidence > 80:
            return "High likelihood of CKD. Immediate medical consultation recommended."
        elif confidence > 50:
            return "Moderate likelihood of CKD. Please consult a nephrologist for further evaluation."
        else:
            return "Possible CKD indicators detected. Consider follow-up testing."
    else:
        if confidence < 20:
            return "Low CKD risk detected, but monitor kidney health regularly."
        else:
            return "CKD unlikely based on current indicators. Maintain healthy lifestyle."

# RUN THE APP
if __name__ == '__main__':
    app.run(debug=True, port=5000)