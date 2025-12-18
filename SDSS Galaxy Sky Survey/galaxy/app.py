
from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load Model
model_path = 'kmodel.pkl'
selected_features_names = None

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)
        # Handle both bundled and direct model saves for robustness
        if isinstance(model_bundle, dict) and 'model' in model_bundle:
            model = model_bundle['model']
            scaler = model_bundle.get('scaler')
            selected_features_names = model_bundle.get('selected_features')
        else:
            model = model_bundle
            scaler = None # Expecting raw input if no scaler
else:
    model = None
    print("Warning: kmodel.pkl not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('inner-page.html', prediction_text="Error: Model not loaded.")

    try:
        # Extract features from form
        input_data = request.form.to_dict()
        
        # Determine which features to use and in what order
        if selected_features_names:
            # Filter and order inputs based on what the model expects
            features = [float(input_data[feature]) for feature in selected_features_names]
        else:
            # Fallback for models trained without feature selection saving (or old pickle)
            # Make sure we read them in a consistent order corresponding to typical training: u, g, r, i, z, redshift
            default_order = ['u', 'g', 'r', 'i', 'z', 'redshift']
            features = [float(input_data[f]) for f in default_order if f in input_data]
            
            # If the user somehow didn't provide all inputs or order is ambiguous, this might still fail, 
            # but usually form inputs match.
            if len(features) == 0:
                 # Fallback to values() but this is risky as order is not guaranteed in older python/framework versions
                 features = [float(x) for x in request.form.values()]

        final_features = [np.array(features)]
        
        # Scale if scaler is available
        if scaler:
            final_features = scaler.transform(final_features)
            
        prediction = model.predict(final_features)
        
        # Mapping back to class name 
        output = "STARFORMING" if prediction[0] == 1 else "STARBURST"
        
        return render_template('inner-page.html', prediction_text=f'Galaxy Type: {output}')
    except Exception as e:
        return render_template('inner-page.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, port=2222)