# app.py

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Initialize Flask app
app = Flask(__name__)

# Step 1: Prepare the training data
Yobs = np.array([
    137,118,124,124,120,129,122,142,128,114,
    132,130,130,112,132,117,134,132,121,128
])
W = np.array([
    0,1,1,1,0,1,1,0,0,1,
    1,0,0,1,0,1,0,0,1,1
])
X = np.array([
    19.8,23.4,27.7,24.6,21.5,25.1,22.4,29.3,20.8,20.2,
    27.3,24.5,22.9,18.4,24.2,21.0,25.9,23.2,21.6,22.8
])

# Step 2: Train the regression model when app starts
X_reg = sm.add_constant(pd.DataFrame({'W': W, 'X': X}))  # Add intercept
model = sm.OLS(Yobs, X_reg).fit()

# Step 3: Define prediction route
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get input W and X from query parameters
        W_input = float(request.args.get('W'))
        X_input = float(request.args.get('X'))

        # Prepare input feature for prediction
        features = np.array([[1, W_input, X_input]])  # Add constant
        prediction = model.predict(features)[0]

        return jsonify({
            'predicted_engagement_score': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Step 4: Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
