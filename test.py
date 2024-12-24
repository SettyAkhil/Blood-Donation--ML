from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('svc_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.form.to_dict()

    # Extract features from the input data
    features = [int(data['months_since_last_donation']), int(data['number_of_donations']),
                int(data['total_volume_donated']), int(data['months_since_first_donation'])]
    
    # Convert features to numpy array and reshape
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Return the prediction as JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
