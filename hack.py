from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

# Load the trained model
with open('svc_model.pkl', 'rb') as f:
    svc_model = pickle.load(f)

# Route to the home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template("index.html")

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve input values from the form
        months_last_donation = int(request.form['months_last_donation'])
        num_donations = int(request.form['num_donations'])
        total_volume_donated = int(request.form['total_volume_donated'])
        months_since_first_donation = int(request.form['months_since_first_donation'])
        
        # Check if the number of donations is less than or equal to 15 or total volume donated is less than or equal to 1400
        if num_donations <= 15 or total_volume_donated <= 1400:
            result = 'Eligible for Donation'
        else:
            # Preprocess the input data
            input_data = [[months_last_donation, num_donations, total_volume_donated, months_since_first_donation]]
            scaler = StandardScaler()
            input_data_scaled = scaler.fit_transform(input_data)
            
            # Make prediction
            prediction = svc_model.predict(input_data_scaled)
            
            # Determine prediction label
            if prediction[0] == 0:
                result = 'Not Eligible for Donation'
            else:
                result = 'Eligible for Donation'
        
        return render_template('result.html', prediction=result)
if __name__ == '__main__':
    app.run(debug=True)
