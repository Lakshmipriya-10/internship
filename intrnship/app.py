from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        quiz_total = float(request.form['quiz_total'])
        attendance = float(request.form['attendance'])
        overall_performance = float(request.form['overall_performance'])

        # Create a DataFrame with the input data
        new_data = pd.DataFrame({
            'Quiz Total': [quiz_total],
            'Attendance %': [attendance],
            'Overall Performance': [overall_performance]
        })

        # Make predictions
        predictions = model.predict(new_data)

        # Get the predicted eligibility
        predicted_eligibility = 'Eligible' if predictions[0] == 1 else 'Not Eligible'

        # Return the result
        return render_template('result.html', prediction=predicted_eligibility)
    except Exception as e:
        return f"Error: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
