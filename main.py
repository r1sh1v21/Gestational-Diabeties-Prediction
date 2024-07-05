import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the trained classifier
with open('frauddetection.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Flask app
app = Flask(__name__)

# Route to render the HTML form
@app.route("/")
def home():
    return render_template("home.html")

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Extract the input values from the form
    features = [int(request.form[field]) for field in ["age","numPregnancies","gestationPrevPregnancy","bmi","hdl","familyHistory","largeChildBirthDefault","pcos","sysBP","diaBP","ogtt","hemoglobin","sedentaryLifestyle","prediabetes"]]

    try:
        # Make prediction
        prediction = model.predict([features])[0]

        # Convert prediction to human-readable format
        prediction_text = "yeas" if prediction == 1 else "no"
    
        # Return the prediction as a response
        return render_template("prediction_result.html", prediction=prediction_text)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
