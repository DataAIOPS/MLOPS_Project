## Write a flask API which will download model from mlflow model registry and serve the predictions
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model

model = pickle.load(open("./model/best_model_LogisticRegression.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Extract form data
            features = [
                float(request.form['age']), float(request.form['income']), float(request.form['loanamount']),
                float(request.form['creditscore']), float(request.form['monthsemployed']),
                float(request.form['numcreditlines']), float(request.form['interestrate']), float(request.form['loanterm']),
                float(request.form['dtiratio']), int(request.form['education']), int(request.form['employmenttype']),
                int(request.form['maritalstatus']), int(request.form['hasmortgage']), int(request.form['hasdependents']),
                int(request.form['loanpurpose']), int(request.form['hascosigner'])
            ]
            
            # Convert to numpy array and reshape for prediction
            input_array = np.array([features]).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            
            result = "Default" if prediction == 1 else "No Default"
            return render_template("index.html", prediction_text=f"Prediction: {result}")
        
        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {str(e)}")
    
    return render_template("index.html", prediction_text="")

if __name__ == "__main__":
    app.run(debug=True)