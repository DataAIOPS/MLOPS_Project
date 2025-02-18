import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open("./model/best_model_LogisticRegression.pkl","rb"))

@app.route("/",methods=["GET","POST"])
def predict():
    if request.method=="POST":
        try:
            features = [
                float(request.form['age']), float(request.form['income']), float(request.form['loanamount']),
                float(request.form['creditscore']), float(request.form['monthsemployed']),
                float(request.form['numcreditlines']), float(request.form['interestrate']), float(request.form['loanterm']),
                float(request.form['dtiratio']), int(request.form['education']), int(request.form['employmenttype']),
                int(request.form['maritalstatus']), int(request.form['hasmortgage']), int(request.form['hasdependents']),
                int(request.form['loanpurpose']), int(request.form['hascosigner'])
            ]

            input_array = np.array([features]).reshape(1,-1)
            prediction = model.predict(input_array)[0]

            result = "Defalt" if prediction==1 else "No Default"
            return render_template("index.html",prediction_text=f"Prediction={result}")
        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {str(e)}")
    else:
        return render_template("index.html", prediction_text="")


if __name__ == "__main__":
    app.run(debug=True)