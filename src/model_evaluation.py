## Evalaute the trained model performance and pushed it mlflow model registry if model passed the baseline

import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

processed_data_path="data/processed_data/"
MODEL_SAVE_PATH="model"

X_train = pd.read_csv(os.path.join(processed_data_path,"X_train.csv"))
y_train = pd.read_csv(os.path.join(processed_data_path,"y_train.csv"))
X_test = pd.read_csv(os.path.join(processed_data_path,"X_test.csv"))
y_test = pd.read_csv(os.path.join(processed_data_path,"y_test.csv"))

def model_eval():

    model_files = [f for f in os.listdir(MODEL_SAVE_PATH) if f.endswith('.pkl')]

    model_file = model_files[0]  # Load the first saved model
    model_path = os.path.join(MODEL_SAVE_PATH, model_file)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Loaded model: {model_file}")

    y_pred_test = model.predict(X_test)

    y_pred_train = model.predict(X_train)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    class_report_test = classification_report(y_test, y_pred_test)

    accurac_train = accuracy_score(y_train, y_pred_train)
    conf_matri_train = confusion_matrix(y_train, y_pred_train)
    class_report_train = classification_report(y_train, y_pred_train)

    print(accuracy_test)
    print(accurac_train)


if __name__ == "__main__":
    model_eval()