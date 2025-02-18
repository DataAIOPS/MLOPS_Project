## Use processed data to train model and use mlflow to track model experiments for multiple models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import pandas as pd
import os
from sklearn.metrics import accuracy_score


processed_data_path="data/processed_data/"
MODEL_SAVE_PATH="model"

X_train = pd.read_csv(os.path.join(processed_data_path,"X_train.csv"))
y_train = pd.read_csv(os.path.join(processed_data_path,"y_train.csv"))


models = {
        # "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=100),
        # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

def train_models():
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        print(f"Training {name}...")

        # Train the model
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred_train)

        print(f"{name} Accuracy: {accuracy:.4f}")

        # Save the best-performing model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    # Save best model
    best_model_path = os.path.join(MODEL_SAVE_PATH, f"best_model_{best_model_name}.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best model saved: {best_model_name} with Accuracy: {best_accuracy:.4f} at {best_model_path}")

if __name__ == "__main__":
    train_models()
