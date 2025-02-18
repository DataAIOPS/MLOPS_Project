## Do required processing on raw_data and store it in processed_data

## Collect data from kaggle anf store it in raw_data

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


PROCESSED_DATA_PATH = "data/processed_data/"
RAW_DATA_PATH = "data/raw_data/Loan_default.csv"
PLOT_PATH = "plots"

def train_test_split(data):
    non_loan_data = (data[data["default"] == 0])
    loan_data = (data[data["default"] == 1])    
    test_non_loan_data = non_loan_data[:67000]
    test_loan_data = loan_data[:8800]
    train_non_loan_data = non_loan_data[67000:]
    train_loan_data = loan_data[8800:]
    test_data = pd.concat([test_loan_data,test_non_loan_data])
    train_data = pd.concat([train_non_loan_data,train_loan_data])

    X_test = test_data.drop('default', axis=1)
    y_test = test_data['default']

    X_train = train_data.drop('default', axis=1)
    y_train = train_data['default']

    return X_test,y_test,X_train,y_train

def data_encoding(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = pd.factorize(data[col])[0]
    return data

def feature_scaling(data):
    scaler = MinMaxScaler()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

def clean_data(data):
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    data.columns = data.columns.str.replace('-', '_').str.replace('.', '_')
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    return data

def EDA_data(data):
    # Histogram for Age
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'], bins=15, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig("plots/age_hist.png")

    # Count plot for Employment Type
    plt.figure(figsize=(10, 6))
    sns.countplot(x='EmploymentType', data=data)
    plt.title('Employment Type Distribution')
    plt.savefig("plots/emp_dist.png")

    # Count plot for Default
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Default', data=data)
    plt.title('Default Distribution')
    plt.savefig("plots/default_dist.png")

    # credit score
    plt.figure(figsize=(10,5))
    plt.hist(data['CreditScore'], bins=20, edgecolor='black')
    plt.title('Credit Score Distribution')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.savefig("plots/credit_score_dist.png")

def load_and_save_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df.drop(columns=['LoanID'],inplace=True)
    print(df.columns)
    EDA_data(df)
    df = data_encoding(df)
    df = clean_data(df)
    df = feature_scaling(df)
    X_test,y_test,X_train,y_train = train_test_split(df)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_train.to_csv(os.path.join(PROCESSED_DATA_PATH,"X_train.csv"),index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_PATH,"X_test.csv"),index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_PATH,"y_train.csv"),index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_PATH,"y_test.csv"),index=False)
    
if __name__ == "__main__":
    load_and_save_data()