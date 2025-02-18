## Collect data from kaggle anf store it in raw_data

import os
import pandas as pd

KAGGLE_DATA_PATH = "data/kaggle/Loan_default.csv"
RAW_DATA_PATH = "data/raw_data/Loan_default.csv"

def load_and_save_data():

    df = pd.read_csv(KAGGLE_DATA_PATH)

    df.to_csv(RAW_DATA_PATH, index=False)
    
if __name__ == "__main__":
    load_and_save_data()