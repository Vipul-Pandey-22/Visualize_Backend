import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split

def save_data(X, y, n, classes):
    try:
        logging.info("Saving into artifacts")
        df_X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(n)])
        df_y = pd.DataFrame(y, columns=["Target"])
        # Combine features and target into one DataFrame
        df = pd.concat([df_X, df_y], axis=1)
        # Create directory if it doesn't exist
        os.makedirs('artifacts/classification', exist_ok=True)
        # Save DataFrame to CSV

        if classes:
            os.makedirs('artifacts/classification', exist_ok=True)
            df.to_csv('artifacts/classification/generated_data.csv', index=False)
            logging.info("Performing train test split....")
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
            train_data.to_csv('artifacts/classification/train_data.csv', index=False)
            test_data.to_csv('artifacts/classification/test_data.csv', index=False)
            logging.info("Test and Training data saved in artifacts")
        

        os.makedirs('artifacts/regression', exist_ok=True)
        df.to_csv('artifacts/regression/generated_data.csv', index=False)
        logging.info("Performing train test split....")
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        train_data.to_csv('artifacts/regression/train_data.csv', index=False)
        test_data.to_csv('artifacts/regression/test_data.csv', index=False)
        logging.info("Test and Training data saved in artifacts")
        
    except Exception as e:
        raise CustomException(e, sys)