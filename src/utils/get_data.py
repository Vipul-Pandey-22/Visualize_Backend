import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.exception import CustomException
from src.logger import logging
from .artifacts import save_data


def get_classification_data(params):
    try:
        logging.info("Creating dataset for classification")
        X, y = make_classification(
        n_samples=params.n_samples,
        n_features=params.n_features,
        n_classes=params.n_classes,
        n_clusters_per_class=params.n_clusters_per_class,
        random_state=params.random_state
        )
        logging.info("Saving data into artifacts")
        save_data(X, y, params.n_features, params.n_classes)
        print(X)
        data = {"X": X.tolist(), "y": y.tolist()}
        logging.info("Classification data is created")
        return data
    except Exception as e:
        raise CustomException(e, sys)
    
def get_regression_data(params):
    try:
        logging.info("Creating dataset for classification")
        X, y = make_regression(
        n_samples=params.n_samples,
        n_features=params.n_features,
        random_state=params.random_state
        )
        logging.info("Saving data into artifacts")
        save_data(X, y, params.n_features, None)
        data = {"X": X.tolist(), "y": y.tolist()}
        logging.info("Classification data is created")
        return {"status": "Ok", "data": data}
    except Exception as e:
        raise CustomException(e, sys)