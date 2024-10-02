import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from fastapi import APIRouter
from typing import List
from pydantic import BaseModel
from src.utils.get_data import get_classification_data, get_regression_data

router = APIRouter()

# Input model for classification parameters
class ClassificationParams(BaseModel):
    n_samples: int
    n_features: int
    n_classes: int
    n_clusters_per_class: int
    random_state: int

class RegressionParams(BaseModel):
    n_samples: int = 1000
    n_features: int = 4
    random_state: int = 41

@router.post("/generate_classification")
async def create_classification(params: ClassificationParams):
    try:
        data = get_classification_data(params)
        return {"status": "Ok", "data": data}
        
    except Exception as e:
        raise CustomException(e, sys)
    
@router.post("/generate_regression")
async def create_regression(params: RegressionParams):
    try:
        data = get_regression_data(params)
        return {"status": "Ok", "data": data}
    except Exception as e:
        raise CustomException(e, sys)


