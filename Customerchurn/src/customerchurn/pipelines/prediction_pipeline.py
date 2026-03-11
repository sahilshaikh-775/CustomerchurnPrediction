import pandas as pd
import numpy as np

from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
from src.customerchurn.utils import read_yaml,load_object
import sys
import os
import json


class PredictionPipeline:
    def __init__(self):
        self.processor_path = "artifacts/preprocessor.pkl"
        self.model_path = "artifacts/models/best_model.pkl"
        self.metrics_path = "artifacts/metrics/best_model_metrics.json"

        self.processor = load_object(self.processor_path)
        self.model = load_object(self.model_path)

        with open(self.metrics_path,'r',encoding='utf-8') as f:
            metrics = json.load(f)
        
        self.threshold = float(metrics.get("best_threshold_from_cv",0.35))
    
    def predict(self,input_dict:dict):
        try:
            df = pd.DataFrame([input_dict])
            
            X = self.processor.transform(df)
            proba = float(self.model.predict_proba(X)[0, 1])
            pred = int(proba >= self.threshold)
            label = "Yes" if proba>=self.threshold else "No"

            return {
                "churn_probability": proba,
                "churn_prediction": label,
                "threshold_used": self.threshold
            }

        except Exception as e:
            raise CustomException(e,sys)