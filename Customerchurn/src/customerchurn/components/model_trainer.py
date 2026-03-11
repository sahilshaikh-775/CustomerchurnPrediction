from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
from src.customerchurn.utils import save_object,load_object,read_yaml

import numpy as np
import os
import json
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
                roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,confusion_matrix,classification_report
)


@dataclass
class ModelTrainerConfig:
    config_path:str = os.path.join('configs','config.yaml')
    model_path: str = os.path.join('artifacts','model.pkl')
    metrics_path:str = os.path.join('artifacts','metrics','meterics.json')

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config
        self.cfg = read_yaml(self.config.config_path)

        self.model_path = self.cfg.get('artifacts',{}).get('model_path',self.config.model_path)
        self.metrics_path = self.cfg.get('artifacts',{}).get('metrics_path',self.config.metrics_path)

        params = self.cfg.get('model',{}).get('params',{})
        self.max_iter = int(params.get('max_iter',2000))
        self.class_weight = params.get('class_weight','balanced')

    def _best_threshold_by_f1(self,y_true,y_prob,thresholds=None):
        if thresholds is None:
            thresholds = np.linspace(0.05,0.95,19)

        
        best = {"threshold":0.5,"f1":-1,"precision":None,"recall":None}
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true,y_pred,zero_division=0)
            if f1 > best['f1']:
                best = {
                    "threshold":float(t),
                    "f1":float(f1),
                    "precision":float(precision_score(y_true,y_pred,zero_division=0)),
                    "recall": float(recall_score(y_true,y_pred,zero_division=0))
                }
            
        return best
    

    def initiate_model_trainer(self,X_train_path,X_test_path,y_train_path,y_test_path):
        try:
            logging.info('Starting model training - Logistic Regression')
            
            X_train = np.load(X_train_path, allow_pickle=True)
            X_test = np.load(X_test_path,allow_pickle=True)
            y_train = np.load(y_train_path,allow_pickle=True)
            y_test = np.load(y_test_path,allow_pickle=True)

            model = LogisticRegression(
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                n_jobs=None
            )

            model.fit(X_train,y_train)

            y_prob = model.predict_proba(X_test)[:,1]

            roc_auc = float(roc_auc_score(y_test,y_prob))
            pr_auc = float(average_precision_score(y_test,y_prob))

            best = self._best_threshold_by_f1(y_test,y_prob)
            thr = best["threshold"]

            y_pred = (y_prob >= thr).astype(int)

            cm = confusion_matrix(y_test, y_pred, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()

            metrics = {
                "model": "LogisticRegression",
                "roc_auc": roc_auc,
                "pr_auc":pr_auc,
                "best_threshold_by_f1": best,
                "confusion_matrix": {
                    "tn":int(tn),"fp":int(fp),
                    "fn":int(fn),"tp":int(tp)
                },
                "classification_report": classification_report(y_test,y_pred,labels=[0,1],output_dict=True,zero_division=0)
            }

            os.makedirs(os.path.dirname(self.metrics_path),exist_ok=True)
            with open(self.metrics_path, 'w',encoding='utf-8') as f:
                json.dump(metrics,f,indent=2)
            
            save_object(self.model_path,model)

            logging.info(f"Model save to:{self.model_path}")
            logging.info(f"Metrics save at:{self.metrics_path}")
            logging.info(f"ROC-AUC:{roc_auc:.4f} | PR-AUC:{pr_auc:.4f} | Best F1 Threshold:{thr}")

            return self.model_path, self.metrics_path

        except Exception as e:
            raise CustomException(e,sys)