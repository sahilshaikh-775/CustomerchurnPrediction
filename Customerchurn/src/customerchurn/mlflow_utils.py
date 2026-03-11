import os
import mlflow
import dagshub
from src.customerchurn.logger import logging

def setup_mlflow(cfg: dict):

    mlcfg = cfg.get("mlflow",[])
    if not mlcfg.get("enables",True):
        logging.info("MLFlow disabled in config.")
        return False
    
    exp_name = mlcfg.get("experiment_name","Churn-prediction")
    dh = mlcfg.get("dagshub", {})
    owner = dh.get("repo_owner")
    repo = dh.get("repo_name")

    if owner and repo:
        dagshub.init(repo_owner=owner,repo_name=repo,mlflow=True)
        logging.info(f"Mlflow tracking set up to Dagshub repo: {owner}/{repo}")
    else:
        logging.info("No Dagshub repo infor provided. Using local MLflow tracking")
    
    mlflow.set_experiment(exp_name)
    logging.info(f"Mlflow experiment set: {exp_name}")

    return True