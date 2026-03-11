from src.customerchurn.logger import logging
from src.customerchurn.exception import CustomException
from src.customerchurn.utils import read_yaml

import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_input: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    test_size: float
    random_state: int
    stratify_col: str 

class DataIngestion:
    def __init__(self, config_path:str = "configs/config.yaml"):
        cfg = read_yaml(config_path)

        self.ingestion_config = DataIngestionConfig(
            raw_data_input=cfg['data']['raw_data_input'],  
            raw_data_path=cfg['data']['raw_data_path'],  
            train_data_path=cfg['data']['train_data_path'],
            test_data_path=cfg['data']['test_data_path'],
            test_size=cfg['split']['test_size'],
            random_state=cfg['split']['random_state'],
            stratify_col=cfg['split']['stratify_col'],
        )

    def initiate_data_ingestion(self):
        try:

            logging.info(f"Starting data ingestion")


            df=pd.read_csv(self.ingestion_config.raw_data_input)
            logging.info(f'Reading data from: {self.ingestion_config.raw_data_input} | shape: {df.shape}')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)


            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info(f"Raw file created at: {self.ingestion_config.raw_data_path}")

            col = self.ingestion_config.stratify_col
            if col not in df.columns:
                raise ValueError("Stratify Col not present")
            
            if df[col].isna().any():
                n_null= int(df[col].isna().sum())
                raise ValueError("Stratify column: {col} contains {n_null} null values")


            train_set,test_set = train_test_split(
                df,
                test_size=self.ingestion_config.test_size,
                random_state=self.ingestion_config.random_state,
                stratify=df[col]
                )
            

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info(f"Train file created at: {self.ingestion_config.train_data_path} | shape: {train_set.shape}")

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"Test file created at: {self.ingestion_config.test_data_path} | shape: {test_set.shape}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)