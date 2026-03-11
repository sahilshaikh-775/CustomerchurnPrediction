from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
from src.customerchurn.utils import read_yaml, save_object,load_object

import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder


@dataclass
class DataTransformationConfig:
    config_path:str = os.path.join('configs','config.yaml')
    schma_path:str = os.path.join('configs','schema.yaml')
    preprocessor_path:str = os.path.join('artifacts','preprocessor.pkl')
    X_train_path:str = os.path.join('artifacts','X_train.npy')
    X_test_path:str = os.path.join('artifacts','X_test.npy')
    y_train_path:str = os.path.join('artifacts','y_train.npy')
    y_test_path:str = os.path.join('artifacts','y_test.npy')


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config
        self.cfg = read_yaml(self.config.config_path)
        self.schema = read_yaml(self.config.schma_path)

        self.target = self.cfg["target"]["name"]
        self.drop_cols = self.cfg.get("features",{}).get("drop_columns",[])

        self.numeric_cols = self.schema.get("numeric_columns",[])
        self.categorical_cols = self.schema.get("categorical_columns",[])

        self.categorical_cols = [c for c in self.categorical_cols if c != self.target]

    def clean_telco(self,df:pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if "TotalCharges" in df.columns:
            s = df["TotalCharges"].astype(str).str.strip().replace({"":np.nan,"nan":np.nan})
            df["TotalCharges"] = pd.to_numeric(s,errors="coerce")

        if "tenure" in df.columns:
            mask = (df["tenure"] == 0) & (df["tenure"].isna())
            df.loc[mask,"TotalCharges"] = 0.0

        return df
    

    def get_preprocessor(self) -> ColumnTransformer:
        try:
            num_pipeline = Pipeline(steps= [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            cat_pipleine = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num",num_pipeline,self.numeric_cols),
                    ("cat",cat_pipleine,self.categorical_cols)
                ],
                remainder="drop"
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path:str,test_path:str):
        try:
            logging.info("Data Transformation Started")

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            train_df = self.clean_telco(train_df)
            test_df = self.clean_telco(test_df)

            y_train = (train_df[self.target]=="Yes").astype(int).values
            y_test = (test_df[self.target]=="Yes").astype(int).values

            X_train = train_df.drop(columns=[self.target] + self.drop_cols,errors="ignore")
            X_test = test_df.drop(columns=[self.target] + self.drop_cols, errors="ignore")

            preprocessor = self.get_preprocessor()
            X_train_trr = preprocessor.fit_transform(X_train)
            X_test_trr = preprocessor.transform(X_test)

            save_object(self.config.preprocessor_path,preprocessor)

            np.save(self.config.X_train_path,X_train_trr)
            np.save(self.config.X_test_path,X_test_trr)
            np.save(self.config.y_train_path,y_train)
            np.save(self.config.y_test_path,y_test)
            
            logging.info("Data Transformation is complete")
            logging.info(f"Save preprocessor in: {self.config.preprocessor_path}")

            return (
                self.config.X_train_path, self.config.X_test_path,
                self.config.y_train_path, self.config.y_test_path,
                self.config.preprocessor_path

            )

        except Exception as e:
            raise CustomException(e,sys)