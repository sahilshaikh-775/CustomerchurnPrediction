from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
from src.customerchurn.utils import read_yaml

import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import os
import sys


@dataclass
class DataValidationConfig:
    schema_path:str = os.path.join('configs','schema.yaml')
    report_path:str = os.path.join('artifacts','validation','validation_report.txt')


class DataValidation:
    def __init__(self,config: DataValidationConfig = DataValidationConfig()):
        self.config = config
        self.schema = read_yaml(self.config.schema_path)
    
    def write_report(self,Lines: List[str]) -> None:
        os.makedirs(os.path.dirname(self.config.report_path), exist_ok=True)
        with open(self.config.report_path,"w",encoding='utf-8') as f:
            f.write('\n'.join(Lines))
    
    def validate_dataframe(self,df:pd.DataFrame) -> Tuple[bool,List[str]]:
      
        report:List[str] = []
        overall_ok = True

        # required
        required_cols = self.schema.get('required_columns',[])
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            overall_ok=False
            report.append(f"[FAIL] Missing required columns: {missing}")
        else:
            report.append("[PASS] All required columns are present")

        if not overall_ok:
            return overall_ok, report
        
        # 2. Target Checks
        target_cols = self.schema.get('target_column')
        allowed_values = self.schema.get('target_allowed_values',[])

        if target_cols not in df.columns:
            overall_ok = False
            report.append(f"[FAIL] Target column '{target_cols}' not found") 
        else:
            bad_vals = sorted(set(df[target_cols].dropna().unique()) - set(allowed_values))
            if bad_vals:
                overall_ok=False
                report.append(f"[FAIL] Target column  has invalid values: {bad_vals}(allowed={allowed_values})")
            else:
                report.append(f"[PASS] Target values are valid")
        

        # 2. Unique Columns

        for col in self.schema.get('unique_columns',[]):
            if df[col].duplicated().any():
                overall_ok=False
                n_dup=int(df[col].duplicated().sum())

                report.append(f"[FAIL] Uniqe columns {col} has {n_dup} duplicate values")
            else:
                report.append(f"[PASS] Unique columns {col} has no duplicates")
        
        # 3. Numeric columns
        for col in self.schema.get('numeric_columns',[]):
            s = df[col]

            if s.dtype == "object":
                s = s.astype(str).str.strip()
                s = s.replace({"":pd.NA, "nan":pd.NA })
            
            coerced = pd.to_numeric(s,errors="coerce")
            n_bad = int(((~s.isna()) & (coerced.isna())).sum())

            if n_bad > 0:
                overall_ok=False
                report.append(f"[FAIL] Numerical Column {col} has {n_bad} non-numeric values")
            else:
                report.append(f"[PASS] Numeric column {col} is numeric or convertible")

        # 4. Integer Checks
        for col in self.schema.get('integer_columns',[]):
            coerced=pd.to_numeric(df[col], errors='coerce')
            non_int = coerced.dropna().apply(lambda x:float(x).is_integer() is False).sum()

            if non_int>0:
                overall_ok=False
                report.append(f"[FAIL] Integer column {col} has {non_int} non-integer values")
            else:
                report.append(f"[PASS] Integer column {col} is integer-like")

        # 5. Categorical values
        cat_cols = self.schema.get('categorical_columns',[])
        missing_cat_col = [c for c in cat_cols if c not in df.columns]
        if missing_cat_col:
            overall_ok=False
            report.append(f"[FAIL] Missing Categorical column: {missing_cat_col}")
        else:
            report.append(f"[PASS] All Categorical Column present")


        self.write_report(report)
        return overall_ok,report


    def initiate_data_validation(self,train_path:str,test_path:str) :
        try:
            logging.info("Starting data validation ")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            train_ok, train_report = self.validate_dataframe(train_df)
            test_ok, test_report = self.validate_dataframe(test_df)

            if not (train_ok and test_ok):
                raise ValueError(f"Data Validation failed. Check report at {self.config.report_path}")
            
            final_report = ["===== TRAIN ====="] + train_report + ["", "===== TEST ====="] + test_report
            self.write_report(final_report)

            logging.info(f"Validation is complete. Data report saved at {self.config.report_path}")


            return self.config.report_path

        except Exception as e:
            raise CustomException(e,sys)