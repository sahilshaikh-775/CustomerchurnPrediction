from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
from src.customerchurn.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.customerchurn.components.data_validation import DataValidation, DataValidationConfig
import sys


if __name__ == "__main__":

    try:
        data_ingestion = DataIngestion(config_path="configs/config.yaml")
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Complete")

        data_validation = DataValidation()
        report_path = data_validation.initiate_data_validation(train_path,test_path)
        logging.info("Data Validation Completed")

    except Exception as e:
        raise CustomException(e,sys)