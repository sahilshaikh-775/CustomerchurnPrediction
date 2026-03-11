from src.customerchurn.exception import CustomException
from src.customerchurn.logger import logging
import sys

from src.customerchurn.components.data_ingestion import DataIngestion
from src.customerchurn.components.data_validation import DataValidation
from src.customerchurn.components.data_transformation import DataTransformation
from src.customerchurn.components.model_trainer import ModelTrainer  
from src.customerchurn.components.model_improvement import ModelImprovement   


class TrainPipeline():
    def run(self):
        try:
            logging.info("=====Train Pipeline Started=======")

            ingestion = DataIngestion(config_path="configs/config.yaml")
            train_path, test_path = ingestion.initiate_data_ingestion()


            logging.info(f"Ingestion Completed: {train_path} | {test_path}")
            
            # Validation
            validator = DataValidation()
            report_path = validator.initiate_data_validation(train_path,test_path)
            

            logging.info(f"Validation Passed. Report: {report_path}")

            transformer = DataTransformation()
            X_train_path, X_test_path, y_train_path, y_test_path, preproc_path = transformer.initiate_data_transformation(
                train_path,test_path
            )

            trainer = ModelTrainer()
            model_path,metrics_path = trainer.initiate_model_trainer(
                X_train_path,X_test_path,y_train_path,y_test_path
            )
            logging.info(f"Training model is completed. Model:{model_path} | Metrics: {metrics_path}")

            improver = ModelImprovement()
            best_model_path,best_metrics_path,leaderboard_path = improver.initiate_model_improvement(
                X_train_path,X_test_path,y_train_path,y_test_path
            )

            logging.info(f"Best model save at: {best_model_path}")
            logging.info(f"Best metrics save at: {best_metrics_path}")
            logging.info(f"Leaderboard save at: {leaderboard_path}")

            logging.info("=======Train Pipeline Completed=======")
            

            return best_model_path    
                


        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    TrainPipeline().run()