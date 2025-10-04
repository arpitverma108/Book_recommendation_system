from books_recommendation_system.components.stage_00_data_ingestion import DataIngestion
from books_recommendation_system.components.stage_01_data_validation import DataValidation
from books_recommendation_system.components.stage_02_data_transformation import DataTransformation
from books_recommendation_system.components.stage_03_model_trainer import ModelTrainer
class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_data_ingestion(self):
        self.data_ingestion.initiate_data_ingestion()
        self.data_validation.initiate_data_validation()
        self.data_transformation.initiate_data_transformation()
        self.model_trainer.initiate_model_trainer()
