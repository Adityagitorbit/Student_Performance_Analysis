import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numerical_column = ["writing_score", "reading_score"]
            categorical_column = [
                    "gender", 
                    "race_ethnicity", 
                    "parental_level_of_education", 
                    "lunch", 
                    "test_preparation_course"
                ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                    ]
                )
            
            logging.info(f"Numerical Columns Standard: {numerical_column}")
            
            
            categorical_pipeline = Pipeline(
                steps=[
                    
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            
            logging.info(f"Categorical columns: {categorical_column}")
            
            
            preprocessor = ColumnTransformer(
                transformers=[
                    
                    ("num_pipeline", numerical_pipeline, numerical_column),
                    ("cat_pipeline", categorical_pipeline, categorical_column)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            logging.info("Data Transformation Initiated")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Data loaded successfully")
            
            logging.info("Obtaining Preprocessor")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            logging.info("Data Preprocessing Started")
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            
            logging.info("Applying Preprocessing object on train data frame")
            
            train_df_transformed = preprocessor_obj.fit_transform(input_feature_train_df)
            test_df_transformed = preprocessor_obj.transform(input_feature_test_df)
            
            logging.info("Data Preprocessing Completed")
            
            logging.info("Data Transformation Completed")
            
            train_arr = np.c_[
                train_df_transformed,
                np.array(target_feature_train_df)
            ]
            
            
            test_arr = np.c_[
                test_df_transformed,
                np.array(target_feature_test_df)
            ]
            
            save_object(
                
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                
            )
            
        except Exception as e:
            raise CustomException(e, sys)