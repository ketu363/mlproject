#  Here we are going to train diffrent diffrent model and after training we will see what accuracy we are gettin with diffrent models.

import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Creating the config file to take any input(will take whatever we required for the model training )
@dataclass
class ModelTrainerConfig:
    # we will save the our model in the Artifact file in .pkl extention 
    trained_model_file_path = os.path.join("artifacts","model.pkl")

# Model training class
class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], # Take out the last colum and feed everyhting in X_train
                train_array[:,-1],  # Lat row value as y_train value
                test_array[:,:-1], # Takte out all the value exept the last column and fill in X_ttest
                test_array[:,-1]  # Take the last column and fill in y_test
            )
            
            # Creating the dictionory of the model
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decison Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(),
                "AdaBoost Classifier": AdaBoostClassifier()

            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # To get the best model score from the dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            # If the best model score is less than 60% then will rise the exception that no best model is found
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best Found Model on both training and testing dataset")

            # Save the best model in pkl file format
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
              
            # View the predicted output for the test data
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square


            

        except Exception as e:
            raise CustomException(e, sys)
            