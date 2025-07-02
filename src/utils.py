# U tils have all the common files which we are going to use and import in entire projrct

import os
import sys 

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)        
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            # Taking the all madels one by on in model variable
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Applying greed search cv
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            # Setting the best parameter to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Fitiing the model
            #model.fit(X_train, y_train)

            # Doing prediction on X_train and y_train data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            

            # r2 score evaluation on the training model
            train_model_score = r2_score(y_train, y_train_pred)

            # r2 score evaluation on the test mode
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Creating report of the models
            report[list(models.keys())[i]]  =  test_model_score

            return report

        
    except Exception as e:
        raise CustomException(e,sys)   