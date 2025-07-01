# Any data we want ot read from some data source 

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# This is data ingestion class so if any data requried we can give this class
# There is decorator @data class by use of it we can directly define our class variable.
@dataclass
class DataIngectionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngectionConfig()

    # if data is store in the data base (mango db) then we will read like below
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            # Here we can reade from mango db or api or any kind of db but for simplisity we are reading from the csv file as of now.
            df = pd.read_csv("src/notebook/data/stud.csv")
            # We alway best practice to write the logs so if any error willl come we can easily finde where the error is by the help of the log.
            logging.info("Read the dataset as dataframe")
            
            # Creating the directorises for all the train test data and exist_ok =True will make sure if the file is allready exist it will not create again it from new it will use that file.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            # create train test split
            train_set, test_set = train_test_split(df, test_size=0.2,random_state=42)
            # save the train test split data in the csv
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is compleated")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
            
# initiate it for running 
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)






