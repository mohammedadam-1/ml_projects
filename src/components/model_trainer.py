import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_trainer_path: str=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split the train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "KNN" : KNeighborsRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "GradientBoosting" : GradientBoostingRegressor()

            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                               models=models)
            
            best_model_name = max(model_report, key=model_report.get)

            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]
            logging.info('Best model found on train and test dataset')

            save_object(
                file_path = self.model_trainer_config.model_trainer_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_Score = r2_score(y_test, predicted)

            return r2_Score
           
        except Exception as e:
            raise CustomException(e, sys)


