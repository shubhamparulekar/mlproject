import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingClassifier,
                              AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
      self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array, test_array):
        try:
           logging.info("Splitting training and testing data")
           X_train, y_train, X_test, y_test=(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
           )
           models={
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'AdaBoost': AdaBoostRegressor(),
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False),
                'XGBoost': XGBRegressor()
           }

           params={
                'Random Forest': {
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Gradient Boosting': {
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [58,16,32,64,128,256]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },
                'Linear Regression': {},
                'Decision Tree': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                },
                'CatBoost': {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                'XGBoost': {
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [58,16,32,64,128,256]
                }
           }

           model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models=models,param=params)
           best_model_score = max(model_report.values())

           model_names=list(model_report.keys())
           model_scores=list(model_report.values())

           best_model_index = model_scores.index(best_model_score)
           best_model_name = model_names[best_model_index]
           
        #    best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
           
           if best_model_score < 0.6:
               raise CustomException("No best model found with sufficient accuracy",)
           logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
           
           best_model=models[best_model_name]

           save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
           )

           predicted =best_model.predict(X_test)
           r2_square = r2_score(y_test, predicted)
           return r2_square



        except Exception as e:
            raise CustomException(e, sys)