import os, pdb, pprint
from preprocessing.base_model_preprocessor import build_preprocessor
from preprocessing.cleaner import clean_data
import numpy as np
import pandas as pd
import sqlite3, math
import xgboost as xgb

#from xgboost import XGBClassifier
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

class xgbModel():
    def __init__(self, param_grid):
        self.best_params = None
        self.best_model = None
        self.param_grid = param_grid

    def _get_stats(self, y_test, predictions, print_values=False):
        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions, average='binary')
        score = f1_score(y_test, predictions, average='binary')
        if print_values:
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            print("Precision: %.2f%%" % (precision * 100.0))
            print('Recall: %.3f' % (recall * 100.0))
            print('F-Measure: %.3f' % (score * 100.0))

        out = {}
        out['accuracy'] = accuracy
        out['precision'] = precision
        out['recall'] = recall
        out['score'] = score
        return out

    def fit(self, X, y):
        print("Cross validating to find hyperparameters...")
        # Increase penalty for false negatives
        model_tuned = xgb.XGBClassifier(scale_pos_weight=10)
        grid_search = GridSearchCV(model_tuned,
                                   self.param_grid,
                                   cv=5,
                                   scoring='accuracy')
        # Fit the GridSearchCV object to the training data
        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        self.best_params['scale_pos_weight'] = 10
        print("Training model...")
        self.best_model = xgb.XGBClassifier(**self.best_params)
        self.best_model.fit(X, y)

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Model has not been trained")
        return self.best_model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return classification_report(y, y_pred, output_dict=True)


if __name__ == "__main__":
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.5, 0.7, 1]
    }
    DATA_DIRECTORY = "../../data/processed_data"
    train_path = os.path.join(DATA_DIRECTORY, "processed_X_train_transaction_data.csv")
    test_path = os.path.join(DATA_DIRECTORY, "processed_X_test_transaction_data.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df.isFraud
    y_test = test_df.isFraud
    X_train = train_df.drop(columns="isFraud")
    X_test = test_df.drop(columns="isFraud")
    model = xgbModel(param_grid)

    model.fit(X_train, y_train)
    report = model.evaluate(X_test, y_test)
    pprint.pprint(report)
