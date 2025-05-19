import os, pdb
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
from sklearn.model_selection import GridSearchCV

DATA_DIRECTORY = "../../data"
PROCESSED_X_TRAIN_NAME = "processed_X_train_transaction_data.csv"
PROCESSED_X_TEST_NAME = "processed_X_test_transaction_data.csv"
PROCESSED_Y_TRAIN_NAME = "processed_Y_train_transaction_data.csv"
PROCESSED_Y_TEST_NAME = "processed_Y_test_transaction_data.csv"

def main():
    ## ============================ Load Data ============================
    train_path = os.path.join(DATA_DIRECTORY, "raw_data/train_transaction.csv")
    train_transaction_data = pd.read_csv(train_path) 
    #test_transaction_data = pd.read_csv('../data/raw_data/test_transaction.csv') 

    ## =========================== Clean Data ============================
    cat_cols = ["ProductCD", 
                        "card1", "card2", "card3", "card4", "card5", "card6",
                        "addr1", "addr2",
                        "P_emaildomain", "R_emaildomain",
                        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"
                    ]
    union_list = list(set(cat_cols).union({"TransactionID", "isFraud", }))
    sub_df = train_transaction_data.drop(columns=union_list)
    numeric_cols = sub_df.columns
    X_train, X_test, y_train, y_test = clean_data(train_transaction_data, split=True, cat_cols=cat_cols, y_col="isFraud")
    ## ========================= Preprocess Data =========================
    prep = build_preprocessor(numeric_cols, cat_cols)
    X_train_processed = prep.fit_transform(X_train)
    X_test_processed = prep.transform(X_test)
    feat_names = prep.get_feature_names_out()
    X_train_processed = pd.DataFrame(X_train_processed, columns=feat_names)
    X_train_processed['isFraud'] = y_train.reset_index(drop=True)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feat_names)
    X_test_processed['isFraud'] = y_test.reset_index(drop=True)
    ## ============================ Save Data ============================
    processed_X_train_path = os.path.join(DATA_DIRECTORY, "processed_data", PROCESSED_X_TRAIN_NAME)
    processed_X_test_path = os.path.join(DATA_DIRECTORY, "processed_data", PROCESSED_X_TEST_NAME)
    #processed_Y_train_path = os.path.join(DATA_DIRECTORY, "processed_data", PROCESSED_Y_TRAIN_NAME)
    #processed_Y_test_path = os.path.join(DATA_DIRECTORY, "processed_data", PROCESSED_Y_TEST_NAME)

    X_train_processed.to_csv(processed_X_train_path, index=False)
    X_test_processed.to_csv(processed_X_test_path, index=False)
    #y_train.to_csv(processed_X_train_path)
    #y_test.to_csv(processed_X_train_path)

if __name__ == "__main__":
    main()
