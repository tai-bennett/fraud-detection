import os, pdb
from preprocessing.base_model_preprocessor import build_preprocessor
from preprocessing.cleaner import clean_data, impute, mySMOTE
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
MASK_TRAIN_NAME = "mask_train_transaction_data.csv"
MASK_TEST_NAME = "mask_test_transaction_data.csv"
IMPUTED_X_TRAIN_NAME = "impute_X_train_transaction_data.csv"
IMPUTED_X_TEST_NAME = "impute_X_test_transaction_data.csv"
Y_TRAIN_NAME = "Y_train_transaction_data.csv"
Y_TEST_NAME = "Y_test_transaction_data.csv"

def main():
    ## ============================ Load Data ============================
    train_path = os.path.join(DATA_DIRECTORY, "raw_data/train_transaction.csv")
    train_transaction_data = pd.read_csv(train_path) 
    #test_transaction_data = pd.read_csv('../data/raw_data/test_transaction.csv') 

    ## =========================== Clean Data ============================
    print("Cleaning data...")
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


    y_train.to_csv(os.path.join(DATA_DIRECTORY, "processed_data", Y_TRAIN_NAME), index=False)
    y_test.to_csv(os.path.join(DATA_DIRECTORY, "processed_data", Y_TEST_NAME), index=False)
    ## ========================= Preprocess Data =========================
    print("Preprocessing data...")
    # build and fit preprocessor
    prep = build_preprocessor(numeric_cols, cat_cols)
    X_train_processed = prep.fit_transform(X_train)

    # preprocess test set
    X_test_processed = prep.transform(X_test)

    # impute data
    X_train_impute = impute(X_train, cat_cols, numeric_cols)
    X_test_impute = impute(X_test, cat_cols, numeric_cols)

    # preprocess data
    X_train_impute = prep.transform(X_train_impute)
    X_test_impute = prep.transform(X_test_impute)

    # rebuild data as dataframes
    feat_names = prep.get_feature_names_out()
    X_train_processed = pd.DataFrame(X_train_processed, columns=feat_names)
    X_train_imputed = pd.DataFrame(X_train_impute, columns=feat_names)
    X_train_processed['isFraud'] = y_train.reset_index(drop=True)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feat_names)
    X_test_imputed = pd.DataFrame(X_test_impute, columns=feat_names)
    X_test_processed['isFraud'] = y_test.reset_index(drop=True)
    
    ## ============================ Save Data ============================
    print("Saving processed data...")
    processed_X_train_path = os.path.join(DATA_DIRECTORY, "processed_data", PROCESSED_X_TRAIN_NAME)
    processed_X_test_path = os.path.join(DATA_DIRECTORY, "processed_data", PROCESSED_X_TEST_NAME)
    X_train_processed.to_csv(processed_X_train_path, index=False)
    X_test_processed.to_csv(processed_X_test_path, index=False)

    ## ============================ Mask Data  ============================
    print("Making masked data...")
    mask_train = X_train_processed.isna().drop(columns="isFraud")
    mask_test = X_test_processed.isna().drop(columns="isFraud")

    mask_train_path = os.path.join(DATA_DIRECTORY, "processed_data", MASK_TRAIN_NAME)
    mask_test_path = os.path.join(DATA_DIRECTORY, "processed_data", MASK_TEST_NAME)
    mask_train.to_csv(mask_train_path, index=False)
    mask_test.to_csv(mask_test_path, index=False)

    ## ========================== Imputed Data  ===========================
    print("Imputing data...")
    impute_train_path = os.path.join(DATA_DIRECTORY, "processed_data", IMPUTED_X_TRAIN_NAME)
    impute_test_path = os.path.join(DATA_DIRECTORY, "processed_data", IMPUTED_X_TEST_NAME)
    #X_train_imputed = imputate(X_train_processed)
    #X_test_imputed = imputate(X_test_processed)
    X_train_imputed.to_csv(impute_train_path, index=False)
    X_test_imputed.to_csv(impute_test_path, index=False)

    

if __name__ == "__main__":
    main()
