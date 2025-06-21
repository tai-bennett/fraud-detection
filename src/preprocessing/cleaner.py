import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def clean_data(df, y_col=None, split=True, cat_cols=[]):
    # change categorical variables of type float to int
    for name in cat_cols:
        if df.dtypes[name] == float:
            df[name] = df[name].astype('Int64')

    # if specified output, separate it
    if y_col is not None:
        y = df[y_col]
        X = df.drop(columns=y_col)
    else:
        X = df

    # if split
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.2)
        #y_train = y_train.reset_index(drop=True)
        #y_test = y_test.reset_index(drop=True)
        return X_train, X_test, y_train, y_test
    else:
        return X

def impute(df, cat_cols, numeric_cols):
    for name in df.columns:
        if name in cat_cols:
            df[name] = df[name].fillna(df[name].mode()[0])
        elif name in numeric_cols:
            df[name] = df[name].fillna(df[name].mean())
    return df

def mySMOTE(df, y_name, seed=123):
    y = df[y_name]
    X = df.drop(columns=y_name)
    smote = SMOTE(random_state=seed)
    X_res, y_res = smote.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_res, columns=X.columns)
    df_resampled['label'] = y_res
    return df_resampled


