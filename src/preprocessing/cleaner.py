import pandas as pd
from sklearn.model_selection import train_test_split

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
        return X_train, X_test, y_train, y_test
    else:
        return X
