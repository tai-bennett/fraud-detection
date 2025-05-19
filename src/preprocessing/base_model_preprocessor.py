import numpy as np
import pandas as pd
import sqlite3, math
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .base import DropHighNaNColumns, RareCategoryGrouper

def build_preprocessor(numeric_cols, categorical_cols):
    enc = OneHotEncoder(handle_unknown='ignore')

    cat_transformer = Pipeline([
        ('drop_nan_cols', DropHighNaNColumns(threshold=0.75)),
        ('rare_class_combine', RareCategoryGrouper(min_freq=3000)),
        ('onehot', enc)
    ])

    num_transformer = Pipeline([
        ('drop_nan_cols', DropHighNaNColumns(threshold=0.75)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_transformer, categorical_cols),
        ('num', num_transformer, numeric_cols)
    ])

    return preprocessor
    
