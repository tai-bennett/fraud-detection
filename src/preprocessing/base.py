from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DropHighNaNColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.columns_to_keep_ = []

    def fit(self, X, y=None):
        # Store columns where the fraction of NaNs is below the threshold
        self.columns_to_keep_ = X.columns[X.isnull().mean() < self.threshold].tolist()
        return self

    def transform(self, X):
        return X[self.columns_to_keep_].copy()

    def get_feature_names_out(self, input_features=None):
        # Return the final set of columns after dropping
        return self.columns_to_keep_

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=10):
        self.min_freq = min_freq
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        X_str = X.astype(str)
        for col in X.columns:
            freq = X[col].value_counts()
            self.frequent_categories_[col] = freq[freq >= self.min_freq].index.tolist()
        return self

    def transform(self, X):
        X_str = X.astype(str)
        X_out = X_str.copy()
        for col in X.columns:
            X_out[col] = X_out[col].where(X_out[col].isin(self.frequent_categories_[col]), other='Other')
        self.feature_names_out_ = X_out.columns
        return X_out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
