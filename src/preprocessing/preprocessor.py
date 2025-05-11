import pdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Preprocessor():
    def __init__(self, df, drop_method=None, drop_list=None):
        self.df = df
        self.drop_method = drop_method
        self.drop_list = drop_list
        self.dropped_cols = []
        self.onehot_encoders = {}
        self.cols = self.df.columns

    def run(self):
        self.drop()
        self.one_hot()

    #def drop(self, method=None, drop_list=None):
    def drop(self):
        method = self.drop_method
        drop_list = self.drop_list
        if (method is not None):
            if "missing" in method.keys():
                percent_missing = self.df.isnull().sum() * 100 / len(self.df)
                mask = percent_missing > method["missing"]
                cols = self.df.columns[mask]
                self.dropped_cols = self.dropped_cols.append(cols)
                self.df = self.df.drop(cols, axis=1)

        if drop_list is not None:
            for name in drop_list:
                self.df = self.df.drop(name, axis=1)
                self.dropped_cols = self.dropped_cols.append(name)

        self.cols = self.df.columns

    def one_hot(self):
        df_cat = self.df.select_dtypes(include='object')
        for name in df_cat.columns:
            current_col = self.df[name].to_frame()
            pdb.set_trace()
            nan_mask = self.df[name].isna()
            cats = self.df[name].unique()
            #encoder = OneHotEncoder(categories=cats.tolist(), handle_unknown='ignore')
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            cleaned = [x for x in cats if not pd.isna(x)]
            hot = pd.get_dummies(self.df[name], prefix=name, dtype=int)
            hot[nan_mask] = np.nan
            if set(cleaned) == {'T', 'F'}:
                hot = hot[name + "_T"]
            self.df = self.df.join(hot)
            self.df = self.df.drop(name, axis=1)
            # self.onehot_dict[name] = 
    

                
        
if __name__ == "__main__":
    test_df = pd.DataFrame({
          'A': ['T', 'F', 'T', np.nan, 'F'],
          'B': [np.nan, 'p', np.nan, 'g', 'c'],
          'C': [1, 2, 3, 4, 5],
          'D': [1.0, np.nan, np.nan, 5.0, 4.3]
        })
    preprocessor = Preprocessor(test_df)
    #preprocessor.drop(method={"missing": 50})
    preprocessor.one_hot()
    pdb.set_trace()


