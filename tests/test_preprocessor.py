import pdb
import unittest
import pandas as pd
import numpy as np
from preprocessing.preprocessor import Preprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': ['T', 'F', 'T', np.nan, 'F'],
            'B': [np.nan, 'p', np.nan, 'g', 'c'],
            'C': [1, 2, 3, 4, 5],
            'D': [1.0, np.nan, np.nan, 5.0, 4.3]
            })
        self.preprocessor = Preprocessor(self.df)

    def test_drop(self):
        self.preprocessor.drop(method={"missing": 50}, drop_list=['A'])
        df = self.preprocessor.df
        cols = set(df.columns.to_list())
        self.assertEqual(cols, {'B', 'D', 'C'})

    def test_one_hot(self):
        self.preprocessor.one_hot()
        df = self.preprocessor.df
        cols = set(df.columns.to_list())
        self.assertEqual(cols, {'C', 'D', 'A_T', 'B_p', 'B_g', 'B_c'})
        

if __name__ == "__main__":
    unittest.main()
