import pandas as pd
from sklearn.compose import make_column_selector as selector


class PreProcessing:
    def __init__(self):
        pass

    # Method to drop rows with missing values
    def dropna(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    def imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Selecting columns with object data type, which are assumed to be qualitative/categorical
        qualitatives = selector(dtype_include=object)(df)
        # Filling missing values in columns with mode for qualitative/categorical data and mean for quantitative/numerical data
        return df.apply(lambda s: s.fillna(s.mode()[0]) if s.name in qualitatives else s.fillna(s.mean()), axis=0)
