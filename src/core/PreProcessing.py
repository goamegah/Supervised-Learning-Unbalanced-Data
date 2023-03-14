import pandas as pd
from sklearn.compose import make_column_selector as selector


class PreProcessing:
    def __init__(self):
        pass

    def dropna(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    def imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        qualitatives = selector(dtype_include=object)(df)
        return df.apply(lambda s: s.fillna(s.mode()[0]) if s.name in qualitatives \
            else s.fillna(s.mean()), axis=0)
