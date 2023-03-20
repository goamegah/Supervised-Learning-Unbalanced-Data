import numpy as np
import pandas as pd
import functools as ft
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from src.core.PreProcessing import PreProcessing


class Processing:

    def __init__(self):
        pass

    def preprocessing(self, dataset: object, method="drop") -> pd.DataFrame:
        """
        :param dataset: dataset given by user
        :param method: if "drop":drop NaN values elif "imputation": do imputation
        :return:
        """
        preprocess = PreProcessing()
        if method == "drop" and isinstance(dataset, pd.DataFrame):
            return preprocess.dropna(dataset)
        elif method == "imputation" and isinstance(dataset, pd.DataFrame):
            return preprocess.imputation(dataset)

    def summary(self, df: pd.DataFrame) -> dict:
        constant_features = [c for c in df.columns if df[c].min() == df[c].max()]
        return {
            "Missing Values": df.isna().sum(),
            "Constant Features": constant_features
        }

    def remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[[c for c in df.columns if c not in self.summary(df)["Constant Features"]]]

    def remove_outliers(self, df: pd.DataFrame, threshold=1.5) -> pd.DataFrame:
        qualitatives = selector(dtype_include=object)(df)
        quantitatives = [c for c in df.columns if c not in qualitatives]

        def quantile(s):
            assert isinstance(s, pd.Series)
            q_025 = s.quantile(.25)
            q_075 = s.quantile(.75)
            iq = q_075 - q_025
            return ~((s > q_075 + threshold * iq) | (s < q_025 - threshold * iq))

        mask = ft.reduce(lambda x, y: x & y, [quantile(df[col]) for col in quantitatives])
        return df.loc[mask].copy()

    def standard_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        :param df: dataframe and all columns represent quantitative variable
        :return: dataframe rescaled
        """
        return StandardScaler().fit_transform(df)

    def normalize(self, df: pd.DataFrame, norm="l2") -> pd.DataFrame:
        """
        :param df: dataframe and all columns represent quantitative variable
        :return: dataframe normalized
        """
        return normalize(df, norm=norm)

    def transform_categorical(self, df: pd.DataFrame, l_order: list = None, drop="first") -> dict:
        qualitatives = selector(dtype_include=object)(df)
        ordinal_cols = []
        if qualitatives:
            if l_order:
                if not set(l_order).issubset(set(qualitatives)):
                    raise Exception(f"{l_order} is not include to {qualitatives}")
                else:
                    ordinal_cols = [c for c in qualitatives if c in l_order]
                    ordinal_encoder = OrdinalEncoder()
                    ordinals = ordinal_encoder.fit_transform(df[ordinal_cols])
                    ordinal_mapping = ordinal_encoder.categories_
        nominal_encoder = OneHotEncoder(sparse_output=False, drop=drop)
        nominal_cols = [c for c in qualitatives if c not in ordinal_cols]
        nominals = nominal_encoder.fit_transform(df[nominal_cols])
        nominal_mapping = nominal_encoder.get_feature_names_out()
        if len(ordinal_cols):
            column_names = list(ordinal_cols) + list(nominal_mapping)
            return \
                {
                    "numericals": pd.DataFrame(
                        data=np.hstack([ordinals, nominals]),
                        columns=column_names),
                    "mapping": dict(zip(ordinal_cols, ordinal_mapping))
                }
        else:
            column_names = list(nominal_mapping)
            return {"numericals": pd.DataFrame(data=nominals, columns=column_names)}

    def df_to_numerical(self, df: pd.DataFrame, l_order: list = None, drop="first") -> dict:
        numericals_columns = self.transform_categorical(df, l_order, drop=drop)
        qualitatives = selector(dtype_include=object)(df)
        quantitatives = [c for c in df.columns if c not in qualitatives]
        df_transform = numericals_columns["numericals"].reset_index(drop=True) \
            .join(df[quantitatives].reset_index(drop=True), how="inner")

        if "mapping" in numericals_columns.keys():
            return \
                {
                    "df_transform": df_transform,
                    "ordinal_mapping": numericals_columns["mapping"]
                }
        else:
            return \
                {
                    "df_transform": df_transform
                }

# %%
