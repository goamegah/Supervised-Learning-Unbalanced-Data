import numpy as np
import pandas as pd
import functools as ft
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from src.core.PreProcessing import PreProcessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class Processing:

    def __init__(self):
        pass

    def preprocessing(self, dataset: object, method="drop") -> pd.DataFrame:
        """
        Preprocess the input dataset by either dropping missing values or imputing them.
        :param dataset: the input dataset
        :param method: the method to use for preprocessing - either "drop" or "imputation"
        :return: the preprocessed dataset

        """
        preprocess = PreProcessing()
        if method == "drop" and isinstance(dataset, pd.DataFrame):
            return preprocess.dropna(dataset)
        elif method == "imputation" and isinstance(dataset, pd.DataFrame):
            return preprocess.imputation(dataset)

    def summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of the input DataFrame, including the number of missing values and constant features.
        :param df: the input DataFrame
        :return: a dictionary containing the number of missing values and constant features
        """
        # find all features with constant values
        constant_features = [c for c in df.columns if df[c].min() == df[c].max()]
        return {
            "Missing Values": df.isna().sum(),
            "Constant Features": constant_features
        }

    def remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all constant features from the input DataFrame.
        :param df: the input DataFrame
        :return: the DataFrame with constant features removed
        """
        # remove all columns with constant values from the DataFrame
        return df[[c for c in df.columns if c not in self.summary(df)["Constant Features"]]]

    def remove_outliers(self, df: pd.DataFrame, threshold=1.5) -> pd.DataFrame:
        """
        Remove all outliers from the input DataFrame using the specified threshold.
        :param df: the input DataFrame
        :param threshold: the threshold to use for identifying outliers
        :return: the DataFrame with outliers removed
        """
        # separate the quantitative and qualitative variables in the DataFrame
        qualitatives = selector(dtype_include=object)(df)
        quantitatives = [c for c in df.columns if c not in qualitatives]

        def quantile(s):
            """
            Define a helper function for identifying outliers using quantiles.
            :param s: a Series of quantitative data
            :return: a boolean mask indicating whether each value is an outlier
            """
            assert isinstance(s, pd.Series)
            q_025 = s.quantile(.25)
            q_075 = s.quantile(.75)
            iq = q_075 - q_025
            return ~((s > q_075 + threshold * iq) | (s < q_025 - threshold * iq))
        # use the helper function to identify and remove outliers
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
        """
        Transforms categorical variables in the dataframe to ordinal or one-hot encoded
        based on the given order or drop strategy.
        :param df: Input DataFrame to transform
        :param l_order: List of columns to encode as ordinal. If None, all categorical columns will be one-hot encoded.
        :param drop: Strategy to use for dropping one of the one-hot encoded columns. Must be one of {"first", "if_binary", None}.
        :return: Dictionary containing the transformed numerical DataFrame and the ordinal mapping (if applicable).
        """
        qualitatives = selector(dtype_include=object)(df)
        ordinal_cols = []
        if qualitatives:
            if l_order:
                # If a specific order is given, check if all columns are in the dataframe
                if not set(l_order).issubset(set(qualitatives)):
                    raise Exception(f"{l_order} is not include to {qualitatives}")
                else:
                    # Subset the ordinal columns and use OrdinalEncoder to transform them
                    ordinal_cols = [c for c in qualitatives if c in l_order]
                    ordinal_encoder = OrdinalEncoder()
                    ordinals = ordinal_encoder.fit_transform(df[ordinal_cols])
                    ordinal_mapping = ordinal_encoder.categories_
        # Use OneHotEncoder to transform the nominal columns
        nominal_encoder = OneHotEncoder(sparse_output=False, drop=drop)
        nominal_cols = [c for c in qualitatives if c not in ordinal_cols]
        nominals = nominal_encoder.fit_transform(df[nominal_cols])
        nominal_mapping = nominal_encoder.get_feature_names_out()
        # If there are any ordinal columns, merge them with the one-hot encoded nominal columns
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
        """
        Converts the input DataFrame to a numerical representation by transforming its categorical variables.
        :param df: Input DataFrame to convert
        :param l_order: List of columns to encode as ordinal. If None, all categorical columns will be one-hot encoded.
        :param drop: Strategy to use for dropping one of the one-hot encoded columns.
        :return: Dictionary containing the transformed numerical DataFrame and the ordinal mapping (if applicable).
        """

        numericals_columns = self.transform_categorical(df, l_order, drop=drop)
        qualitatives = selector(dtype_include=object)(df)
        quantitatives = [c for c in df.columns if c not in qualitatives]
        # Merge the transformed categorical variables with the quantitative variables
        df_transform = numericals_columns["numericals"].reset_index(drop=True) \
            .join(df[quantitatives].reset_index(drop=True), how="inner")
        # If ordinal columns were transformed, return the mapping as well
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

    def oversampling(self, df_X: pd.DataFrame, s_y: pd.Series, method="SMOTE", random_state=42, perc_minority="auto"):
        # Function to perform oversampling
        if method == "SMOTE":
            # If method is SMOTE, use SMOTE algorithm for oversampling
            sm = SMOTE(random_state=random_state, sampling_strategy=perc_minority)
            # Fit and resample the data using SMOTE
            return sm.fit_resample(df_X, np.array(s_y).reshape(-1))

    def undersampling(self, df_X: pd.DataFrame, s_y: pd.Series, method="RandomUnderSampling", perc_minority=.5):
        # Function to perform undersampling
        if method == "RandomUnderSampling":
            # If method is RandomUnderSampling, use RandomUnderSampler algorithm for undersampling
            undersample = RandomUnderSampler(sampling_strategy=perc_minority)
            # Fit and resample the data using RandomUnderSampler
            return undersample.fit_resample(df_X, np.array(s_y).reshape(-1))
