import pandas as pd
import functools as ft
from sklearn.preprocessing import StandardScaler,normalize

class Processing:

    def __init__(self):
        pass

    def remove_outliers(self,df:pd.DataFrame,col_names:list[str],threshold=1.5) -> pd.DataFrame :
        assert set(col_names).issubset(set(df.columns))
        def quantile(s):
            assert isinstance(s,pd.Series)
            q_025=s.quantile(.25)
            q_075=s.quantile(.75)
            iq=q_075-q_025
            return ~((s>q_075+threshold*iq) & (s<q_025-threshold*iq))
        mask=ft.reduce(lambda x,y:x & y,[quantile(df[col]) for col in col_names])
        return df.loc[mask]

    def standard_scaler(self,df:pd.DataFrame):
        """
        :param df: dataframe and all columns represent quantitative variable
        :return: dataframe rescaled
        """
        return StandardScaler().fit_transform(df)

    def normalize(self,df:pd.DataFrame,norm="l2"):
        """
        :param df: dataframe and all columns represent quantitative variable
        :return: dataframe normalized
        """
        return normalize(df,norm=norm)


