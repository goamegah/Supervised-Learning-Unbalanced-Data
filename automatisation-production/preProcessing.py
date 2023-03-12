import pandas as pd
CATEGORY={"qualitative","quantitative"}

class PreProcessing:
    def __init__(self):
        pass
    def dropna(self,df:pd.DataFrame) -> pd.DataFrame :
        return df.dropna()
    def imputation(self,df:pd.DataFrame,dict_categories:dict[str,str]) -> pd.DataFrame:
        assert set(dict_categories.keys()).issubset(set(df.columns))
        assert set(dict_categories.values()).issubset(CATEGORY)
        return df.apply(lambda s:s.fillna(s.mode()[0]) if dict_categories[str(s.name)]=="qualitative" \
            else s.fillna(s.mean()),axis=0)