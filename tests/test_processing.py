import pandas as pd
from src.core import Processing

if __name__ == "__main__":
    file_path="/home/khaldi/Documents/data_app_machine/whole_data.csv"
    sep=","
    df=pd.read_csv(file_path,sep=sep)
    process=Processing()
    df=process.preprocessing(df,method="imputation")
    result=process.df_to_numerical(df,l_order=["MaritalStatus"])

