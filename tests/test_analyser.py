import pandas as pd
from src.core.Analyser import Analyser
from src.core.Processing import Processing
import matplotlib.pyplot as plt

if __name__ == "__main__":
    outcome="Attrition"
    positive_mod="Yes"

    file_path="/home/khaldi/Documents/data_app_machine/whole_data.csv"
    sep=","
    df=pd.read_csv(file_path,sep=sep)
    process=Processing()
    df_c=process.preprocessing(df,method="imputation")
    #result=process.df_to_numerical(df,l_order=["MaritalStatus"])
    a=Analyser()
    fig,ax=plt.subplots()
    #a.bar_chart(col_name="MaritalStatus",ax=ax)
    #a.correlation_heatmap(df_c)
    summary=a.summary(df)
    quanti=summary["features"]["quantitative_features"]
    a.prop_churn_by_numerical(df,outcome,positive_mod,quanti[0],ax,alpha=0.9)



