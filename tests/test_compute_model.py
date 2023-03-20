import src.core.model.Processing as pro
import pandas as pd
from src.core import ComputeModel

if __name__=="__main__":
    file_path="/home/khaldi/Documents/data_app_machine/whole_data.csv"
    sep=","
    outcome="Attrition"
    positive_mod="Yes"
    process=pro.Processing()
    df=pd.read_csv(file_path,sep=sep)
    df=process.preprocessing(df,method="drop")
    df=process.remove_outliers(df)
    df=process.remove_constant_features(df)
    relevant_features=["Department","JobRole","MaritalStatus","BusinessTravel","JobSatisfaction","WorkLifeBalance","Age","NumCompaniesWorked","TotalWorkingYears"]
    df_new=df[relevant_features]
    hyperparameters= {
        "penalty":"l2",
        "C":1.,
        "solver":"liblinear",
        "multi_class":"auto",
        "n_jobs":None,
        "max_iter":250
    }
    c=ComputeModel(df[relevant_features],df[outcome],"Logistic Regression",hyperparameters=hyperparameters,process=process,positive_mod="Yes")
    c.fit()






