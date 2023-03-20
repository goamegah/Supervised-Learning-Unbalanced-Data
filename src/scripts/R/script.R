if(!require("dplyr")) install.packages("dplyr")
if(!require("ggplot2")) install.packages("ggplot2")

library("dplyr")
library("ggplot2")

analysis_df=function(df){
  dims=c(nrow(df),ncol(df))
  missings=sapply(df,function(x){
    sum(is.na(x))
  })
  constants=sapply(df,function(x){
    min(x)==max(x)
  })
  return(list(dimensions=dims,missings=sum(missings) > 0,constants=sum(constants) > 0))


}


diag_barres_qualitative = function(df,col_name) {
  df_gp= df %>%
    group_by(!!sym(col_name)) %>%
    summarize( proportion=n()/nrow(df))

  return(
    ggplot(df_gp, aes(x = !!sym(col_name), y = proportion)) +
      geom_bar(stat="identity",width = 0.8) +
      ggtitle(paste("Diagramme en barres de ",col_name)) +
      xlab(col_name) + ylab("Proportion")
  )
}



get_categories = function(df){

  return(sapply(df, function(x) {
    if (is.integer(x)){
      "quantitative discrète"
    }
    if (is.numeric(x)) {
      if (sum(x-floor(x)==0)/length(x)>.99) {
        "quantitative discrète"
      } else {
        "quantitative continue"
      }
    } else if (is.character(x)) {
      "qualitative nominale"
    } else if (is.logical(x)) {
      "qualitative ordinale"
    } else if (is.factor(x)) {
      if (is.ordered(x)) {
        "qualitative ordinale"
      } else {
        "qualitative nominale"
      }
    } else {
      "Inconnu"
    }
  }))
}


histogram=function(df,outcome,positive_modality,variable_quanti){
  #hypothesis :outcome -- boolan vector
  df[outcome]= if_else(as.logical(df[outcome]==positive_modality),positive_modality,paste("Non",positive_modality))
  df=df[,c(outcome,variable_quanti)]
  ggplot(df, aes(x=!!sym(variable_quanti), fill=!!sym(outcome), color=!!sym(outcome))) +
    geom_histogram(position="identity",alpha=0.5)
}



prop_churn_by_cat=function(df,outcome,positive_modality,category){
  #hypothesis :outcome -- boolan vector
  df[outcome]= if_else(as.logical(df[outcome]==positive_modality),positive_modality,paste("Non",positive_modality))
  df=df[,c(outcome,category)]
  df[["value"]]=1

  return(
    ggplot(df, aes(x = !!sym(category), y = value,fill=!!sym(outcome))) +
      geom_bar(position="stack", stat="identity")+
    ggtitle(paste("Diagramme en barres représentant la proportion du churn par modalité: ",category)) +
      xlab(category) + ylab("Proportion")
  )

}

correlation_vars_quanti=function(df_quanti){
  return(cor(df_quanti))
}

seance1=function(file_path,outcome,positive_modality,sep=";"){
  df=read.csv(file_path,sep=sep)
  analysis_df= analysis_df(df)
  dimensions=analysis_df[["dimensions"]]
  missings=analysis_df[["missings"]]
  constants=analysis_df[["constants"]]
  bar_plot=diag_barres_qualitative(df,outcome)
  variables_type=get_categories(df)
  variables_quali=variables_type[variables_type %in% c("qualitative ordinale","qualitative nominale")]
  variables_quanti=variables_type[!variables_type %in% c("qualitative ordinale","qualitative nominale")]
  churn_by_cats=lapply(names(variables_quali),function(x){prop_churn_by_cat(df,outcome,positive_modality,x)})
  histograms=lapply(names(variables_quanti),function(x){histogram(df,outcome,positive_modality,x)})

  correlations=correlation_vars_quanti(df[names(variables_quanti)])

  list(dimensions=dimensions,missings=missings,constants=constants,bar_plot=bar_plot,churn_by_cats=churn_by_cats,histograms=histograms,correlations=correlations)
}

BANK_PATH="/home/khaldi/Documents/master_mlsd/app_machine/seance1/datasets/bank-additional-full.csv"
CREDIT_PATH="/home/khaldi/Documents/master_mlsd/app_machine/seance1/datasets/creditcard.csv"
FRAUD_PATH="/home/khaldi/Documents/master_mlsd/app_machine/seance1/datasets/whole data.csv"

bank=seance1(BANK_PATH,"y","yes")
credit=seance1(CREDIT_PATH,"Class",sep=",")
fraud=seance1(FRAUD_PATH,"Class",sep=",")