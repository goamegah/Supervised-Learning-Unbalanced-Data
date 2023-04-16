import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector as selector
from matplotlib.axes._axes import Axes
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import chi2_contingency


class Analyser:
    def __init__(self):
        pass

    # Define a method to return a summary of the DataFrame including qualitative and quantitative columns
    def summary(self, df_c: pd.DataFrame) -> dict:
        # Select the qualitative columns using sklearn's make_column_selector method
        qualitatives = selector(dtype_include=object)(df_c)

        # Return a dictionary containing the qualitative and quantitative columns, as well as a description of the DataFrame
        return {"features":
            {
                "qualitative_columns": qualitatives,
                "quantitative_columns": [c for c in df_c if c not in qualitatives]
            },
            "describe": df_c.describe(),
        }

    def bar_chart(self, df_c: pd.DataFrame, col_name, ax: Axes, with_proportion=False) -> object:
        # Define a method to create a barchart for a specified qualitative column in the DataFrame
        summary = self.summary(df_c)
        if not col_name in summary["features"]["qualitative_columns"]:
            raise Exception(f"{col_name} should be a qualitative colums from {df_c}")
        cats_heights = df_c[col_name].value_counts()
        weight = 1 if not with_proportion else 1 / cats_heights.sum()
        cats_heights *= weight
        ax.bar(cats_heights.index, cats_heights.values)
        ax.set_title(f"Frequency Barchart of {col_name}")
        ax.set_xlabel("Modality")
        ax.set_ylabel("Frequency")
        # Return the frequencies/proportions as a pandas Series object
        return cats_heights

    def bar_chart_annotated(
            self,
            df_c: pd.DataFrame,
            col_name,
            ax: Axes,
            with_proportion=False
    ) -> object:
        summary = self.summary(df_c)
        if not col_name in summary["features"]["qualitative_columns"]:
            raise Exception(f"{col_name} should be a qualitative columns from {df_c}")
        cats_heights = df_c[col_name].value_counts()
        weight = 1 if not with_proportion else 1 / cats_heights.sum()
        cats_heights *= weight
        ax.bar(cats_heights.index, cats_heights.values)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: cats_heights.index[int(x)]))

        y_label = 'Frequency' if with_proportion else 'Counts'

        ax.set_title(f'{y_label} Barchart of {col_name}')
        ax.set_xlabel("Modality")
        ax.set_ylabel(f'{y_label}')

        # we use ax.set_xticks and pass in a range of integers from 0 to the length of counts.
        ax.set_xticks(range(len(cats_heights)))
        # we use ax.set_xticklabels and pass in the index values from counts.
        # This sets the tick labels to be the same as the original index values in counts.
        ax.set_xticklabels(cats_heights.index)

        # pour des labels plus compacts
        ax.ticklabel_format(axis='y', scilimits=(1, 4))

        # annotations
        for i, j in enumerate(cats_heights):
            ax.text(i, j+max(cats_heights.values)/80, cats_heights.iloc[i], ha='center')

        return cats_heights

    def prop_churn_by_cats(self, df_c: pd.DataFrame, outcome: str, cats_name: str, loc="upper right") -> None:
        # Define a method to create a stacked barchart of the proportions of churn vs. non churn for a specified categorical column in the DataFrame
        summary = self.summary(df_c)
        # Raise an exception if either the outcome or the categorical column is not qualitative
        if not (cats_name in summary["features"]["qualitative_columns"]) or \
                not (outcome in summary["features"]["qualitative_columns"]):
            raise Exception(f"{outcome} and {cats_name} should be a qualitative columns from {df_c}")
        # Calculate the counts of churn and non-churn for each modality of the categorical column
        churn_by_cats = df_c.groupby([cats_name, outcome]).size().unstack()
        # Calculate the proportions of churn and non-churn for each modality of the categorical column
        proportions = churn_by_cats.apply(lambda x: x / x.sum(), axis=1)
        # Create a stacked barchart of the proportions
        ax = proportions.plot(kind='bar', stacked=True)
        ax.set_title(f"Proportions of churn vs. non churn by {cats_name}")
        ax.set_xlabel(cats_name)
        ax.set_ylabel('Proportion')
        ax.legend(loc=loc)

    def prop_churn_by_numerical_hist(
            self,
            df_c: pd.DataFrame,
            outcome: str,
            positive_mod: str,
            numericals_name: str,
            ax: Axes,
            n_bins=5,
            alpha=.75,
            density=True,
            loc="upper right"
    ) -> None:
        """
        Parameters:
        df_c (pd.DataFrame): The DataFrame containing the data to be plotted.
        outcome (str): The name of the qualitative feature.
        positive_mod (str): The value of the qualitative feature that is considered positive.
        numericals_name (str): The name of the quantitative feature.
        ax (Axes): The Axes object on which to plot the histogram.
        n_bins (int): The number of bins to use for the histogram. Default is 5.
        alpha (float): The transparency of the bars in the histogram. Default is 0.75.
        density (bool): Whether to normalize the histogram so that the area under the bars sums to 1. Default is True.
        loc (str): The location of the legend. Default is "upper right".
        The histogram is split into two parts: one for rows where the outcome is the positive_mod,
        and one for rows where it is not.
        """
        summary = self.summary(df_c)
        if not numericals_name in summary["features"]["quantitative_columns"] or \
                not outcome in summary["features"]["qualitative_columns"]:
            raise Exception(f"{outcome} should be a qualitative columns  and \
            {numericals_name} should be a quantitative column from {df_c}")
        mask_churn = df_c[outcome] == positive_mod
        churn = df_c.loc[mask_churn].copy()
        non_churn = df_c.loc[~mask_churn].copy()
        counts, bins, _ = ax.hist(
            [churn[numericals_name], non_churn[numericals_name]],
            bins=n_bins, alpha=alpha,
            label=[positive_mod, f"Not {positive_mod}"],
            color=["blue", "black"],
            density=density,
            stacked=True
        )
        ax.set_title(f"histogram of {numericals_name} conditioned by {outcome}")
        ax.set_ylabel("Density" if density == True else "Count")
        ax.legend(loc=loc)

    def prop_churn_by_numerical_boxplot(
            self,
            df_c: pd.DataFrame,
            outcome: str,
            positive_mod: str,
            numericals_name: str,
            ax: Axes
    ):
        """
           Plots a boxplot of a quantitative feature conditioned by a qualitative feature, where
           the qualitative feature is the "outcome" and the quantitative feature is "numericals_name".
           The boxplot is split into two parts: one for rows where the outcome is the positive_mod,
           and one for rows where it is not.

           Parameters:
           df_c (pd.DataFrame): The DataFrame containing the data to be plotted.
           outcome (str): The name of the qualitative feature.
           positive_mod (str): The value of the qualitative feature that is considered positive.
           numericals_name (str): The name of the quantitative feature.
           ax (Axes): The Axes object on which to plot the boxplot.

        """
        summary = self.summary(df_c)
        if not (numericals_name in summary["features"]["quantitative_columns"]) or \
                not (outcome in summary["features"]["qualitative_columns"]):
            raise Exception(
                f"{outcome} should be a qualitative columns  and \
                {numericals_name} should be a quantitative column from {df_c}"
            )
        mask_churn = df_c[outcome] == positive_mod
        churn = df_c.loc[mask_churn][numericals_name].copy()
        non_churn = df_c.loc[~mask_churn][numericals_name].copy()
        ax.boxplot([np.array(churn).reshape(-1), np.array(non_churn).reshape(-1)], notch=True)
        ax.set_xticklabels([positive_mod, f"Not{positive_mod}"])
        ax.set_title(f"Boxplot of {numericals_name} conditioned by {outcome}")

    def correlation_heatmap(self, df_c: pd.DataFrame, annot=False):
        # Compute the correlation matrix for the numerical columns in the given DataFrame
        df_corr = df_c.corr(numeric_only=True)
        # Plot a heatmap of the correlation matrix, with optional annotations
        return sns.heatmap(df_c[df_corr.index].corr(), annot=annot, cmap="RdYlGn", fmt=".2f")

    def quali_xquali_khi2(self, quali_y: pd.Series, quali_x: pd.Series):
        """
        The method takes two Pandas Series objects quali_y and quali_x as inputs.
        returns the p-value for the chi-square test of independence between the two categorical variables.
        """
        # Check if the inputs are Series objects
        if not (isinstance(quali_x, pd.Series) and isinstance(quali_y, pd.Series)):
            raise TypeError(f"{quali_y} and {quali_x} should be a Series objects")
        # Compute the contingency table of the two categorical variables
        res = pd.crosstab(quali_y, quali_x)
        # Compute the chi-square statistic and p-value for the contingency table
        Khi2_obs, p_value, ddl, theo_count = chi2_contingency(res)
        # Return the p-value
        return p_value
