import numpy as np
import pandas as pd
from scipy.stats._stats_py import SignificanceResult
from sklearn.compose import make_column_selector as selector
from matplotlib.axes._axes import Axes
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import chi2_contingency


class Analyser:
    def __init__(self):
        pass

    def summary(self, df_c: pd.DataFrame) -> dict:
        qualitatives = selector(dtype_include=object)(df_c)

        return {"features":
            {
                "qualitative_columns": qualitatives,
                "quantitative_columns": [c for c in df_c if c not in qualitatives]
            },
            "describe": df_c.describe(),
        }

    def bar_chart(self, df_c: pd.DataFrame, col_name, ax: Axes, with_proportion=False) -> object:
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
        summary = self.summary(df_c)
        if not (cats_name in summary["features"]["qualitative_columns"]) or \
                not (outcome in summary["features"]["qualitative_columns"]):
            raise Exception(f"{outcome} and {cats_name} should be a qualitative columns from {df_c}")
        churn_by_cats = df_c.groupby([cats_name, outcome]).size().unstack()
        proportions = churn_by_cats.apply(lambda x: x / x.sum(), axis=1)
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
        df_corr = df_c.corr(numeric_only=True)
        return sns.heatmap(df_c[df_corr.index].corr(), annot=annot, cmap="RdYlGn", fmt=".2f")

    def quali_xquali_khi2(self, quali_y: pd.Series, quali_x: pd.Series) -> SignificanceResult:
        if not (isinstance(quali_x, pd.Series) and isinstance(quali_y, pd.Series)):
            raise TypeError(f"{quali_y} and {quali_x} should be a Series objects")
        res = pd.crosstab(quali_y, quali_x)
        Khi2_obs, p_value, ddl, theo_count = chi2_contingency(res)
        return p_value
