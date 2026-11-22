"""Looking at the data: column types, target analysis, missing values, correlations."""
import numpy as np
import pandas as pd

from .config import CAR_TH, CAT_TH, REFERENCE_RANGES


def grab_col_names(dataframe, cat_th=CAT_TH, car_th=CAR_TH, verbose=True):
    """Names of the categorical, numeric and categorical-but-cardinal columns.

    Numeric columns with fewer than `cat_th` unique values count as categorical.
    Returns (cat_cols, num_cols, cat_but_car).
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    if verbose:
        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def target_variable_analysis(df, cat_cols, num_cols, target='Outcome'):
    """Print the target mean per categorical column and the numeric means per target value."""
    print('CATEGORICAL COLUMNS', '\n',
          df.groupby(cat_cols).agg({target: 'mean'}), '\n\n')
    print('NUMERICAL COLUMNS', '\n',
          df.pivot_table(values=num_cols, columns=target, aggfunc='mean'), '\n\n')


def missing_values_table(dataframe, na_name=False):
    """Print the count and share of missing values per column; optionally return those columns."""
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def high_corr_column_grabber(corr_matrx, dataframe):
    """For each column, the pair of columns it is strongly (> 0.6) correlated with."""
    high_corrs = {}
    for col in corr_matrx.columns:
        high_corr = np.where(corr_matrx[col] > 0.6)
        if high_corr[0].size > 1:
            col1 = dataframe.columns[high_corr[0][0]]
            col2 = dataframe.columns[high_corr[0][1]]
            if col1 not in high_corrs.items():
                high_corrs[col1] = [col2]
            else:
                high_corrs[col1].append(col2)

    return high_corrs


def reference_report(df, reference=REFERENCE_RANGES):
    """Print the data mean next to the reference range of each measurement."""
    for k, v in reference.items():
        print('Dataframe: ', df[k].mean(), '---- Reference:', v)
        print(k, '-->', -df[k].mean() + ((v[1] - v[0]) + v[0]), end='\n\n')


def rare_analyser(dataframe, target, cat_cols):
    """Print count, share and target mean for every value of the given columns."""
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
