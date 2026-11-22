"""Cleaning: impossible zeros, outliers and missing values."""
import numpy as np

from .config import MUST_NON_ZERO_COLS, OUTLIER_Q1, OUTLIER_Q3, REFERENCE_RANGES


def mark_zeros_as_missing(df, cols=MUST_NON_ZERO_COLS):
    """Replace the impossible zeros with NaN (in place) and return the frame."""
    for col in cols:
        df.loc[df[col] == 0.0, col] = np.nan
    return df


def outlier_thresholds(dataframe, col_name, q1=OUTLIER_Q1, q3=OUTLIER_Q3):
    """Lower and upper outlier limit of a column (quantiles q1 and q3, 1.5 x range)."""
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """True if the column has values outside its outlier limits."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) |
                 (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def remove_outlier(dataframe, col_name):
    """The frame without the rows whose value in `col_name` is outside the outlier limits."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) |
                                      (dataframe[col_name] > up_limit))]
    return df_without_outliers


def drop_outliers(df, num_cols):
    """Apply the outlier filter column by column, each time to the full frame.

    As in the original script, the result is the frame filtered by the last numeric column.
    """
    new_df = df
    for col in num_cols:
        new_df = remove_outlier(df, col)
    return new_df


def check_anomaly(dataframe, reference=REFERENCE_RANGES):
    """Print, per measurement, whether all values lie inside its clinical reference range."""
    for col in reference.keys():
        if ((dataframe[col].min() >= reference[col][0]) &
                (dataframe[col].max() <= reference[col][1])):
            print(col, ' : all in normal range')
        else:
            print(col, ' : ALERT')


def impute_missing(df):
    """Fill the gaps with the nearest value, then drop what is still missing (in place)."""
    df.interpolate(method='nearest', inplace=True)
    df.dropna(inplace=True)
    return df
