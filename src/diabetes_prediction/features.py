"""Feature engineering: clinical categories, encoding and scaling."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

from .config import (AGE_LABELS, AGE_QUANTILES, BINS, ENCODE_COLS, INSULIN_CAP_Q1, INSULIN_CAP_Q3,
                     INSULIN_CEILING, RARE_COLS, RARE_PERC, SCALE_COLS)
from .preprocessing import outlier_thresholds


def add_category(df, col, new_col=None):
    """Add `<col>_CAT`: the column cut into clinical categories (see BINS in config)."""
    inner, labels = BINS[col]
    edges = [df[col].min()] + inner + [df[col].max()]
    df[new_col or f'{col}_CAT'] = pd.cut(df[col], edges, labels=labels, include_lowest=True)
    return df


def cap_insulin(df):
    """Squeeze the highest insulin values towards the ceiling (in place)."""
    low_limit, up_limit = outlier_thresholds(df, 'Insulin', q1=INSULIN_CAP_Q1, q3=INSULIN_CAP_Q3)
    df.loc[df['Insulin'] > up_limit, 'Insulin'] = (up_limit +
                                                   INSULIN_CEILING * (df.loc[df['Insulin'] > up_limit, 'Insulin'] /
                                                                      df.Insulin.max()))
    return df


def add_features(df):
    """Add all clinical category columns. Insulin is categorised first and then capped."""
    add_category(df, 'BMI')
    add_category(df, 'Insulin')
    cap_insulin(df)
    add_category(df, 'SkinThickness')
    add_category(df, 'BloodPressure')
    add_category(df, 'DiabetesPedigreeFunction')
    add_category(df, 'Glucose')
    df['Age_CAT'] = pd.qcut(df.Age, q=AGE_QUANTILES, labels=AGE_LABELS)
    return df


def label_encoder(dataframe, col):
    """Replace a column by integer codes (in place) and return the frame."""
    labelencoder = LabelEncoder()
    dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe


def rare_encoder(dataframe, rare_perc, rare_cols):
    """Return a copy where values rarer than `rare_perc` are replaced by 'Rare'."""
    temp_df = dataframe.copy()

    rare_columns = rare_cols
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def encode_features(df):
    """Encode the category columns and the rare pregnancy counts."""
    for col in ENCODE_COLS:
        label_encoder(df, col)
    df = rare_encoder(df, RARE_PERC, RARE_COLS)
    for col in RARE_COLS:
        label_encoder(df, col)
    return df


def scale_features(df, cols=SCALE_COLS):
    """Scale the measurements with a robust scaler (in place)."""
    df[cols] = RobustScaler().fit_transform(df[cols])
    return df


def engineer(df):
    """From the cleaned frame to the modelling frame: categories, encoding, scaling."""
    add_features(df)
    df = encode_features(df)
    return scale_features(df)
