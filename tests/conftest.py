"""Synthetic Pima-like data for the tests: no real dataset is needed."""
import numpy as np
import pandas as pd
import pytest


def make_diabetes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    glucose = rng.normal(120, 30, n).clip(40, 199).round()
    bmi = rng.normal(32, 6.5, n).clip(18, 60).round(1)
    age = rng.integers(21, 70, n)
    outcome = (rng.random(n) < 1 / (1 + np.exp(-(glucose - 130) / 20 - (bmi - 32) / 15))).astype(int)
    df = pd.DataFrame({
        "Pregnancies": rng.integers(0, 14, n),
        "Glucose": glucose,
        "BloodPressure": rng.normal(70, 12, n).clip(30, 120).round(),
        "SkinThickness": rng.normal(26, 10, n).clip(7, 80).round(),
        "Insulin": rng.gamma(2, 70, n).clip(14, 600).round(),
        "BMI": bmi,
        "DiabetesPedigreeFunction": rng.gamma(2, 0.25, n).clip(0.08, 2.4).round(3),
        "Age": age,
        "Outcome": outcome,
    })
    for col, share in (("SkinThickness", .25), ("Insulin", .45), ("BloodPressure", .05), ("Glucose", .01), ("BMI", .01)):
        df.loc[rng.random(n) < share, col] = 0   # the impossible zeros of the real data
    return df


@pytest.fixture
def raw():
    return make_diabetes()


@pytest.fixture
def csv_file(tmp_path, raw):
    path = tmp_path / "diabetes.csv"
    raw.to_csv(path, index=False)
    return path


@pytest.fixture
def prepared(raw):
    """Frame after zeros, outliers and imputation (before feature engineering)."""
    from diabetes_prediction import drop_outliers, grab_col_names, impute_missing, mark_zeros_as_missing
    df = raw.copy()
    mark_zeros_as_missing(df)
    _, num_cols, _ = grab_col_names(df, verbose=False)
    new_df = drop_outliers(df, num_cols)
    return impute_missing(new_df)
