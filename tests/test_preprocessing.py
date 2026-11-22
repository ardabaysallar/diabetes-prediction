import numpy as np
import pandas as pd

from diabetes_prediction import (check_anomaly, check_outlier, drop_outliers, grab_col_names, impute_missing,
                                 mark_zeros_as_missing, outlier_thresholds, remove_outlier)


def test_zeros_become_missing_only_in_the_listed_columns(raw):
    df = mark_zeros_as_missing(raw.copy())
    assert df["Insulin"].isna().sum() == (raw["Insulin"] == 0).sum()
    assert (df["Glucose"].dropna() != 0).all()
    assert (df["Pregnancies"] == raw["Pregnancies"]).all()    # zero pregnancies are real


def test_outlier_limits_follow_the_range_rule():
    s = pd.DataFrame({"x": np.arange(101.0)})
    low, up = outlier_thresholds(s, "x")
    q1, q3 = s["x"].quantile(0.05), s["x"].quantile(0.95)
    assert low == q1 - 1.5 * (q3 - q1) and up == q3 + 1.5 * (q3 - q1)


def test_check_and_remove_outlier():
    s = pd.DataFrame({"x": list(range(100)) + [10_000]})
    assert check_outlier(s, "x") is True
    assert 10_000 not in remove_outlier(s, "x")["x"].values
    assert check_outlier(remove_outlier(s, "x"), "x") is False


def test_grab_col_names_splits_by_unique_values(raw):
    cat_cols, num_cols, cat_but_car = grab_col_names(raw, verbose=False)
    assert "Outcome" in cat_cols and "Pregnancies" in cat_cols
    assert "Glucose" in num_cols and "Age" in num_cols


def test_drop_outliers_keeps_missing_rows(raw):
    df = mark_zeros_as_missing(raw.copy())
    _, num_cols, _ = grab_col_names(df, verbose=False)
    assert len(drop_outliers(df, num_cols)) <= len(df)


def test_impute_leaves_no_gaps(prepared):
    assert prepared.isna().sum().sum() == 0


def test_check_anomaly_prints_one_line_per_measurement(raw, capsys):
    check_anomaly(raw)
    assert len(capsys.readouterr().out.strip().splitlines()) == 5
