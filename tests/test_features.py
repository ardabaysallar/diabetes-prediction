import numpy as np
import pandas as pd
import pytest

from diabetes_prediction import (add_features, encode_features, label_encoder, rare_encoder, scale_features)
from diabetes_prediction.features import add_category, cap_insulin


def test_bmi_categories_follow_the_clinical_classes():
    df = pd.DataFrame({"BMI": [16.0, 22.0, 27.0, 35.0, 40.0]})
    add_category(df, "BMI")
    assert df["BMI_CAT"].tolist() == ["underweight", "normal", "overweight", "obese", "obese"]


def test_glucose_has_six_categories(prepared):
    add_category(prepared, "Glucose")
    assert set(prepared["Glucose_CAT"].dropna().unique()) <= {"Low", "Low-Normal", "Normal", "High-Normal", "High", "Extreme"}


def test_insulin_is_categorised_before_it_is_capped(prepared):
    df = prepared.copy()
    before_max = df["Insulin"].max()
    add_features(df)
    assert df["Insulin"].max() <= before_max
    assert "Insulin_CAT" in df.columns


def test_cap_insulin_only_moves_values_above_the_limit():
    df = pd.DataFrame({"Insulin": np.arange(1.0, 101.0)})
    low = df["Insulin"].copy()
    cap_insulin(df)
    limit = low.quantile(0.7) + 1.5 * (low.quantile(0.7) - low.quantile(0.3))
    assert (df.loc[low <= limit, "Insulin"] == low[low <= limit]).all()


def test_add_features_adds_all_seven_category_columns(prepared):
    df = add_features(prepared.copy())
    for col in ("Age_CAT", "DiabetesPedigreeFunction_CAT", "Glucose_CAT", "SkinThickness_CAT",
                "Insulin_CAT", "BMI_CAT", "BloodPressure_CAT"):
        assert col in df.columns


def test_age_groups_are_quartiles(prepared):
    df = add_features(prepared.copy())
    shares = df["Age_CAT"].value_counts(normalize=True)
    assert shares.tolist() == pytest.approx([0.25] * 4, abs=0.06)


def test_label_encoder_gives_integer_codes():
    df = pd.DataFrame({"c": ["b", "a", "c", "a"]})
    assert label_encoder(df, "c")["c"].tolist() == [1, 0, 2, 0]


def test_rare_encoder_replaces_rare_values():
    df = pd.DataFrame({"p": [1] * 98 + [2, 3]})
    out = rare_encoder(df, 0.03, ["p"])
    assert out["p"].tolist().count("Rare") == 2
    assert df["p"].tolist().count(1) == 98      # original is untouched


def test_encode_then_scale_gives_numeric_frame(prepared):
    df = add_features(prepared.copy())
    df = scale_features(encode_features(df))
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    assert abs(df["Glucose"].median()) < 1e-9     # robust scaler centres on the median
