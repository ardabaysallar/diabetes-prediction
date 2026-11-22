import pytest

from diabetes_prediction import (add_features, encode_features, evaluate, scale_features, split_data,
                                 tune_logistic_regression, tune_random_forest)
from diabetes_prediction.config import QUICK_LOGREG_GRID, QUICK_RF_GRID


@pytest.fixture
def modelling(prepared):
    return scale_features(encode_features(add_features(prepared.copy())))


def test_split_is_stratified_and_70_30(modelling):
    X_train, X_test, y_train, y_test = split_data(modelling)
    assert len(X_train) + len(X_test) == len(modelling)
    assert len(X_test) == pytest.approx(0.30 * len(modelling), abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.02)
    assert "Outcome" not in X_train.columns


def test_logistic_regression_tuning_and_evaluation(modelling):
    X_train, X_test, y_train, y_test = split_data(modelling)
    grid = tune_logistic_regression(X_train, y_train, QUICK_LOGREG_GRID, n_jobs=1, verbose=0)
    result = evaluate(grid, X_train, y_train, X_test, y_test)
    assert set(result["best_params"]) == {"classifier__C", "classifier__penalty", "classifier__solver"}
    assert 0.5 < result["test_accuracy"] <= 1.0
    assert result["confusion_matrix"].sum() == len(y_test)
    assert 0.5 < result["roc_auc"] <= 1.0


def test_random_forest_tuning(modelling):
    X_train, X_test, y_train, y_test = split_data(modelling)
    grid = tune_random_forest(X_train, y_train, QUICK_RF_GRID, n_jobs=1, verbose=0)
    result = evaluate(grid, X_train, y_train, X_test, y_test)
    assert result["train_accuracy"] >= result["test_accuracy"] - 0.2
    assert result["best_params"]["classifier__n_estimators"] == 50
