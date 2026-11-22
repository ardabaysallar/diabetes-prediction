"""Splitting, tuning and evaluating the two classifiers."""
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, classification_report, confusion_matrix, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, LOGREG_GRID, RF_GRID, SEED, TARGET, TEST_SIZE


def split_data(df, target=TARGET, test_size=TEST_SIZE, seed=SEED):
    """Stratified train and test split. Returns X_train, X_test, y_train, y_test."""
    train, test = model_selection.train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target])
    return train.drop(target, axis=1), test.drop(target, axis=1), train[target], test[target]


def _tune(classifier, X_train, y_train, param_grid, cv_folds, n_jobs, verbose):
    pipeline = Pipeline(steps=[('classifier', classifier)])
    kfold = StratifiedKFold(n_splits=cv_folds, random_state=None)  # same class ratio in every fold
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=kfold, verbose=verbose, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    return grid


def tune_logistic_regression(X_train, y_train, param_grid=LOGREG_GRID, seed=SEED, cv_folds=CV_FOLDS,
                             n_jobs=-1, verbose=1):
    """Grid search over a logistic regression. Returns the fitted GridSearchCV."""
    return _tune(LogisticRegression(random_state=seed), X_train, y_train, param_grid, cv_folds, n_jobs, verbose)


def tune_random_forest(X_train, y_train, param_grid=RF_GRID, seed=SEED, cv_folds=CV_FOLDS,
                       n_jobs=-1, verbose=1):
    """Grid search over a random forest. Returns the fitted GridSearchCV."""
    return _tune(RF(random_state=seed), X_train, y_train, param_grid, cv_folds, n_jobs, verbose)


def evaluate(grid, X_train, y_train, X_test, y_test):
    """Best parameters, accuracies, confusion matrix and the classification report of a fitted grid."""
    predicted = grid.predict(X_test)
    result = {
        'best_params': grid.best_params_,
        'train_accuracy': grid.score(X_train, y_train),
        'test_accuracy': grid.score(X_test, y_test),
        'confusion_matrix': confusion_matrix(y_test, predicted),
        'report': classification_report(y_test, predicted),
    }
    if hasattr(grid, 'predict_proba'):
        fpr, tpr, _ = roc_curve(y_test, grid.predict_proba(X_test)[:, 1])
        result['roc_auc'] = auc(fpr, tpr)
    return result
