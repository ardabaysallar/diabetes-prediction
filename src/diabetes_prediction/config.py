"""Defaults shared by the whole package."""
import numpy as np

DATA_FILE = "diabetes.csv"
TARGET = "Outcome"

# These measurements can never be zero in a living person, so a zero means "missing".
MUST_NON_ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Clinical reference ranges (low, high) used to check the data and to build the categories.
REFERENCE_RANGES = {'Glucose': [1, 155],
                    'BloodPressure': [60, 90],
                    'SkinThickness': [25, 40],
                    'BMI': [18.5, 25],
                    'Insulin': [16, 166]}

# Outlier limits: quantiles and the 1.5 x range rule
OUTLIER_Q1 = 0.05
OUTLIER_Q3 = 0.95

# Column type detection: numeric columns with fewer unique values than CAT_TH count as categorical
CAT_TH = 18
CAR_TH = 30

# Clinical bins: the inner cut points. The data minimum and maximum are added as outer edges.
BINS = {
    'BMI': ([18.5, 25, 30], ['underweight', 'normal', 'overweight', 'obese']),
    'Insulin': ([110, 150, 240], ['Low', 'normal', 'high', 'extreme']),
    'SkinThickness': ([25, 40], ['Low', 'normal', 'high']),
    'BloodPressure': ([76.2, 98.1], ['Low', 'normal', 'high']),
    'DiabetesPedigreeFunction': ([0.245, 0.526, 0.806, 1.111], ['Low', 'Low-Normal', 'Normal', 'High-Normal', 'High']),
    'Glucose': ([89.1, 107.1, 123.1, 143.1, 165.1], ['Low', 'Low-Normal', 'Normal', 'High-Normal', 'High', 'Extreme']),
}
AGE_QUANTILES = [0.0, 0.25, 0.5, 0.75, 1]
AGE_LABELS = [1, 2, 3, 4]

# Insulin values above the 30%-70% limit are squeezed towards a ceiling of 240
INSULIN_CAP_Q1, INSULIN_CAP_Q3, INSULIN_CEILING = 0.3, 0.7, 240

ENCODE_COLS = ['Age_CAT', 'DiabetesPedigreeFunction_CAT', 'Glucose_CAT', 'SkinThickness_CAT',
               'Insulin_CAT', 'BMI_CAT', 'BloodPressure_CAT']
RARE_PERC = 0.014
RARE_COLS = ['Pregnancies']
SCALE_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Modelling
SEED = 42
TEST_SIZE = 0.30
CV_FOLDS = 10

LOGREG_GRID = [
    {'classifier__C': np.arange(0.1, 2.0, 0.05),
     'classifier__penalty': ['l1'],
     'classifier__solver': ['liblinear', 'saga']},
    {'classifier__C': np.arange(0.1, 2.0, 0.05),
     'classifier__penalty': ['l2'],
     'classifier__solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']},
]
RF_GRID = [{'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__n_estimators': [50, 100, 150, 200, 250, 300],
            'classifier__max_depth': [5, 7, 10, 13],
            'classifier__min_samples_split': [2, 3, 4],
            'classifier__max_features': [8, 10, 12],
            'classifier__min_samples_leaf': [1, 2, 3]}]

# Small grids for a quick run (--quick) and for tests
QUICK_LOGREG_GRID = [{'classifier__C': [0.5, 1.0], 'classifier__penalty': ['l2'], 'classifier__solver': ['lbfgs']}]
QUICK_RF_GRID = [{'classifier__n_estimators': [50], 'classifier__max_depth': [5, 7], 'classifier__max_features': [8]}]
