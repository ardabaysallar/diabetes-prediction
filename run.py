"""Command line entry: python run.py --data data/diabetes.csv

The full grid search is slow; use --quick for small grids or --skip-models to stop after the data work.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from diabetes_prediction import (add_features, check_anomaly, check_outlier, drop_outliers, encode_features,  # noqa: E402
                                 evaluate, grab_col_names, high_corr_column_grabber, impute_missing,
                                 mark_zeros_as_missing, missing_values_table, rare_analyser, reference_report,
                                 scale_features, split_data, target_variable_analysis, tune_logistic_regression,
                                 tune_random_forest)
from diabetes_prediction.config import (DATA_FILE, LOGREG_GRID, QUICK_LOGREG_GRID, QUICK_RF_GRID,  # noqa: E402
                                        RARE_COLS, RF_GRID, TARGET)


def show(name, grid, result):
    print(grid.best_estimator_, '\n')
    print('Best parameters  :', result['best_params'])
    print('\nTraining accuracy:', result['train_accuracy'])
    print('Test accuracy    :', result['test_accuracy'])
    print(f'\n{name} classification report:\n', result['report'])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diabetes prediction: feature engineering and classification.")
    parser.add_argument("--data", default=DATA_FILE, help="csv file (default: diabetes.csv)")
    parser.add_argument("--quick", action="store_true", help="use small parameter grids")
    parser.add_argument("--skip-models", action="store_true", help="stop after the data preparation")
    parser.add_argument("--plots", metavar="DIR", help="save charts into this folder")
    args = parser.parse_args(argv)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.width', 500)

    df = pd.read_csv(args.data)
    mark_zeros_as_missing(df)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    target_variable_analysis(df, cat_cols, num_cols, TARGET)

    for col in num_cols:
        print(col, check_outlier(df, col))
    new_df = drop_outliers(df, num_cols)
    check_anomaly(dataframe=df)

    na_cols = missing_values_table(new_df, True)
    print(high_corr_column_grabber(new_df.corr(), new_df))

    plots = None
    if args.plots:
        from diabetes_prediction import plots
        out = Path(args.plots)
        out.mkdir(parents=True, exist_ok=True)
        bar, matrix = plots.missing_overview(new_df)
        bar.savefig(out / "missing_bar.png")
        matrix.savefig(out / "missing_matrix.png")
        plots.correlation_heatmap(new_df).savefig(out / "correlation.png")

    impute_missing(new_df)
    reference_report(new_df)

    add_features(new_df)
    rare_analyser(new_df, TARGET, RARE_COLS)
    new_df = encode_features(new_df)
    scale_features(new_df)

    if args.skip_models:
        print(new_df.describe().T)
        return

    X_train, X_test, y_train, y_test = split_data(new_df)
    print('X_train and y_train:', X_train.shape, y_train.shape)
    print('X_test and y_test  :', X_test.shape, y_test.shape)

    lr = tune_logistic_regression(X_train, y_train, QUICK_LOGREG_GRID if args.quick else LOGREG_GRID)
    show('Logistic regression', lr, evaluate(lr, X_train, y_train, X_test, y_test))
    rf = tune_random_forest(X_train, y_train, QUICK_RF_GRID if args.quick else RF_GRID)
    show('Random forest', rf, evaluate(rf, X_train, y_train, X_test, y_test))

    if plots:
        plots.confusion_matrix_plot(y_test, lr.predict(X_test)).savefig(out / "confusion_matrix.png")
        plots.roc_plot(lr, X_test, y_test).savefig(out / "roc_curve.png")
        print("charts saved in", out)


if __name__ == "__main__":
    main()
