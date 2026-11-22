"""Optional charts. Each function returns the figure so the caller decides to show or save it."""
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def missing_overview(df):
    """Missing-value bar chart and matrix. Returns two figures."""
    ax1 = msno.bar(df, inline=False)
    ax2 = msno.matrix(df, inline=False)
    return ax1.figure, ax2.figure


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, ax=ax)
    return fig


def missing_histograms(dataframe, na_cols, title='HISTOGRAMS'):
    fig, axs = plt.subplots(len(na_cols))
    fig.suptitle(title)
    for count, col in enumerate(na_cols):
        axs[count].hist(dataframe[col])
    return fig


def confusion_matrix_plot(actual, predicted):
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(actual, predicted), annot=True, fmt='.0f', ax=ax,
                xticklabels=["Outcome:1", "Outcome:0"], yticklabels=["Outcome:1", "Outcome:0"])
    ax.set_ylabel('ACTUAL')
    ax.set_xlabel('PREDICTED')
    return fig


def roc_plot(grid, X_test, y_test):
    fpr, tpr, _ = roc_curve(y_test, grid.predict_proba(X_test)[:, 1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label='LR (ROC-AUC = %0.2f)' % auc(fpr, tpr))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (TPR vs FPR at each probability threshold)')
    ax.legend(loc="lower right")
    return fig
