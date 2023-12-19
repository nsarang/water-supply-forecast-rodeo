import pandas as pd


def iqr_outliers(group: pd.Series, lq=0.25, uq=0.75, scale=1.5):
    Q1 = group.quantile(lq)
    Q3 = group.quantile(uq)
    IQR = Q3 - Q1
    return group.between((Q1 - scale * IQR), (Q3 + scale * IQR))
