import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def outlier_treatment(data_column):
    sorted(data_column)
    Q1, Q3 = np.percentile(data_column, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


def bounds_dict(df, sel_columns):
    quant = {}
    for col in sel_columns:
        quant[col] = {}
        quant[col]['L'], quant[col]['U'] = outlier_treatment(df[col])
    return quant


def clean_df(da, sel_columns):
    b = bounds_dict(da, sel_columns)
    for col in sel_columns:
        l = b[col]['L']
        u = b[col]['U']
        da.drop(da[(da[col] > u) | (da[col] < l)].index, inplace=True)
    return da
