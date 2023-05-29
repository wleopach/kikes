import pandas as pd
import numpy as np
import hdbscan
from sklearn.metrics import pairwise_distances


def dbcv_score(df, metric):
    df_work = df.copy()
    # Convert the 'data' column of the dataframe to a numpy array

    labels = df_work.pop('LABELS')
    X = np.array(df_work)

    # Calculate the DBCV score
    result = hdbscan.validity.validity_index(X, labels, metric=metric)

    return result
