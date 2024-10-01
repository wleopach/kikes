import pandas as pd
import numpy as np
import hdbscan
from joblib import load
from warnings import filterwarnings
from sklearn.preprocessing import PowerTransformer


def dbcv_score(result, index, metric, mask):
    df_work = result.loc[index].copy()
    # Convert the 'data' column of the dataframe to a numpy array

    labels = df_work.pop('LABELS')
    X = np.array(df_work)
    X = X.astype(np.double)
    sizes = np.bincount(labels + 1)
    noise_size = sizes[0]
    cluster_size = sizes[1:]
    total = noise_size + np.sum(cluster_size)
    num_clusters = len(cluster_size)
    DSC = np.zeros(num_clusters)
    min_outlier_sep = np.inf  # only required if num_clusters = 1
    correction_const = 2
    DSPC_wrt = np.ones(num_clusters) * np.inf
    max_distance = 0
    for edge in mask.iterrows():
        label1 = labels[int(edge[1]["from"])]
        label2 = labels[int(edge[1]["to"])]
        length = edge[1]["distance"]
        max_distance = max(max_distance, length)
        if label1 == -1 and label2 == -1:
            continue
        elif label1 == -1 or label2 == -1:
            # If exactly one of the points is noise
            min_outlier_sep = min(min_outlier_sep, length)
            continue
        if label1 == label2:
            # Set the density sparseness of the cluster
            # to the sparsest value seen so far.
            DSC[label1] = max(length, DSC[label1])
        else:
            # Check whether density separations with
            # respect to each of these clusters can
            # be reduced.
            DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
            DSPC_wrt[label2] = min(length, DSPC_wrt[label2])
    # In case min_outlier_sep is still np.inf, we assign a new value to it.
    # This only makes sense if num_clusters = 1 since it has turned out
    # that the MR-MST has no edges between a noise point and a core point.
    min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep
    # DSPC_wrt[Ci] might be infinite if the connected component for Ci is
    # an "island" in the MR-MST. Whereas for other clusters Cj and Ck, the
    # MR-MST might contain an edge with one point in Cj and ther other one
    # in Ck. Here, we replace the infinite density separation of Ci by
    # another large enough value.
    #
    # TODO: Think of a better yet efficient way to handle this.
    correction = correction_const * (
        max_distance if num_clusters > 1 else min_outlier_sep
    )
    DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction
    V_index = [
        (DSPC_wrt[i] - DSC[i]) / max(DSPC_wrt[i], DSC[i])
        for i in range(num_clusters)
    ]
    relative_validity = np.sum(
        [(cluster_size[i] * V_index[i]) / total for i in range(num_clusters)]
    )
    # Calculate the DBCV score
    validity_index = hdbscan.validity.validity_index(X, labels, metric=metric)
    df_work['LABELS'] = labels
    counts = cluster_counts(df_work, 'LABELS')

    return relative_validity, validity_index, counts


def cluster_counts(df, column):
    counts = df[column].value_counts()
    counts = counts.reset_index()
    counts.columns = ['LABELS', 'COUNT']
    counts = counts.sort_values(['LABELS'])
    return counts


def cluster_coverage(df, column):
    clustered = df[column] >= 0
    coverage = np.sum(clustered) / len(df)
    return clustered, coverage


def load_clf(path):
    """
    Given a pretrained DenseClus this method loads the model from path
    :param path: place where the model is stored
    :return: embedding with labels and clf
    """
    clf = load(path)
    labels = clf.score()
    embedding = pd.DataFrame(clf.mapper_.embedding_)
    embedding['LABELS'] = labels

    return embedding, clf


def check_is_df(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Requires DataFrame as input")


def extract_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts categorical features into binary dummy dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features

    Returns:
        pd.DataFrame: binary dummy DataFrame of categorical features
    """
    check_is_df(df)

    categorical = df.select_dtypes(exclude=["float", "int"])
    if categorical.shape[1] == 0:
        raise ValueError("No Categories found, check that objects are in dataframe")

    categorical_dummies = pd.get_dummies(categorical)

    return categorical_dummies


def extract_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts numerical features into normailzed numeric only dataframe

    Parameters:
        df (pd.DataFrame): DataFrame with numerical and categorical features

    Returns:
        pd.DataFrame: normalized numerical DataFrame of numerical features
    """
    check_is_df(df)

    numerical = df.select_dtypes(include=["float", "int"])
    if numerical.shape[1] == 0:
        raise ValueError("No numerics found, check that numerics are in dataframe")

    return transform_numerics(numerical)


def transform_numerics(numerical: pd.DataFrame) -> pd.DataFrame:
    """Power transforms numerical DataFrame

    Parameters:
        numerical (pd.DataFrame): Numerical features DataFrame

    Returns:
        pd.DataFrame: Normalized DataFrame of Numerical features
    """

    check_is_df(numerical)

    for names in numerical.columns.tolist():
        pt = PowerTransformer(copy=False)
        # TO DO: fix this warning message
        filterwarnings("ignore")
        numerical.loc[:, names] = pt.fit_transform(
            np.array(numerical.loc[:, names]).reshape(-1, 1),
        )
        filterwarnings("default")

    return numerical


def normalize_array(arr):
    # Calculate the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Normalize the array using the formula: (x - min) / (max - min)
    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr, min_val, max_val


def normalize_new(arr, min_val, max_val):
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr
