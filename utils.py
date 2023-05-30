import pandas as pd
import numpy as np
import hdbscan
import time
from denseclus import DenseClus


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


def fit_DenseClus(df, params):
    """Fits a DenseClus and returns all relevant information
        df = data
        params = dict with the prameters for the DenseClus

        returns -------------
        embedding =  transformed data points
        clustered = boolean vector decides if  not noise
        result = data frame with the embedding a and LABELS
        DBCV = score
        coverage = notNoise/total-points



    """
    np.random.seed(params['SEED'])
    clf = DenseClus(
        random_state=params['SEED'],
        cluster_selection_method=params['cluster_selection_method'],
        min_samples=params['min_samples'],
        n_components=params['n_components'],
        min_cluster_size=params['min_cluster_size'],
        umap_combine_method=params['umap_combine_method'],
        prediction_data=True

    )

    start = time.time()
    clf.fit(df)
    print('time fitting ', (time.time() - start) / 60)
    print(clf.n_components)
    embedding = clf.mapper_.embedding_
    labels = clf.score()

    result = pd.DataFrame(clf.mapper_.embedding_)
    result['LABELS'] = pd.Series(clf.score())
    print('clusters ', len(set(result['LABELS'])) - 1)

    lab_count = result['LABELS'].value_counts()
    lab_count.name = 'LABEL_COUNT'

    lab_normalized = result['LABELS'].value_counts(normalize=True)
    lab_normalized.name = 'LABEL_PROPORTION'
    print('ruido ', lab_normalized[-1])

    labels = pd.DataFrame(clf.score())
    labels.columns = ['LABELS']
    print(cluster_counts(labels, 'LABELS'))
    clustered, coverage = cluster_coverage(labels, 'LABELS')
    print(f"Coverage :{coverage}")
    DBCV = clf.hdbscan_.relative_validity_
    return embedding, clustered, result, DBCV, coverage, clf


