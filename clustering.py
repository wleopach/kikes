import pandas as pd
import numpy as np
import time
from denseclus import DenseClus
from plot_utils import plot_join

dfc = pd.read_csv('./Data/ventasxsemana_11001_cluster_240523_corto.csv')
dfc = dfc[['total_item', ' B _tot', 'L_tot', 'L_PET_tot', 'M_tot', 'M_PET_tot',
           'XL_tot', 'XL_PET_tot', 'YUMBO_tot', 'Momentum', 'freqmes', 'dias_compra', 'recency_class']]

dfc = dfc.head(50000)

dfc.dropna(inplace=True)
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
        umap_combine_method=params['umap_combine_method']

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

    clustered = result['LABELS'] >= 0
    cnts = pd.DataFrame(clf.score())[0].value_counts()
    cnts = cnts.reset_index()
    cnts.columns = ['CLUSTER', 'COUNT']
    print(cnts.sort_values(['CLUSTER']))
    coverage = np.sum(clustered) / clf.mapper_.embedding_.shape[0]
    print(f"Coverage {coverage}")
    DBCV = clf.hdbscan_.relative_validity_
    return embedding, clustered, result, DBCV, coverage, clf


params = dict()
params['cluster_selection_method'] = "eom"
params['min_samples'] = 7
params['n_components'] = 2
params['min_cluster_size'] = 500
params['umap_combine_method'] = "intersection_union_mapper"
params['SEED'] = None
DBCV = -1
while DBCV < 0.45:
    embedding, clustered, result, DBCV, coverage, clf = fit_DenseClus(dfc, params)
    print(DBCV)
plot_join(embedding[clustered, 0], embedding[clustered, 1], result['LABELS'][clustered])
