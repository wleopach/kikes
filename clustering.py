import pandas as pd
from plot_utils import plot_join
from utils import fit_DenseClus, dbcv_score
from config import PATH
from joblib import dump
import datetime
from hdbscan import prediction
dfc0 = pd.read_csv(f'{PATH}ventasxsemana_11001_cluster_240523_corto.csv')
dfc = dfc0[['total_item', ' B _tot', 'L_tot', 'L_PET_tot', 'M_tot', 'M_PET_tot',
            'XL_tot', 'XL_PET_tot', 'YUMBO_tot', 'Momentum', 'freqmes', 'dias_compra', 'recency_class']].copy()

dfc = dfc.head(5000)

dfc.dropna(inplace=True)

params = dict()
params['cluster_selection_method'] = "eom"
params['min_samples'] = 8
params['n_components'] = 3
params['min_cluster_size'] = int(len(dfc) * 2 / 100)
params['umap_combine_method'] = "intersection_union_mapper"
params['SEED'] = None
DBCV = -1
while DBCV < 0.45:
    embedding, clustered, result, DBCV, coverage, clf = fit_DenseClus(dfc, params)
    print(DBCV)
    plot_join(embedding[clustered, 0], embedding[clustered, 1], result['LABELS'][clustered])

active = dfc.query("recency_class=='Activo1'|recency_class=='Activo2'|recency_class=='Reciente1'").copy()
indices = set(active.index)
mst_df = clf.hdbscan_.minimum_spanning_tree_.to_pandas()
mask = mst_df[(mst_df['from'].isin(indices)) & (mst_df['to'].isin(indices))]

rel, local_dbcv, counts = dbcv_score(result, active.index, 'euclidean', mask)
print(f" The DBCV for the active is {local_dbcv}")
print(f" The relative DBCV for the active is {rel}")
print(counts)
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"DenseClus_{formatted_time}.jolib"
dump(clf, f'{PATH}model.joblib')

sample = result[[0, 1, 2]]

prediction.approximate_predict(clf.hdbscan_, sample)