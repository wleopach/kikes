import pandas as pd
from plot_utils import plot_join
from utils import dbcv_score, extract_numerical
from config import PATH
from joblib import dump
from predict import Pred
import datetime
import random
from senseclus import fit_SenseClus
from hdbscan import prediction
import numpy as np

dfc0 = pd.read_csv(f'{PATH}ventasxsemana2_train_11001_cluster_270623.csv')
max_year = max(dfc0['year'])
max_week_dict = dict()
for row in dfc0.iterrows():
    max_week_dict.setdefault(f"{row[1]['year']}", row[1]['numweek'])
    if row[1]['numweek'] > max_week_dict[f"{row[1]['year']}"]:
        max_week_dict[f"{row[1]['year']}"] = row[1]['numweek']

num_weeks = dict()
for year in dfc0['year'].unique():
    for week in range(1, max_week_dict[f"{year}"] + 1):
        if year == max_year:
            num_weeks[(year, week)] = max_week_dict[f"{year}"] - week
        else:
            num_weeks[(year, week)] = (max_week_dict[f"{year}"] - week +
                                       (max_year - year - 1) * 52 + max_week_dict[f"{max_year}"])

dfc0['NW'] = dfc0.apply(lambda x: num_weeks[x['year'], x['numweek']], axis=1)

dfc0['cliente_descuento'] = dfc0['cliente_descuento'].astype(object)
dfc0 = dfc0.query("NW< 26")

dfc = dfc0[['total_item', ' B _tot', 'tipo_negocio2', 'L_tot', 'L_PET_tot', 'M_tot', 'M_PET_tot', 'cliente_descuento',
            'XL_tot', 'XL_PET_tot', 'YUMBO_tot', 'Momentum', 'freqmes', 'dias_compra', 'recency_class']].copy()

numerical = dfc.select_dtypes(include=["float", "int"])
null_cols = []
for col in numerical.columns:
    if sum(dfc[col]) == 0:
        null_cols.append(col)
col_numerics = ['total_item', ' B _tot', 'L_tot', 'L_PET_tot', 'M_tot', 'M_PET_tot',
       'XL_tot', 'XL_PET_tot', 'YUMBO_tot', 'Momentum', 'freqmes']

noise = np.random.normal(loc=0, scale=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.000001, 0.001],
                         size=dfc[col_numerics].shape)
dfc[col_numerics] = dfc[col_numerics] + noise
dfc.drop(columns=null_cols, inplace=True)
dfc.dropna(inplace=True)
original_index = dfc.reset_index(drop=False)['index']
dfc.reset_index(inplace=True, drop=True)
params = dict()
params['cluster_selection_method'] = "eom"
params['min_samples'] = 10
params['n_neighbors'] = 60
params['n_components'] = 3
params['min_cluster_size'] = int(len(dfc) * 1.5 / 100)
params['umap_combine_method'] = "intersection_union_mapper"
params['SEED'] = None
DBCV = -1
while DBCV < 0.3:
    print(f"runing with {params['umap_combine_method']}")
    embedding, clustered, result, DBCV, coverage, clf = fit_SenseClus(dfc, params)
    params['umap_combine_method'] = random.choice(["intersection_union_mapper", "intersection", "union"])
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
dump(clf, f'{PATH}{file_name}')
original_data = dfc0.loc[original_index]
sample = result[[0, 1, 2]]

predictor = Pred(clf)

pred1 = predictor.predict_new(dfc.head(10))
pred2 = prediction.approximate_predict(clf.hdbscan_, sample.head(10))
