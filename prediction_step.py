import pandas as pd
import numpy as np
from config import PATH
from predict import Pred
import joblib
clf = joblib.load(f"{PATH}DenseClus_2023-06-29_21-33-08.joblib")
pred = Pred(clf)

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

# noise = np.random.normal(loc=0, scale=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.000001, 0.001],
#                          size=dfc[col_numerics].shape)
# dfc[col_numerics] = dfc[col_numerics] + noise
dfc.drop(columns=null_cols, inplace=True)
dfc.dropna(inplace=True)
original_index = dfc.reset_index(drop=False)['index']
dfc.reset_index(inplace=True, drop=True)

new = pred.predict_new(dfc.head(100))
