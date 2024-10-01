# Columnas para el Clustering

* total_item = Gasto total por semana en pesos a precios
de 2015
* tipo_negocio = 41 categorias 

* ?_tot = Total compra huevo tipo ? en pesos
* Momentum = índice de la tendencia de compra del cliente (apetito nacional)
* 

***
# Observaciones 
* L y XL son los que más se venden 
* Revisar freqmes por los saltos igual a 1
* dias compra (a,b) = (b,a)
* Quitar observaciones sin nombre
* 300 en recency
dfc0.loc[(300> dfc0['recency']) & (dfc0['recency']> 2),'recency'].hist(bins = 100)

# Workflow:
Cargar datos para predecir y tener listo el script de predicción,
usando Airflow 

***

# PREGUNTAS

* ¿RC01 qué tipo de negocio es?
* 

