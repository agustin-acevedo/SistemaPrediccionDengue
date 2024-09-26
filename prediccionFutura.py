import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar el conjunto de datos unificado
data = pd.read_csv('./DataSet/Dengue_Dataset.csv', parse_dates=['fecha'], index_col='fecha')

# Mostrar las primeras filas para verificar
print(data.head())
# Verificar las columnas
print(data.columns)
# Transformar los casos de dengue con logaritmo
data['log_cases'] = np.log(data['casos_corrientes'] + 1)

# Verificar las columnas
print(data.columns)

# Verificar si hay NaNs o infinitos en las variables exógenas (covariables)
print(data[['precipitacion_corrientes','temperatura']].isna().sum())  # NaN check
print(np.isinf(data[['precipitacion_corrientes','temperatura']]).sum())  # Inf check

# Eliminar filas que contienen NaN o Inf
data_cleaned = data[['precipitacion_corrientes','temperatura']].replace([np.inf, -np.inf], np.nan).dropna()

# Asegúrate de que el conjunto de datos sea consistente
data = data.loc[data_cleaned.index]


# Rellenar NaN con la media de cada columna
#data[['precipitacion_corrientes','temperatura']] = data[['precipitacion_corrientes','temperatura']].fillna(data.mean())

# También puedes usar el método 'ffill' para rellenar con el valor anterior
# data[['precipitacion_corrientes','temperatura']] = data[['precipitacion_corrientes','temperatura']].fillna(method='ffill')

# Reemplazar valores infinitos con el máximo valor permitido en las columnas
data[['precipitacion_corrientes','temperatura']] = data[['precipitacion_corrientes','temperatura']].replace([np.inf, -np.inf], np.nan)
data[['precipitacion_corrientes','temperatura']] = data[['precipitacion_corrientes','temperatura']].fillna(data.max())


# Definir el modelo SARIMA con variables climáticas como covariables
sarimax_model = SARIMAX(
    data['log_cases'],  # La serie temporal de casos transformados
    order=(1, 0, 0),  # Parámetros ARIMA (p,d,q)
    seasonal_order=(1, 1, 0, 12),  # Parámetros estacionales (P,D,Q,s)
exog=data[['precipitacion_corrientes','temperatura']]  # Variables climáticas
)

# Ajustar el modelo
sarimax_results = sarimax_model.fit(disp=False)

# Creo un rango de fechas futuras (Tomando a partir de la ultima fecha que tiene el dataset)
future_dates = pd.date_range(start='2019-01-01', end='2020-12-31', freq='M')

# Obtenemos los valores de las variables. Como no las conocemos hacemos una estimacion de la media de los valores anteriores 

future_exog = pd.DataFrame({
    'temperatura': [data['temperatura'].mean()] * len(future_dates),
    'precipitacion_corrientes': [data['precipitacion_corrientes'].mean()] * len(future_dates),
}, index=future_dates)
