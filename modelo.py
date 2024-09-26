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


# Hacer predicciones para el periodo de validación
forecast = sarimax_results.get_prediction(start='06/01/2007', end='29/12/2018', exog=data[['precipitacion_corrientes','temperatura']])
predicted_cases = forecast.predicted_mean

# Revertir la transformación logarítmica
predicted_cases_exp = np.exp(predicted_cases) - 1

# Evaluar el rendimiento del modelo
mae = mean_absolute_error(data['casos_corrientes']['06/01/2007':'29/12/2018'], predicted_cases_exp)
r2 = r2_score(data['casos_corrientes']['06/01/2007':'29/12/2018'], predicted_cases_exp)

print(f'MAE: {mae}, R²: {r2}')


# Gráfico de predicciones vs valores reales
plt.figure(figsize=(20, 10))
plt.plot(data.index, data['casos_corrientes'], label='Casos reales')
plt.plot(data.index, predicted_cases_exp, label='Predicciones', color='red')
plt.xlabel('Fecha')
plt.ylabel('Casos de dengue')
plt.title('Predicción de Casos de Dengue con SARIMA')
plt.legend()
plt.show()

#import matplotlib.pyplot as plt

# Crear un gráfico de líneas para los casos de dengue, temperatura y precipitación
plt.figure(figsize=(12, 6))

# Graficar casos de dengue
plt.plot(data.index, data['casos_corrientes'], label='Casos de Dengue', color='blue')

# Graficar la temperatura
plt.plot(data.index, data['temperatura'], label='Temperatura (°C)', color='red')

# Graficar la precipitación
plt.plot(data.index, data['precipitacion_corrientes'], label='Precipitación (mm)', color='green')

# Etiquetas y título
plt.xlabel('Fecha')
plt.ylabel('Valores')
plt.title('Casos de Dengue, Temperatura y Precipitación a lo largo del tiempo')
plt.legend()

plt.show()

# Gráfico de dispersión de temperatura vs casos de dengue
plt.figure(figsize=(8, 6))
plt.scatter(data['temperatura'], data['casos_corrientes'], color='red')

# Etiquetas
plt.xlabel('Temperatura (°C)')
plt.ylabel('Casos de Dengue')
plt.title('Relación entre Temperatura y Casos de Dengue')

plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data['precipitacion_corrientes'], data['casos_corrientes'], color='red')

# Etiquetas
plt.xlabel('Precipitacion (mm)')
plt.ylabel('Casos de Dengue')
plt.title('Relación entre Precipitacion y Casos de Dengue')

plt.show()