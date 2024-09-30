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
future_dates = pd.date_range(start='1/1/2018', end='31/12/2021', freq='W', name = 'fecha')

# Calcular la media y la desviación estándar de las variables climáticas pasadas
mean_temp = data['temperatura'].mean()
std_temp = data['temperatura'].std()

mean_precip = data['precipitacion_corrientes'].mean()
std_precip = data['precipitacion_corrientes'].std()

# Generar valores futuros usando una distribución normal alrededor de la media
np.random.seed(42)  # Para resultados reproducibles
future_exog = pd.DataFrame({
    'temperatura': np.random.normal(mean_temp, std_temp, len(future_dates)),
    'precipitacion_corrientes': np.random.normal(mean_precip, std_precip, len(future_dates))
}, index=future_dates)
# Con esto remplazo los valores negativos en la columna de precipitacion ya que no tiene sentido que sea negativo, los cambio por 0
future_exog['precipitacion_corrientes'] = future_exog['precipitacion_corrientes'].apply(lambda x: max(x, 0))

print(future_exog)

# Guardar las fechas futuras y las variables climáticas en un archivo CSV
future_exog.to_csv('clima_futuro.csv', index= True)

print("Archivo CSV guardado correctamente.")

#------------------------Se empieza con la parte de predicciones-----------------------------

#Continuar aca - investigar como modificar el formato de la fecha 




# Hacer predicciones para las fechas futuras
# Verifica el formato de las fechas en el índice del DataFrame

#print(data.index)

#print(future_exog.index)

# Si 'future_dates' está en formato string, conviértelo a datetime
#data = pd.to_datetime(data,yearfirst = True)

# future_forecast = sarimax_results.get_prediction(
#     start='2019-01-06', 
#     end='2022-12-25', 
#     exog=future_exog
# )

# # Obtener las predicciones
# predicted_cases_future = future_forecast.predicted_mean

# # Revertir la transformación logarítmica para obtener las predicciones reales
# predicted_cases_future_exp = np.exp(predicted_cases_future) - 1

# # Mostrar las predicciones
# print(predicted_cases_future_exp)








# # Obtenemos los valores de las variables. Como no las conocemos hacemos una estimacion de la media de los valores anteriores 

# future_exog = pd.DataFrame({
#     'temperatura': [data['temperatura'].mean()] * len(future_dates),
#     'precipitacion_corrientes': [data['precipitacion_corrientes'].mean()] * len(future_dates),
# }, index=future_dates)

# # Mostrar las fechas futuras y las estimaciones climáticas
# print(future_exog)



# # En esta parte lo que hago es empezar a realizar las predicciones futuras 
# future_forecast = sarimax_results.get_prediction(
#     start=future_dates[0], 
#     end=future_dates[-1], 
#     exog=future_exog
# )

# # Obtener las predicciones
# predicted_cases_future = future_forecast.predicted_mean

# # Revertir la transformación logarítmica para obtener las predicciones reales
# predicted_cases_future_exp = np.exp(predicted_cases_future) - 1

# # Mostrar las predicciones
# print(predicted_cases_future_exp)

# # Crear un gráfico con los datos históricos y las predicciones futuras
# plt.figure(figsize=(10, 6))

# # Graficar los datos históricos
# plt.plot(data.index, data['casos_corrientes'], label='Casos históricos')

# # Graficar las predicciones futuras
# plt.plot(future_dates, predicted_cases_future_exp, label='Predicciones futuras', color='red')

# # Etiquetas y título
# plt.xlabel('Fecha')
# plt.ylabel('Casos de Dengue')
# plt.title('Predicciones de Casos de Dengue para Fechas Futuras')
# plt.legend()

# plt.show()


