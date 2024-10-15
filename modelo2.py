import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Cargar el archivo CSV
file_path = './DataSet/Dengue_Dataset.csv'
df = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset para revisión
print(df.head())

# Convertir la columna 'fecha' al formato datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

# Seleccionar las columnas relevantes para el modelo
df_cleaned = df[['fecha', 'casos_corrientes', 'precipitacion_corrientes', 'temperatura']]

# Verificar si hay valores nulos
missing_data = df_cleaned.isnull().sum()

# Mostrar las primeras filas del dataset después del preprocesamiento
print('Mostrar las primeras filas del dataset después del preprocesamiento')
print(df_cleaned.head(), missing_data)

# Vamos a imputar el valor faltante de temperatura utilizando la interpolación de los valores adyacentes
df_cleaned['temperatura'] = df_cleaned['temperatura'].interpolate()

# Verificar si aún hay valores nulos
missing_data_after_imputation = df_cleaned.isnull().sum()

# Mostrar el resultado de la imputación
print('Mostrar el resultado de la imputación')
print(df_cleaned.head(), missing_data_after_imputation)


# Variables predictoras y variable objetivo
X = df_cleaned[['precipitacion_corrientes', 'temperatura']]
y = df_cleaned['casos_corrientes']

# División en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones con los datos de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'MAE: {mae}, R²: {rmse}')

# Vamos a generar un rango de fechas futuras, manteniendo la periodicidad semanal
# Seleccionamos la última fecha del dataset
last_date = df_cleaned['fecha'].max()

# Generamos las próximas 10 semanas
future_dates = pd.date_range(start=last_date, periods=10, freq='W', name ='fecha')

# Para extrapolar los valores de precipitación y temperatura, calculamos el promedio semanal del dataset existente
#mean_precipitation = df_cleaned['precipitacion_corrientes'].mean()
#mean_temperature = df_cleaned['temperatura'].mean()

mean_temp =  df_cleaned['temperatura'].mean()
std_temp =  df_cleaned['temperatura'].std()

mean_precip =  df_cleaned['precipitacion_corrientes'].mean()
std_precip =  df_cleaned['precipitacion_corrientes'].std()

# # Crear un nuevo DataFrame con las fechas futuras y los valores extrapolados de precipitación y temperatura
# future_data = pd.DataFrame({
#     'fecha': future_dates,
#     'temperatura': np.random.normal(mean_temp, std_temp, len(future_dates)),
#     'precipitacion_corrientes': np.random.normal(mean_precip, std_precip, len(future_dates))
# })
 #Creo un rango de fechas futuras (Tomando a partir de la ultima fecha que tiene el dataset)
#future_dates = pd.date_range(start=last_date, periods=10, freq='W')


# Generar valores futuros usando una distribución normal alrededor de la media
np.random.seed(42)  # Para resultados reproducibles
future_data = pd.DataFrame({
    'fecha': future_dates,
    'temperatura': np.random.normal(mean_temp, std_temp, len(future_dates)),
    'precipitacion_corrientes': np.random.normal(mean_precip, std_precip, len(future_dates))
}, index=future_dates)
# Con esto remplazo los valores negativos en la columna de precipitacion ya que no tiene sentido que sea negativo, los cambio por 0
future_data['precipitacion_corrientes'] = future_data['precipitacion_corrientes'].apply(lambda x: max(x, 0))


# Realizar las predicciones utilizando el modelo entrenado
X_future = future_data[['precipitacion_corrientes', 'temperatura']]
future_data['predicted_casos_corrientes'] = model.predict(X_future)

# Mostrar las predicciones para las próximas 10 semanas
print(future_data[['fecha', 'predicted_casos_corrientes']])
