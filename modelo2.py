import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

print("Xtrain")
print(X_train)
print("xtest: ")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Realizar predicciones con los datos de prueba
y_pred = model.predict(X_test)
print("Prediccion casos: ", y_pred)
# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'MAE: {mae}, R²: {rmse}')

#---------------------------------------------------------------------------------------

future_data = pd.read_csv('./clima_futuro.csv')

# Mostrar las primeras filas del dataset para revisión
print(future_data.head())

# Convertir la columna 'fecha' al formato datetime
future_data['fecha'] = pd.to_datetime(future_data['fecha'], format= '%Y-%m-%d')

# Mostrar las primeras filas del dataset después del preprocesamiento
print('Mostrar las primeras filas del dataset después del preprocesamiento')
print(future_data.head())

# Seleccionar las columnas de interés para las predicciones futuras
future_data = future_data[['fecha','precipitacion_corrientes', 'temperatura']]

# Usar el modelo entrenado para predecir los casos de dengue en fechas futuras
future_predictions = model.predict(future_data[['precipitacion_corrientes', 'temperatura']])
future_predictions = future_predictions.astype(int)
# Mostrar las predicciones
print(future_predictions)

# # # Añadir las predicciones a las fechas futuras y guardar en un archivo
future_data['predicted_casos'] = future_predictions
future_data.to_csv('predicciones_dengue_futuras.csv', index=False)

#Realizo esto para que se pueda mostrar la cantidad de casos por mes, solo para que sea mas visible en los graficos 
# Creo una columna adicional que solo contenga el año y mes en formato YYYY-MM
df_cleaned['fecha_por_mes'] = df_cleaned['fecha'].dt.to_period('M')

# Agrupar por 'fecha_por_mes' y sumar los casos de dengue por cada mes
casos_por_mes = df_cleaned.groupby('fecha_por_mes')['casos_corrientes'].sum().reset_index()

# Mostrar el resultado
print(casos_por_mes)

#REALIZO LO MISMO PARA LOS DATOS DE FECHAS FUTURAS
future_data['fecha_por_mes'] = future_data['fecha'].dt.to_period('M')

# Agrupar por 'fecha_por_mes' y sumar los casos de dengue por cada mes
casos_por_mes_futuro = future_data.groupby('fecha_por_mes')['predicted_casos'].sum().reset_index()

# Mostrar el resultado
print(casos_por_mes_futuro)



#import matplotlib.pyplot as plt
#GRAFICA PARA VER LA CANTIDAD DE CASOS EN EL CONJUNTO DE DATOS HISTORICO
# Graficar los casos de dengue por mes
plt.figure(figsize=(15, 6))

# Gráfico de líneas
plt.plot(casos_por_mes['fecha_por_mes'].astype(str), casos_por_mes['casos_corrientes'], marker='o', linestyle='-', color='b', label='Casos por mes')


# Etiquetas y título
plt.title('Cantidad de casos de Dengue por mes (YYYY-MM)')
plt.xlabel('Fecha (YYYY-MM)',labelpad = 2)
plt.ylabel('Cantidad de casos')

# Rotar las etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=90)
plt.subplots_adjust(bottom = 0.15) 

# Añadir leyenda
plt.legend()

# Mostrar gráfico
#plt.tight_layout()
plt.show()

#GRAFICA PARA VER LA CANTIDAD DE CASOS EN EL CONJUNTO DE DATOS HISTORICO
# Graficar los casos de dengue por mes
plt.figure(figsize=(15, 6))

# Gráfico de líneas
plt.plot(casos_por_mes_futuro['fecha_por_mes'].astype(str), casos_por_mes_futuro['predicted_casos'], marker='o', linestyle='-', color='b', label='Casos por mes')


# Opcional: Gráfico de barras (puedes descomentar si prefieres barras)
# plt.bar(casos_por_mes['fecha_por_mes'].astype(str), casos_por_mes['casos_corrientes'], color='skyblue')

# Etiquetas y título
plt.title('Cantidad de casos de Dengue por mes (YYYY-MM)')
plt.xlabel('Fecha (YYYY-MM)',labelpad = 2)
plt.ylabel('Cantidad de casos')

# Rotar las etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=90)
plt.subplots_adjust(bottom = 0.15) 

# Añadir leyenda
plt.legend()

# Mostrar gráfico
#plt.tight_layout()
plt.show()

# plt.figure(figsize=(14, 7))

# # Gráfico para el conjunto inicial
# plt.scatter(df_cleaned['temperatura'], df_cleaned['casos_corrientes'], color='blue', label='Datos históricos', alpha=0.5)
# plt.scatter(df_cleaned['precipitacion_corrientes'], df_cleaned['casos_corrientes'], color='green', label='Datos históricos (Precipitación)', alpha=0.5)

# # Gráfico para el conjunto futuro
# plt.scatter(future_data['temperatura'], future_predictions, color='red', label='Predicciones futuras', alpha=0.5)
# plt.scatter(future_data['precipitacion_corrientes'], future_predictions, color='orange', label='Predicciones futuras (Precipitación)', alpha=0.5)

# # Añadir leyenda y etiquetas
# plt.title('Comparación de casos de Dengue en base a Temperatura y Precipitación')
# plt.xlabel('Valores de Temperatura y Precipitación')
# plt.ylabel('Casos de Dengue')
# plt.legend()

# plt.tight_layout()
# plt.show()