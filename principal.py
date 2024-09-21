import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



#DECLARACIÓN CONSTANTES
NUM_CARACT = [20]
#CLASIFICADORES = ['DT','NB','ANN','SVM','RF','GBM','VC']
CLASIFICADORES = ['SVM']
#CLASIFICADORES = ['DT','NB']



#APERTURA FICHERO
print ("LECTURA DE FICHERO")
reader = csv.reader(open("./DataSet/Dengue_Dataset.csv"), delimiter=",")
raw_data = list(reader)

np_data = np.asarray(raw_data, dtype=None)

X = np_data[:, 0:-2]  # Seleccionar todas las columnas menos las dos últimas
y = np_data[:, -2]   # Seleccionar la penúltima columna (etiqueta como cadena)





#AGRUPAMOS ETIQUETAS
print("AGRUPAMOS ETIQUETAS")
for i in range(0,len(y)):
    if (y[i] in ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']):
        y[i] = 'Probe'
    elif (y[i] in ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']):
        y[i] = 'R2L'
    elif (y[i] in ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']):
        y[i] = 'U2R'
    elif (y[i] in ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm', 'mailbomb']):
        y[i] = 'DoS'
    elif (y[i] == 'normal'):
        y[i] = 'Normal'
    else:
        y[i] = 'Unknown'





#PREPROCESAMIENTO
print("PREPROCESAMIENTO")

#Preprocesamos el array para que únicamente contega valores normalizados de tipo float32
le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1].astype(str))
X[:,2] = le.fit_transform(X[:,2].astype(str))
X[:,3] = le.fit_transform(X[:,3].astype(str))
y = le.fit_transform(y)

X = X.astype(float)
y = y.astype(float)

#Eliminamos NAN
X = np.nan_to_num(X.astype(float))





#DIVISIÓN DEL DATASET
print ("DIVISIÓN DEL DATASET")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print ("X_train, y_train:", X_train.shape, y_train.shape)
print ("X_test, y_test:", X_test.shape, y_test.shape)


# RESUMEN DE LOS DATOS Y GUARDADO EN DISCO
unique_elements, counts_elements = np.unique(le.inverse_transform(y_train.astype(int)), return_counts=True)
print ("Número de elementos de cada clase en el Train Set:")
print(np.asarray((unique_elements, counts_elements)))
with open("./datos/train_descr.txt","w") as f:
    f.write(str(np.asarray((unique_elements, counts_elements))))

unique_elements, counts_elements = np.unique(le.inverse_transform(y_test.astype(int)), return_counts=True)
print ("Número de elementos de cada clase en el Test Set:")
print(np.asarray((unique_elements, counts_elements)))
with open("./datos/test_descr.txt","w") as f:
    f.write(str(np.asarray((unique_elements, counts_elements))))








# #Cargar el dataset
# df = pd.read_csv('./Dataset/Dengue_Dataset.csv')

# #Ver las primeras filas del dataset
# print(df.head())

# #Ver información general del dataset
# print(df.info())

# #Descripción estadística básica
# print(df.describe())


# X = df[['precipitacion_corrientes', 'temperatura']]
# y = df['casos_dengue']

# #Dividir en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print ("X_train, y_train:", X_train.shape, y_train.shape)
# print ("X_test, y_test:", X_test.shape, y_test.shape)

# #RESUMEN DE LOS DATOS Y GUARDADO EN DISCO

# #Resumen estadístico del conjunto de entrenamiento
# print("Resumen estadístico de X_train:")
# print(X_train.describe())

# print("\nResumen estadístico de y_train:")
# print(y_train.describe())

# #Resumen estadístico del conjunto de prueba
# print("\nResumen estadístico de X_test:")
# print(X_test.describe())

# print("\nResumen estadístico de y_test:")
# print(y_test.describe())
