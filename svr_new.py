# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:23:02 2023

@author: ivoto
"""

#SVR 

#Regresion template

#Importamo librerias
import numpy as np #Para trabajar con math
import matplotlib.pyplot as plt # Para la vizualizacion de datos 
import pandas as pd #para la carga de datos 

#Importamos el dataSet 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values 
y = dataset.iloc[:, 2:3].values 


#Training & Test 
'''
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )
'''

#Escalado de variables(Datos)

#Escalamos el conjunto de entrenamiento
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X) #Ahora quedan escalados entre -1 y 1 pero es una STANDARIZACION (Normal) por lo que tendremos valores mayores a 1 y menores a -1
y = sc_y.fit_transform(y) #Solo detecta la transformacion y la aplica


#Ajustamos la regresion con el dataset

from sklearn.svm import SVR
regression = SVR(kernel='rbf')
regression.fit(X, y)


#Prediccion de nuestro modelos 
y_pred = regression.predict(sc_X.transform([[6.5]])) #tengo que hacer la tranformacion al escalado primero
y_pred = sc_y.inverse_transform([y_pred])
#Visualizacion de los resultados del Modelo Polinomico
#X_grid = np.arange(min(X), max(X), 0.1 )
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red') 
plt.plot(X, regression.predict(X),color = 'blue')
plt.title('Modelo de Regresion SVR ')
plt.xlabel('Posicion del empleado')
plt.ylabel=('Sueldo (en $)')
plt.show()