# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:23:34 2018

@author: Flavio
"""
import sys
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split


def readCSV(nombre):
    """
    Funcion para leer el dataset en formato csv
    :param nombre: nombre del dataset a leer, ej: dataset.csv
    :return dataset: dataset en formato dataframe
    """
    with open(nombre, newline = "") as csvfile:
        try:
            csv.Sniffer().sniff(csvfile.read(), delimiters = ",")
            dataset = pd.read_csv(nombre, sep=',')
        except:
            sys.exit("Fallo al abrir el csv, el delimitador es ','.")
    
    return(dataset)
    

def split_dataset(dataset): 
   """
   Funcion para separa el dataset en datos de entrenamiento y test
   :param dataset: dataset a utilizar 
   :return (training_data, test_data): datos de entrenamiento y datos de test
   """
    
   clase = dataset.columns[-1]
    
   # Se cojen los datos en  
   y = dataset[clase].values
   X = dataset.drop(columns=clase, axis=1)
   X = X.values
   
   # 20% de datos para test
   # Para que divida el dataset de la misma forma en cada ejecucion usar seed 'random_state'
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,  random_state=26)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
  
   features = dataset.columns[:-1]

   training_data = pd.DataFrame(X_train, columns=features)
   training_data[clase] = y_train
   
   test_data = pd.DataFrame(X_test, columns=features)
   test_data[clase] = y_test
   
   return(training_data, test_data)


def entropy(columna):
    """      
    Funcion para calcular la entropia
    entropy = - SUM(P(clasei) * log2(clasei))
    :param columna: columna del dataset a calcular su entropia
    :return e: entropia
    """
    # print(columna)
    size_dataset=len(columna) # longitud de filas
    # unique: array con las clases unicas
    # counts: array con la cantidad de cada clase
    unique, counts = np.unique(columna, return_counts=True)
    e=0 # entropia
    for i in range(len(unique)):
        e += np.sum((counts[i]/size_dataset)*np.log2(counts[i]/size_dataset))
    return(-e)

def entropy_info_gain(dataset, feature_name):
    """ 
    Calcula la ganancia de información en la entropia, parametros:
    :param dataset: el dataset a ser utilizado con los datos
    :param feature_name: el nombre de la carateristica para la que se va a calcular la ganancia de informacion
    :return info_gain: informacion ganada entropia
    """
    
    # se coge el nombre de la columna clase
    target_name = dataset.columns[-1]
    
    # se calcula la entropia total del dataset
    entropia_total = entropy(dataset[target_name])
    ##print(entropia_total)
    
    # calcular las clases unicas y la cantidad de cada uno para la caracteristica(feature_name)
    unique, counts = np.unique(dataset[feature_name], return_counts=True)

    # lista que contiene series con los subdatasets con posibles valores de las clases
    # que puede tomar cada valor de la caracteristica pasada como parametro
    split_valores_car = []  
    # separa en dataframes para cada valor, poniendo las filas en NaN, dropna: elimina las filas NaN
    for i in range(len(unique)):
        split_valores_car.append(dataset.where(dataset[feature_name]==unique[i]).dropna()[target_name])
    for serie in split_valores_car:
        serie.reset_index(drop=True,inplace=True)
    """for i in split_valores_car:
        print(i)"""
    
    # se calcula la entropia para cada posible valor de la caracteristica(feature_name)
    entropia_cada_valor = []
    for i in split_valores_car:       
        entropia_cada_valor.append(entropy(i))
    """for i in entropia_cada_valor:
        print(i)"""
        
    # informacion de cada caracteristica
    info = 0
    for i in range(len(entropia_cada_valor)):
        info += counts[i]/len(dataset) * entropia_cada_valor[i] 
        
    # finalmente se calcula la informacion ganada: resta de la entropia total menos la entropia para cada caracteristica
    info_gain = entropia_total - info;

    return(info_gain)


def metrica_accuracy(dataset, feature_name):
    '''
       Funcion para calcular el accuracy respecto a la clase mayoritaria
       :param dataset: datos del dataset
       :param feature_name: nombre de la caracteristica
       :return (1 - errores_totales/len(dataset)): accuracy
    '''
    # se coge el nombre de la columna clase
    target_name = dataset.columns[-1]
    
    errores_totales = 0 # Errores para la caracteristica seleccionada

    # calcular las clases unicos y la cantidad de cada uno para la caracteristica(feature_name)
    unique, counts = np.unique(dataset[feature_name], return_counts=True)

    # lista que contiene series con los subdatasets con posibles valores de las clases
    # que puede tomar cada valor de la caracteristica pasada como parametro
    split_valores_car = []
    # separa en dataframes para cada valor, poniendo las filas en NaN, dropna: elimina las filas NaN
    for i in range(len(unique)):
        split_valores_car.append((dataset.where(dataset[feature_name]==unique[i]).dropna()[target_name])) 
    for serie in split_valores_car:
        serie.reset_index(drop=True,inplace=True)

    # se obtiene la clase mayoritaria respecto a los subsets creados
    clases_mayoritarias = []
    for i in split_valores_car:
        unique, counts = np.unique(i,return_counts=True)
        clases_mayoritarias.append(unique[np.argmax(counts)])
        
    # Obtenemos el numero de errores y acumulamos
    errores = 0
    for i in range(len(clases_mayoritarias)):
        serie = split_valores_car[i]
        for j in range(len(split_valores_car[i])):
            if(serie[j] != clases_mayoritarias[i]): # Comprobamos si la "prediccion" falla
                errores += 1 # Incrementamos el numero de errores
        errores_totales += errores
        errores = 0
   
    return(1 - errores_totales/len(dataset))
    
    
def metrica_gini(dataset, feature_name):
    """
    Funcion para calcular el indice de gini respecto de cada caracteristica pasada por parametro, del dataset
    :param dataset: datos del dataset
    :param feature_name: nombre de la caracteristica
    :return gini: indice de gini para esa columna pasada feature_name
    """
    # se coge el nombre de la columna clase
    target_name = dataset.columns[-1]
    
    # calcular las clases unicos y la cantidad de cada uno para la caracteristica(feature_name)
    unique, counts = np.unique(dataset[feature_name], return_counts=True)
    
    # lista que contiene series con los subdatasets con posibles valores de las clases
    # que puede tomar cada valor de la caracteristica pasada como parametro
    split_valores_car = []
    # separa en dataframes para cada valor, poniendo las filas en NaN, dropna: elimina las filas NaN
    for i in range(len(unique)):
        split_valores_car.append((dataset.where(dataset[feature_name]==unique[i]).dropna()[target_name])) 
    for serie in split_valores_car:
        serie.reset_index(drop=True,inplace=True)
        
    # se calcula sum(pi^2) para cada subset
    gini_subset = 0 
    indice_gini = [] # para cada subset se calcula el indice gini
    gini = 0
    for j in range(len(split_valores_car)):
        unique, counts = np.unique(split_valores_car[j],return_counts=True)
        #print(unique)
        #print(counts)
        for i in counts:  # indice gini por cada subset, Gini(D)
            gini_subset += (i/len(split_valores_car[j])) ** 2 # porcion de filas de cada subset para la car i

        indice_gini.append(1-gini_subset)
        gini_subset = 0
        gini += (len(split_valores_car[j])/len(dataset)) * indice_gini[j] # formula
    
    return gini


def prediccion(arbol, dato, dataset):
    """
    Funcion para predecir una muestra con los valores de las caracteristicas
    :param árbol:  ́árbol de decisión
    :param dato: una fila de la base de datos sobre la que queremos predecir la clase.
    :param dataset: dataset original usado para calcula clase mayoritaria

    """
    # Coge cada caracteristica/clave del conjunto de dato dict
    claves_dato = dato.keys()
    # Coge cada clave del conjunto de datos del arbol tipo dict
    claves_arbol = arbol.keys()
    
    # por defecto la clase será la mayoritaria
    unique, counts = np.unique(dataset[dataset.columns[-1]],return_counts=True)
    clase = unique[np.argmax(counts)]
    
    # Recorre la lista de claves de dato
    for clave in claves_dato:
        # Si la clave esta es el nodo raiz del arbol
        if clave in claves_arbol:
            # intenta lo siguiente
            try:
                valor = dato[clave] # coger el valor de la clave en el dict dato pasado como parametro a la func
                # el valor anterior pasa a ser una clave2 en el arbol[clave][clave2]
                # Coge el valor(puede ser otro dict) en el arbol con Clave:clave -> Clave:valor
                # la variable clase es el valor en el arbol
                clase = arbol[clave][valor] 
            except:
                # se devuelve la clase mayoritaria del dataset
                return clase
             
            # Si el contenido de clase tiene un diccionario sigue buscando dentro de este
            # que pasa a ser el nueva variable arbol, hasta que la clase sea un nodo hoja
            if isinstance(clase, dict):
                return prediccion(clase, dato, dataset)
            else:
                return clase # retorna la clase: nodo hoja

def accuracy_del_arbol(dataset,arbol):
    """
    Funcion para comprobar el accuracy del modelo arbol que se ha creado antes
    :param dataset: dataset cargado
    :param arbol: modelo con el arbo creado
    :return accuracy: accuracy del arbol
    """
    
    # Se coge el nombre de la columna clase
    target_name = dataset.columns[-1]
    
    # Crea un array con las filas y borra la columna clase del dataset que se le pasa por parametro
    # to_dict: Convertir el dataframe a un diccionario, records -> [{column -> value}, … , {column -> value}]
    data = dataset.iloc[:,:-1].to_dict(orient = "records")
    
    # Crea un dataframe llamado predicho con la columna de prediccion
    predicho = pd.DataFrame(columns=["prediccion"]) 

    # Calcula la prediccion
    for i in range(len(data)):
        predicho = predicho.append({'prediccion': prediccion(arbol,data[i],dataset)}, ignore_index=True)
    
    # Se calcula el accuracy
    count = 0
    for i in range(len(predicho)):
        if  predicho.iloc[i]["prediccion"] == dataset.iloc[i][target_name]:
            count += 1
            
    accuracy = count/len(dataset)*100
    
    return(accuracy)




