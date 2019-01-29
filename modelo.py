# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:24:41 2018

@author: Flavio
"""
import sys
import numpy as np
import funciones_auxiliares

def DT(dataset, metrica, profundidad_maxima=None):
    """
    Funcion para crear el algoritmo del arbol de decision, toma 4 parametros:
    :param dataset: los datos con los que crear el arbol de decision, en la primera iteracion se usa el dataset entero.
    :param metrica: metica elegida-> entropia, gini o accuracy
    :param profundidad_maxima: default None, profundidad del arbol maxima
    :return arbol: arbol de decision generado
    """ 
    
    # se incializa el arbol
    arbol = {}
    
    # se cogen las caracteristicas
    features =  dataset.columns[:-1]
    
    # se coje el nombre de la columna clase
    clase = dataset.columns[-1]
    
    # A continuación se definen los criterios de parada, si se cumple alguno se retorna el nodo 'hoja'
    
    # Primer criterio: Si todas las filas pertenecen a la misma clase devolvemos una hoja
    if len(np.unique(dataset[clase])) <= 1:
        return np.unique(dataset[clase])[0] # hoja clase
    
   # Segundo criterio: si ya no hay caracteristicas a procesar devuleve la clase mayoritaria
   # o si la profundidad maxima que se introduce como parametro llega a cero
    elif len(features) == 0 or profundidad_maxima == 0:
        unique, counts = np.unique(dataset[clase],return_counts=True)
        return unique[np.argmax(counts)]
    
   # Finalmente si no se cumple ninguna parada el arbol puede seguir creciendo
    else:
    
        # Se selecciona la mejor caracteristica con respecto a la métrica escogida 
        valores_metrica = []
        if metrica == 'entropia':
            for f in features:
                valores_metrica.append(funciones_auxiliares.entropy_info_gain(dataset,f))
        elif metrica == 'accuracy':
            for f in features:
                valores_metrica.append(funciones_auxiliares.metrica_accuracy(dataset,f))
        elif metrica == 'gini':
            for f in features:
                valores_metrica.append(funciones_auxiliares.metrica_gini(dataset,f))
        else:
            sys.exit("La metrica %s no es correcta, posibles valores: entropia, accuracy o gini." % (metrica))
            
        # Se coje el indice de la mejor car
        if metrica == 'entropia' or metrica == 'accuracy':
            mejor_indice_car = np.argmax(valores_metrica)
        else: # Para el caso de gini se quiere minimizar
            mejor_indice_car = np.argmin(valores_metrica)
        
        # Se selecciona la mejor acaracteristica
        mejor_car = features[mejor_indice_car]
        
        # Se crea la estructura del arbol. El nodo raiz toma el nombre de la mejor carateristica del dataset
        arbol = {mejor_car:{}}
        
        # La rama crece por debajo del nodo raiz a cada posible caracteristica que pueda tomar este nodo
        for value_feature in np.unique(dataset[mejor_car]):

            # Se divide el dataset a lo largo del valor de la característica 
            # con la mayor ganancia de información y se crea el sub_dataset
            sub_dataset = dataset.where(dataset[mejor_car] == value_feature).dropna()
            
            # Se borra la columna con mejor caracteristica que ya se ha procesado
            sub_dataset = sub_dataset.drop(columns=mejor_car, axis=1)
            
            # Se llama al de nuevo a esta funcion de forma recursiva con los nuevos parametros
            if profundidad_maxima == None:
                sub_arbol = DT(sub_dataset, metrica)
            else: 
                sub_arbol = DT(sub_dataset, metrica, profundidad_maxima - 1)
            
            # Se añade los nuevos nodos
            arbol[mejor_car][value_feature] = sub_arbol

    return(arbol)
