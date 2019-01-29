# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:35:55 2018

@author: Flavio
"""
import sys
import modelo
import funciones_auxiliares
import pandas as pd

def inicio(datos, metrica, split=None, profundidad_maxima=None):
    """
    Funcion para iniciar iniciar el modelo y para hacer la prueba de prediccion, 
    tambien saca el accuracy del modelo dependiendo de la metrica elegida
    :param datos: el conjunto de datos de entrenamiento (en formato dataframe de Pandas)
    :param metrica: nombre de la métrica: accuracy o gini o entropia
    :pram split: split del dataset usar 'split' o 'no_split'
    :param profundidad_maxima: profundidad maxima del arbol 
    """

    # Separar los datos en train y test
    if split == 'split':
        training_data = funciones_auxiliares.split_dataset(datos)[0]
        test_data = funciones_auxiliares.split_dataset(datos)[1]  
    elif split == 'no_split' or split == None:
        training_data = datos
        test_data = datos
    else:
        sys.exit("Parametros incorrectos, más información con -h.")
        
    # Se crea el modelo del arbol y se entrena
    arbol = modelo.DT(training_data, metrica, profundidad_maxima)
    
    print("========================================================")
    
    # Se printea el arbol de decision, que es de tipo diccionario
    print("El siguiente diccionario contiene el arbol de decision:")
    print(arbol)
    
    print("========================================================")
  
    # Se saca el accuracy del arbol
    print("El accuracy del modelo con metrica %s es: %.2f%%: " % (metrica, funciones_auxiliares.accuracy_del_arbol(test_data,arbol))) 
   
    print("========================================================")
    

    # Se prueba a predecir una muestra/fila de datos
    fila = 4 # fila del dataset original a predecir, empieza en 0
    if fila >= 0 or fila <= len(datos)-1:

        datos_a_predeccir = datos.iloc[fila][:-1]
        
        keys = datos.iloc[fila][:-1].keys()
        cadena = "Para las caracteristicas-> "
        
        #d = {'car1':0, 'car2':0, 'car3': 1}
        #datos_a_predeccir = pd.Series(d)
        
        for i in range(len(datos_a_predeccir)):
            cadena = cadena + ("%s: %s, " % (str(keys[i]),str(datos_a_predeccir[i])))
            
        print(cadena)
    
        print("Predice que es de la clase-> %s" % (str(funciones_auxiliares.prediccion(arbol,datos_a_predeccir,datos))))
    
    else:
        print("Prediccion: el dataset no tiene la fila %d, prueba con otra." % (fila))
        
    print("========================================================")
    

def main(argv):
    """
    Funcion main del proyecto
    :param argv: elementos que se pasan al ejecutar el proyecto
    argv[0] -> nombre del archivo py
    argv[1] -> nombre del fichero csv con el dataset
    argv[2] -> nombre de la metrica: accuracy o gini o entropia
    argv[3](opcional) -> split del dataset usar 'split' o 'no_split'
    argv[4](opcional) -> profundidad maxima del arbol (numero entero)
    """
    if "-h" in sys.argv:
        print ("argv[0] -> nombre del archivo py")
        print ("argv[1] -> nombre del fichero csv con el dataset")
        print ("argv[2] -> nombre de la metrica: accuracy o gini o entropia")
        print ("argv[3](opcional) -> split del dataset usar 'split' o 'no_split'")
        print ("argv[4](opcional)-> profundidad maxima del arbol (numero entero)")
        sys.exit(1)
        
    list_metricas = ['accuracy', 'gini', 'entropia']
    splited = ['no_split', 'split']
    
    if len(sys.argv) == 5:
        result =  any(elem in argv[2]  for elem in list_metricas)
        result2 =  any(elem in argv[3]  for elem in splited)
        if result and result2:
            try:
                profundidad_maxima = int(argv[4])
            except:
                sys.exit("La profundidad no es un entero.")
            else:
                dataset = funciones_auxiliares.readCSV(argv[1])
                inicio(dataset, argv[2], argv[3], profundidad_maxima)

        else:
            sys.exit("Parametros incorrectos, más información con -h.")
    elif len(sys.argv) == 4:
        result =  any(elem in argv[2]  for elem in list_metricas)
        if argv[3].isdigit() == False:
            result2 =  any(elem in argv[3]  for elem in splited)
        else:
            result2 = True
        
        if result and result2:
            if argv[3].isdigit():
                try:
                    profundidad_maxima = int(argv[3])
                except:
                    sys.exit("La profundidad no es un entero.")
                else:
                    dataset = funciones_auxiliares.readCSV(argv[1])
                    inicio(dataset, argv[2], None, profundidad_maxima)
            else:
                 dataset = funciones_auxiliares.readCSV(argv[1])
                 inicio(dataset, argv[2], argv[3])
        else:
            sys.exit("Parametros incorrectos, más información con -h.")
    elif len(sys.argv) == 3:
        result =  any(elem in argv[2]  for elem in list_metricas)
        if result:
            dataset = funciones_auxiliares.readCSV(argv[1])
            inicio(dataset, argv[2])
        else:
            sys.exit("Parametros incorrectos, más información con -h.")
        
    else:
        sys.exit("Parametros incorrectos, más información con -h.")
    
if __name__ == "__main__":
    main(sys.argv)