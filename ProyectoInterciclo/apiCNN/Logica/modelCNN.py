import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import model_from_json
import itertools
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator

def cargarRNN(nombreArchivoModelo,nombreArchivoPesos):
        
    # Cargar la Arquitectura desde el archivo JSON
    with open(nombreArchivoModelo+'.json', 'r') as f:
        model = model_from_json(f.read())

    # Cargar Pesos (weights) en el nuevo modelo
    model.load_weights(nombreArchivoPesos+'.h5')  

    print("Red Neuronal Cargada desde Archivo") 
    return model

def predict(direccion):
    CATEGORIAS = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]
    IMG_SIZE = 150
    nombreArchivoModelo='apiCNN/Logica/arquitecturaOptimizada'
    nombreArchivoPesos='apiCNN/Logica/pesosOptimizados'
    model=cargarRNN(nombreArchivoModelo,nombreArchivoPesos)

    img_array = cv2.imread(direccion,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE)) 

    X = np.array(new_array).reshape(-1,IMG_SIZE,IMG_SIZE)

    X = X/255.0  
    X = X.reshape(-1,150,150,1)
    
    resultados = model.predict(X)[0]
    maxElement = np.amax(resultados)
    result = np.where(resultados == np.amax(resultados))
    index_sample_label=result[0][0]
    datos=dict()
    datos['pred']=CATEGORIAS[index_sample_label]
    datos['prod']=str(round(maxElement*100, 4))
    return datos

