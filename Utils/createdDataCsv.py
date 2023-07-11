import torch
import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from Utils.enumerated import type_reduction
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class createdData():
    @staticmethod
    def creadtes_file(dataset_input, type_data_input="training", type_reduction_input=type_reduction.not_reduction, datebase_input="affwild2"):
        array_data = []
        array_data_label = []

        for bacth in dataset_input:
            if datebase_input == "affwild2":
                face = bacth['face']
                audio = bacth['audio']
                text = bacth['text']
                label = bacth['label'].index(max(bacth['label']))
                label = np.ravel(label)
                input = np.hstack((face, audio, text))
                array_data.append(input)
                array_data_label.append(label)
            else:
                face = bacth['face'].numpy()
                audio = bacth['audio'].numpy()
                text = bacth['text'].numpy()
                label = torch.argmax(bacth['label'], dim=-1).numpy()
                label = np.ravel(label)
                input = np.concatenate((face, audio, text), axis=0)
                array_data.append(input)
                array_data_label.append(label)



        if type_data_input == "test":

            # Crear un DataFrame de pandas a partir del arreglo
            df = pd.DataFrame(array_data)

            # Escribir el DataFrame en un archivo CSV
            df.to_csv('test_archivo_1.csv', index=False, header=False)

            # Crear un DataFrame de pandas a partir del arreglo
            df = pd.DataFrame(array_data_label)

            # Escribir el DataFrame en un archivo CSV
            df.to_csv('test_archivo_2.csv', index=False, header=False)

        else:
            # Crear un DataFrame de pandas a partir del arreglo
            df = pd.DataFrame(array_data)

            # Escribir el DataFrame en un archivo CSV
            df.to_csv('train_archivo_1.csv', index=False, header=False)

            # Crear un DataFrame de pandas a partir del arreglo
            df = pd.DataFrame(array_data_label)

            # Escribir el DataFrame en un archivo CSV
            df.to_csv('train_archivo_2.csv', index=False, header=False)
