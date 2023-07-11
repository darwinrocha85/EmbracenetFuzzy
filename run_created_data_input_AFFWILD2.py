import pickle
from os.path import join
from Datasets.AFFWILD2 import AFFWILD2
from Utils.createdDataCsv import createdData
from Utils.FusionTransformer import FusionTransformer
from Utils.enumerated import type_reduction
import pandas as pd
import numpy as np

classes = {'Neutral':0, 'Anger':1, 'Disgust':2, 'Fear':3, 'Happiness':4, 'Sadness':5, 'Surprise':6,   'other': 7, 'number':8}

face_data = join('Data', 'train_image_feats.pkl')
audio_data = join('Data', 'train_audio_feats.pkl')
text_data = join('Data', 'train_text_feats.pkl')
label_data = join('Data', 'trainembrace.csv')

with open(face_data, 'rb') as dic:
    face_data = pickle.load(dic)
with open(audio_data, 'rb') as dic:
    audi_data = pickle.load(dic)
with open(text_data, 'rb') as dic:
    text_data = pickle.load(dic)


df_label = pd.read_csv(label_data, delimiter='|')
labels ={}
for index, row in df_label.iterrows():
    labels[index]= index, np.array(row['ilabel'])



print("labels", len(labels))
print("face_data", len(face_data))
print("audi_data", len(audi_data))
print("text_data", len(text_data))


face_data_test = join('Data', 'valid_image_feats.pkl')
audio_data_test = join('Data', 'valid_audio_feats.pkl')
text_data_test = join('Data', 'valid_text_feats.pkl')
label_data_test = join('Data', 'validembrace.csv')

with open(face_data_test, 'rb') as dic:
    face_data_test = pickle.load(dic)
with open(audio_data_test, 'rb') as dic:
    audi_data_test = pickle.load(dic)
with open(text_data_test, 'rb') as dic:
    text_data_test = pickle.load(dic)


df_label = pd.read_csv(label_data_test, delimiter='|')
labels_test ={}
for index, row in df_label.iterrows():
    labels_test[index]= index, np.array(row['ilabel'])


print("labels", len(labels_test))
print("face_data", len(face_data_test))
print("audi_data", len(audi_data_test))
print("text_data", len(text_data_test))

train_dataset = AFFWILD2(classes,labels, face_data, audi_data,
                               text_data, 'average',
                               transform=FusionTransformer(''))

test_dataset = AFFWILD2(classes,labels_test, face_data_test, audi_data_test,
                               text_data_test, 'average',
                               transform=FusionTransformer(''))


createdData.creadtes_file(train_dataset, "training", type_reduction_input=type_reduction.not_reduction, datebase_input="affwild2")
createdData.creadtes_file(test_dataset, "test", type_reduction_input=type_reduction.not_reduction, datebase_input="affwild2")
