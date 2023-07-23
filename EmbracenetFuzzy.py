
import numpy as np
import csv
from Models.Embracenet import Wrapper
import torch
import torch.nn as nn
from torch.optim import Adam
import pickle
from os.path import join
from Datasets.IEMOCAP import DatasetIEMOCAP
from Utils.createdDataCsv import createdData
from Utils.FusionTransformer import FusionTransformer
from Utils.enumerated import type_reduction
from Utils.dataloaders import my_collate
from torch.utils.data import DataLoader
from Utils.training_functions import train_embracenet
import pickle
import sys
import os
import re
from os.path import join

mat_path = ""
learning_rate = 0.0001

classes = {'exc':0, 'neu':1, 'sad':2, 'hap':0, 'ang':3, 'number': 4}

face_data = join('Data', 'facepreds_allsess_v4_55A.p')
audio_data = join('Data', 'audiopreds_allsess_4e_75A.p')
text_data = join('Data', 'text_preds_4e_6-A.p')

BatchSize = 32

with open(face_data, 'rb') as dic:
    face_data = pickle.load(dic)
with open(audio_data, 'rb') as dic:
    audi_data = pickle.load(dic)
with open(text_data, 'rb') as dic:
    text_data = pickle.load(dic)

train_dataset = DatasetIEMOCAP(classes, face_data, audi_data,
                               text_data, 'average',
                               transform=FusionTransformer(''))
test_dataset = DatasetIEMOCAP(classes, face_data, audi_data,
                              text_data, 'average', mode = 'test',
                              transform=FusionTransformer(''))

train_dataloader = DataLoader(train_dataset,
                              batch_size=BatchSize, collate_fn=my_collate)
test_dataloader = DataLoader(test_dataset,
                             batch_size=BatchSize, collate_fn=my_collate)

device = torch.device('cpu')
loss_function = nn.CrossEntropyLoss()
model = Wrapper(
        name="",
        device=device,
        n_classes=4,
        size_list=[4,4,4],
        embracesize=16
    )

optimizer = Adam(model.parameters())#SGD(model.parameters(), lr=learning_rate)

train_embracenet(model, learning_rate, train_dataloader, 500, loss_function, optimizer, "", test_dataloader)

if learning_rate != 0:
    base_name = f'model_{"model_name"}_lr_{str(learning_rate).replace(".", "")}'
else:
    base_name = f'model_{"model_name"}_adam'

results_path = join('Results', "method")
torch.save(model.state_dict(), join('Saved Models', f'{base_name}.pth'))
os.system(f'Rscript plots.R {results_path} {base_name}')
os.system(f'rm Rplots.pdf')
