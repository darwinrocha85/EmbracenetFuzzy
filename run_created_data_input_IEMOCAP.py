import pickle
from os.path import join
from Datasets.IEMOCAP import DatasetIEMOCAP
from Utils.createdDataCsv import createdData
from Utils.FusionTransformer import FusionTransformer
from Utils.enumerated import type_reduction

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

createdData.creadtes_file(train_dataset, "training", type_reduction_input=type_reduction.not_reduction)
createdData.creadtes_file(test_dataset, "test", type_reduction_input=type_reduction.not_reduction)
