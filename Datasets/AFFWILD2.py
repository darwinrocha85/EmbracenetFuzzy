from xml.etree.ElementInclude import include
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


#Esta clase es la que se utiliza para trabajar el dataset de IEMOCAP. La saqué del código de JuanPablo Heredia (juan1t0 github)
class AFFWILD2(Dataset):
    def __init__(self, classes, labels, FaceR, AudioR, TextR, method='avg', mode='train', transform=None, omit_modality=None):
        super(AFFWILD2, self).__init__()
        self.Data = {}
        self.DataKeys = []
        self.Face = True
        self.Audio = True
        self.Text = True
        self.Transform = transform
        self.Classes = classes
        self.Mode = mode
        self.Method = method
        self.omit_modality = omit_modality
        self.loadData(FaceR, AudioR, TextR, labels)

    def loadData(self, face_results, audio_results, text_results, label_results):
        LFks = list(face_results.keys())
        LAks = list(audio_results.keys())
        LTks = list(text_results.keys())
        LLks = list(label_results.keys())

        for k in LFks:

            if k in LFks:
                FD = self.convert(face_results[k])
            else:
                FD = None
            if k in LTks:
                TD = text_results[k]
            else:
                TD = None
            if k in LAks:
                AD = self.convert(audio_results[k])
            else:
                AD = None
            if k in LLks:
                LL = [0,0,0,0,0,0,0,0]
                LL[label_results[k][1]] = 1.0
            else:
                LL = np.full(7, -1)


            self.Data[k] = (FD, AD, TD, LL)
        self.DataKeys = list(self.Data.keys())

    def convert(self, facial_data):
        if self.Method[0] == 'a':
            facedata = np.mean(np.stack(facial_data), axis=0)
            facedata = np.expand_dims(facedata, 0)
            facedata = F.softmax(torch.from_numpy(facedata),dim=-1)
            facedata = facedata.numpy()
        elif self.Method[0] == 'v':
            mv = np.bincount(np.argmax(np.stack(facial_data),axis=1)).argmax()
            facedata = np.zeros(facedata[0].shape)
            facedata[mv] = 1.0
            # facedata = torch.from_numpy(facedata)
        elif self.Method[0] == 'c':
            # facedata = torch.from_numpy(np.concatenate(facial_data))
            facedata = np.concatenate(facial_data)

        return facedata

    def __len__(self):
        return len(self.DataKeys)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        avs = np.ones(3)
        label = self.Data[self.DataKeys[idx]][3]
        face = self.Data[self.DataKeys[idx]][0]
        if face is None or self.omit_modality == 'face':
            avs[0] = 0.
            print('face***********')
            face = np.full(self.Classes['number'], 1/self.Classes['number'])

        audio = self.Data[self.DataKeys[idx]][1]
        if audio is None or self.omit_modality == 'audio':
            avs[1] = 0.
            print('face***********')
            audio = np.full(self.Classes['number'], 1/self.Classes['number'])

        text = self.Data[self.DataKeys[idx]][2]
        if text is None or self.omit_modality == 'text':
            avs[2] = 0.
            print('face***********')
            text = np.full(self.Classes['number'], 1/self.Classes['number'])



        sample = {'face': face[0],
                    'audio':audio[0],
                    'text': text,
                    'label':label,
                    'availabilities':avs,
                    'name': self.DataKeys[idx]}

        return sample

    def make_shuffle(self):
        random.shuffle(self.DataKeys)
