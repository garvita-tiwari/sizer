#from __future__ import division
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import ipdb

from data_utils.geomtery import get_vertex_normals, nearest_map, get_res_vert, get_vid

def one_hot(a, num_classes):
    hot = np.zeros((num_classes), np.uint8)
    hot[a] = 1
    return hot


class SizerData(Dataset):


    def __init__(self,  garment_class, garment_layer, mode, batch_size, res='hres', gender='male', feat=None, num_workers = 12,**kwargs):
        super(SizerData, self).__init__()

        self.garment_class = garment_class
        self.garment_layer = garment_layer
        self.mode, self.gender = mode, gender
        self.res = res

        self.layer_size, self.body_size = get_res_vert(garment_class ,self.res,garment_layer)

        #load data from file
        train_data = pickle.load(
            open('/BS/garvita2/static00/ClothSize_data2/training_data/real_{}_size.pkl'.format(garment_class), 'rb'),
            encoding="latin1")
        ipdb.set_trace()
        data_name = np.array(train_data['0']['scan_name'])
        self.trans0 = np.array(train_data['0']['trans']).astype(np.float32)
        self.trans1 = np.array(train_data['1']['trans']).astype(np.float32)
        self.trans2 = np.array(train_data['2']['trans']).astype(np.float32)

        betas0 = np.array(train_data['0']['betas']).astype(np.float32)
        betas1 = np.array(train_data['1']['betas']).astype(np.float32)
        betas2 = np.array(train_data['2']['betas']).astype(np.float32)

        self.betas0 = (betas0 + betas1 + betas2) / 3.

        self.pose0 = np.array(train_data['0']['pose']).astype(np.float32)
        self.pose1 = np.array(train_data['1']['pose']).astype(np.float32)
        self.pose2 = np.array(train_data['2']['pose']).astype(np.float32)

        scan_name0 = np.array(train_data['0']['scan_name'])
        scan_name1 = np.array(train_data['1']['scan_name'])
        scan_name2 = np.array(train_data['2']['scan_name'])

        self.gar_vert0 = np.array(train_data['0']['pose_upper'])[:, :self.layer_size, :].astype(np.float32)
        self.gar_vert1 = np.array(train_data['1']['pose_upper'])[:, :self.layer_size, :].astype(np.float32)
        self.gar_vert2 = np.array(train_data['2']['pose_upper'])[:, :self.layer_size, :].astype(np.float32)

        self.size1 = np.array(train_data['0']['size']).reshape(gar_vert0.shape[0], 1)
        self.size2 = np.array(train_data['1']['size']).reshape(gar_vert0.shape[0], 1)
        self.size3 = np.array(train_data['2']['size']).reshape(gar_vert0.shape[0], 1)

        self.size_label_hot1 = np.array([one_hot(i, 4) for i in size1]).astype(np.float32)
        self.size_label_hot2 = np.array([one_hot(i, 4) for i in size2]).astype(np.float32)
        self.size_label_hot3 = np.array([one_hot(i, 4) for i in size3]).astype(np.float32)

        #load train,val, test split
        split_file = '/BS/garvita/static00/training_dataset/{}_size_split.pkl'.format(self.garment_class)
        self.split_idx = pickle.load(open(split_file, 'rb'),
            encoding="latin1")[mode]

        self.batch_size = batch_size
        self.num_workers = num_workers


    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, idx):


        return {'gar_vert0': np.array(self.gar_vert0[idx], dtype=np.float32),
                'gar_vert1': np.array(self.gar_vert1[idx], dtype=np.float32),
                'gar_vert2': np.array(self.gar_vert2[idx], dtype=np.float32),
                'betas0':np.array(self.betas0[idx], dtype=np.float32),
                'trans0': np.array(self.trans0[idx], dtype=np.float32),
                'trans1': np.array(self.trans1[idx], dtype=np.float32),
                'trans2': np.array(self.trans2[idx], dtype=np.float32),
                'pose0': np.array(self.pose0[idx], dtype=np.float32),
                'pose1': np.array(self.pose1[idx], dtype=np.float32),
                'pose2': np.array(self.pose2[idx], dtype=np.float32),
                'size0': np.array(self.size_label_hot1[idx]),
                'size1': np.array(self.size_label_hot2[idx]),
                'size2': np.array(self.size_label_hot3[idx])}