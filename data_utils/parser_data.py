#from __future__ import division
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import ipdb

from data_utils.geomtery import  get_res_vert, get_vid

DATA_DIR = '/scratch/BS/pool1/garvita/parser/meta_data'
class ParserData(Dataset):


    def __init__(self,  garment_class, garment_layer, mode, batch_size, res='hres', gender='male', feat=None, num_workers = 12,**kwargs):
        super(ParserData, self).__init__()

        self.garment_class = garment_class
        self.garment_layer = garment_layer
        self.mode, self.gender = mode, gender
        self.res = res

        self.layer_size, self.body_size = get_res_vert(garment_class ,self.res,garment_layer)

        #load data from file
        train_data = pickle.load(
            open(os.path.join(DATA_DIR, 'parsernet_{}_vc_vn_aug.pkl'.format(garment_class)), 'rb'),
            encoding="latin1")

        self.reg_mesh = np.array(train_data['reg']).astype(np.float32)[:, :self.body_size, :]
        self.out_gar = np.array(train_data[garment_layer]).astype(np.float32)[:, :self.layer_size, :]
        self.data_name = np.array(train_data["name"])
        self.betas = np.array(train_data["betas"]).astype(np.float32)
        self.pose = np.array(train_data["pose"]).astype(np.float32)
        self.trans = np.array(train_data["trans"]).astype(np.float32)
        all_vc = np.array(train_data['vc'])[:, :self.body_size, :].astype(np.float32)
        all_vn = np.array(train_data['vn'])[:, :self.body_size, :].astype(np.float32)

        if feat is None:
            self.input2net = self.self.body_size
        elif feat =='vn':
            self.input2net = np.concatenate((self.reg_mesh, all_vn), axis=2)
        elif feat == 'vn_vc':
            self.input2net = np.concatenate((self.reg_mesh, all_vc, all_vn), axis=2)


        #load train,val, test split
        split_file = os.path.join(DATA_DIR, '{}_split.pkl'.format(self.garment_class))
        self.split_idx = pickle.load(open(split_file, 'rb'),
            encoding="latin1")[mode]

        self.batch_size = batch_size
        self.num_workers = num_workers


    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, idx):


        return {'inp': np.array(self.input2net[idx], dtype=np.float32),
                'gt_verts': np.array(self.out_gar[idx], dtype=np.float32),
                'betas':np.array(self.betas[idx], dtype=np.float32),
                'trans': np.array(self.trans[idx], dtype=np.float32),
                'pose': np.array(self.pose[idx], dtype=np.float32)}