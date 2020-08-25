#from __future__ import division
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import ipdb

from data_utils.geomtery import get_vertex_normals, nearest_map, get_res_vert, get_vid

class ParserData(Dataset):


    def __init__(self,  garment_class, garment_layer, mode, batch_size, res='hres', gender='male', vc = False, vn= True, num_workers = 12,**kwargs):
        super(ParserData, self).__init__()

        self.garment_class = garment_class
        self.garment_layer = garment_layer
        self.mode, self.gender = mode, gender
        self.res = res

        self.layer_size, self.body_size = get_res_vert(garment_class ,self.res,garment_layer)

        #load data from file
        train_data = pickle.load(
            open('/BS/garvita3/static00/SIZER/dataset/parsernet_{}_vc_vn_aug.pkl'.format(garment_class), 'rb'),
            encoding="latin1")

        self.reg_mesh = np.array(train_data['reg']).astype(np.float32)[:, :self.body_size, :]
        self.out_gar = np.array(train_data[garment_layer]).astype(np.float32)[:, :self.layer_size, :]
        self.data_name = np.array(train_data["name"])
        self.betas = np.array(train_data["betas"]).astype(np.float32)
        self.pose = np.array(train_data["pose"]).astype(np.float32)
        self.trans = np.array(train_data["trans"]).astype(np.float32)
        if vc:
            all_vc = np.array(train_data['vc'])[:, :self.body_size, :].astype(np.float32)
        if vn:
            all_vn = np.array(train_data['vn'])[:, :self.body_size, :].astype(np.float32)

        if vc and vn:
            self.input2net = np.concatenate((self.reg_mesh, all_vc, all_vn), axis=2)
        elif vc:
            self.input2net = np.concatenate((self.reg_mesh, all_vc), axis=2)
        elif vn:
            self.input2net = np.concatenate((self.reg_mesh, all_vn), axis=2)
        else:
            self.input2net = self.self.body_size

        #load train,val, test split
        split_file = '/BS/garvita/static00/training_dataset/{}_split.pkl'.format(self.garment_class)
        self.split_idx = pickle.load(open(split_file, 'rb'),
            encoding="latin1")[mode]

        self.batch_size = batch_size
        self.num_workers = num_workers

        # self.sample_distribution = np.array(sample_distribution)
        # self.sample_sigmas = np.array(sample_sigmas)
        #
        # assert np.sum(self.sample_distribution) == 1
        # assert np.any(self.sample_distribution < 0) == False
        # assert len(self.sample_distribution) == len(self.sample_sigmas)
        #
        # self.path = data_path
        # self.split = np.load(split_file)
        # self.data = self.split[mode]

        #
        # self.res = res
        #
        # self.num_sample_points = num_sample_points
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.voxelized_pointcloud = voxelized_pointcloud
        # self.pointcloud_samples = pointcloud_samples
        #
        # # compute number of samples per sampling method
        # self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)
        #


    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, idx):


        return {'inp': np.array(self.input2net[idx], dtype=np.float32),
                'gt_verts': np.array(self.out_gar[idx], dtype=np.float32),
                'betas':np.array(self.betas[idx], dtype=np.float32),
                'trans': np.array(self.trans[idx], dtype=np.float32),
                'pose': np.array(self.pose[idx], dtype=np.float32)}


    # def get_loader(self, shuffle =True):
    #
    #     return torch.utils.data.DataLoader(
    #             self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
    #             worker_init_fn=self.worker_init_fn)
    #
    # def worker_init_fn(self, worker_id):
    #     random_data = os.urandom(4)
    #     base_seed = int.from_bytes(random_data, byteorder="big")
    #     np.random.seed(base_seed + worker_id)