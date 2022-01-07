"""source: https://github.com/jchibane/ndf"""
from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import ipdb

from data.geomtery import get_res_vert, get_vid

class ParserData(Dataset):


    def __init__(self, mode, data_path, split_file, batch_size, num_workers=12, garment_data=None ,**kwargs):


        self.garment_class = garment_data['garment_class']
        self.garment_layer = garment_data['garment_layer']
        self.res = garment_data['resolution']
        self.gender = garment_data['gender']
        self.feat = garment_data['feat']
        self.num_neigh = garment_data['num_neigh']

        self.layer_size, self.smpl_size = get_res_vert(self.garment_class,self.res, self.garment_layer )
        train_data = np.load(data_path)
        #split = np.load(split_file)[mode]
        #train_data = train_data[split]
        self.reg_mesh = np.array(train_data['reg']).astype(np.float32)[:, :self.smpl_size, :]
        if self.garment_layer == "Body":
            self.out_gar = np.array(train_data["reg"]).astype(np.float32)[:, :self.layer_size, :]
        else:
            self.out_gar = np.array(train_data[self.garment_layer]).astype(np.float32)[:, :self.layer_size, :]
        self.data_name = np.array(train_data["name"])
        self.betas = np.array(train_data["betas"]).astype(np.float32)[:, :10]
        self.pose = np.array(train_data["pose"]).astype(np.float32)
        self.trans = np.array(train_data["trans"]).astype(np.float32)
        self.input2net = self.reg_mesh

        if self.feat == 'vn':
            all_vn = np.array(train_data['vn'])[:, :self.smpl_size, :].astype(np.float32)
            self.input2net = np.concatenate((self.reg_mesh, all_vn), axis=2)
        elif self.feat == 'vn_vc':
            all_vc = np.array(train_data['vc'])[:, :self.smpl_size, :].astype(np.float32)
            self.input2net = np.concatenate((self.reg_mesh, all_vc, all_vn), axis=2)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, idx):

        return {'inp': np.array(self.input2net[idx], dtype=np.float32),
                'gt_verts': np.array(self.out_gar[idx], dtype=np.float32),
                'betas': np.array(self.betas[idx], dtype=np.float32),
                'trans': np.array(self.trans[idx], dtype=np.float32),
                'pose': np.array(self.pose[idx], dtype=np.float32)}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

def one_hot(a, num_classes):
    hot = np.zeros((num_classes), np.uint8)
    hot[a] = 1
    return hot
class SizerData(Dataset):


    def __init__(self, mode, data_path, split_file, batch_size, num_workers=12, garment_data=None ,**kwargs):


        self.garment_class = garment_data['garment_class']
        self.garment_layer = garment_data['garment_layer']
        self.res = garment_data['resolution']
        self.gender = garment_data['gender']
        self.feat = garment_data['feat']

        self.layer_size, self.smpl_size = get_res_vert(self.garment_class,self.res, self.garment_layer )
        path_tmp = '/BS/RVH_3dscan/static00/training_sizer/ClothSize_data2/training_data/real_g5_size.pkl'
        # import pickle as pickle
        # train_data = pickle.load(open(path_tmp, 'rb'), encoding="latin1")
        # ipdb.set_trace()
        train_data = np.load(data_path)
        # np.savez('/BS/RVH_3dscan/static00/SIZER/dataset/g5_size.npz',trans0=train_data['trans'] ,trans1=train_data['trans'] ,trans2=train_data['trans'],betas0=train_data['betas'] ,betas1=train_data['betas'] ,betas2=train_data['betas'],pose0=train_data['pose'] ,pose1=train_data['pose'] ,pose2=train_data['pose'], pose_upper0=train_data['pose_upper'] ,pose_upper1=train_data['pose_upper'] ,pose_upper2=train_data['pose_upper'], size0=train_data['size'], size1=train_data['size'], size2=train_data['size'], scan_name0=train_data['scan_name'], scan_name1=train_data['scan_name'], scan_name2=train_data['scan_name'])
        # tmp = np.load('/BS/RVH_3dscan/static00/SIZER/dataset/g5_size.npz', allow_pickle=True)
        # tmp['size0']['trans']
        self.data_name = np.array(train_data['scan_name0'])
        self.trans0 = np.array(train_data['trans0']).astype(np.float32)
        self.trans1 = np.array(train_data['trans1']).astype(np.float32)
        self.trans2 = np.array(train_data['trans2']).astype(np.float32)

        betas0 = np.array(train_data['betas0']).astype(np.float32)
        betas1 = np.array(train_data['betas1']).astype(np.float32)
        betas2 = np.array(train_data['betas2']).astype(np.float32)

        self.betas0 = (betas0 + betas1 + betas2) / 3.

        self.pose0 = np.array(train_data['pose0']).astype(np.float32)
        self.pose1 = np.array(train_data['pose1']).astype(np.float32)
        self.pose2 = np.array(train_data['pose2']).astype(np.float32)

        scan_name0 = np.array(train_data['scan_name0'])
        scan_name1 = np.array(train_data['scan_name1'])
        scan_name2 = np.array(train_data['scan_name2'])

        self.gar_vert0 = np.array(train_data['pose_upper0'])[:, :self.layer_size, :].astype(np.float32)
        self.gar_vert1 = np.array(train_data['pose_upper1'])[:, :self.layer_size, :].astype(np.float32)
        self.gar_vert2 = np.array(train_data['pose_upper2'])[:, :self.layer_size, :].astype(np.float32)

        size1 = np.array(train_data['size0']).reshape(self.gar_vert0.shape[0], 1)
        size2 = np.array(train_data['size1']).reshape(self.gar_vert0.shape[0], 1)
        size3 = np.array(train_data['size2']).reshape(self.gar_vert0.shape[0], 1)

        self.size_label_hot1 = np.array([one_hot(i, 4) for i in size1]).astype(np.float32)
        self.size_label_hot2 = np.array([one_hot(i, 4) for i in size2]).astype(np.float32)
        self.size_label_hot3 = np.array([one_hot(i, 4) for i in size3]).astype(np.float32)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return len(self.data_name)

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
    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
