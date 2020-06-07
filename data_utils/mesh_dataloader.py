import torch
import os
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import pickle
import ipdb


class ParserData(Dataset):
    def __init__(self, garment_class,  split, batch_0_only=False, gender='neutral', vc = False, vn= True):
        super(ParserData, self).__init__()

        self.garment_class = garment_class
        self.split, self.gender = split, gender
        self.batch_0_only = batch_0_only
        body_points = 6890
        num_gar_point = 1924
        print(garment_class)
        train_data = pickle.load(
            open('/BS/garvita3/static00/SIZER/dataset/parsernet_{}_vc_vn_aug.pkl'.format(garment_class.split('_')[1]),  'rb'), encoding="latin1")
        reg_mesh = np.array(train_data['reg']).astype(np.float32)[:, :body_points, :]
        out_gar = np.array(train_data['UpperClothes']).astype(np.float32)[:, :num_gar_point, :]
        data_name = np.array(train_data["name"])
        betas = np.array(train_data["betas"]).astype(np.float32)
        pose = np.array(train_data["pose"]).astype(np.float32)
        trans = np.array(train_data["trans"]).astype(np.float32)
        if vc:
            all_vc = np.array(train_data['vc'])[:, :body_points, :].astype(np.float32)
        if vn:
            all_vn = np.array(train_data['vn'])[:, :body_points, :].astype(np.float32)

        if vc and vn:
            input2net = np.concatenate((reg_mesh, all_vc, all_vn), axis=2)
        elif vc:
            input2net = np.concatenate((reg_mesh, all_vc), axis=2)
        elif vn:
            input2net = np.concatenate((reg_mesh, all_vn), axis=2)
        else:
            input2net = reg_mesh


        if split == "train":
            num_train = len(data_name) - 8
            chosen_idx = range(num_train)
            input2net = input2net[chosen_idx]
            out_gar = out_gar[chosen_idx]
            betas = betas[chosen_idx]
            pose = pose[chosen_idx]
            trans = trans[chosen_idx]
            # self.chosen_idx = chosen_idx

        if split == "test":
            num_train = len(data_name) - 8
            input2net = input2net[num_train:]
            out_gar = out_gar[num_train:]
            betas = betas[num_train:]
            pose = pose[num_train:]
            trans = trans[num_train:]
            # self.chosen_idx = chosen_idx

        self.input2net = torch.from_numpy(input2net)
        self.out_gar = torch.from_numpy(out_gar)
        self.betas = torch.from_numpy(betas)
        self.pose = torch.from_numpy(pose)
        self.trans = torch.from_numpy(trans)



        self.smoothing = None
        self.smpl = None

    def __len__(self):
        return self.input2net.shape[0]

    def __getitem__(self, item):
        # print(item, self.style_idx, self.shape_idx, self.split)

        input2net, out_gar, betas, pose, trans = self.input2net[item], self.out_gar[item], self.betas[item], self.pose[item], self.trans[item]

        return input2net, out_gar, betas, pose, trans, item