import torch
import os
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import pickle
import ipdb


class ParserData(Dataset):
    def __init__(self, garment_class,  split, layer_size, body_size, batch_0_only=False, gender='neutral', vc = False, vn= True):
        super(ParserData, self).__init__()

        self.garment_class = garment_class
        self.split, self.gender = split, gender
        self.batch_0_only = batch_0_only
        body_points = body_size
        num_gar_point = layer_size
        print(garment_class)
        train_data = pickle.load(
            open('/BS/garvita3/static00/SIZER/dataset/parsernet_{}_vc_vn_aug.pkl'.format(garment_class),  'rb'), encoding="latin1")
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


def one_hot(a, num_classes):
    hot = np.zeros((num_classes), np.uint8)
    hot[a] = 1
    return hot

class SizerData(Dataset):
    def __init__(self, garment_class,  split, batch_0_only=False, gender='neutral', vc = False, vn= True):
        super(SizerData, self).__init__()

        self.garment_class = garment_class
        self.split, self.gender = split, gender
        self.batch_0_only = batch_0_only
        body_points = 6890
        num_gar_point = 7702
        print(garment_class)
        train_data = pickle.load(
            open('/BS/garvita2/static00/ClothSize_data2/training_data/real_{}_size.pkl'.format(garment_class),  'rb'), encoding="latin1")
        data_name = np.array(train_data['0']['scan_name'])
        trans0 = np.array(train_data['0']['trans']).astype(np.float32)
        trans1 = np.array(train_data['1']['trans']).astype(np.float32)
        trans2 = np.array(train_data['2']['trans']).astype(np.float32)

        betas0 = np.array(train_data['0']['betas']).astype(np.float32)
        betas1 = np.array(train_data['1']['betas']).astype(np.float32)
        betas2 = np.array(train_data['2']['betas']).astype(np.float32)

        betas0 = (betas0 + betas1 + betas2)/3.

        pose0 = np.array(train_data['0']['pose']).astype(np.float32)
        pose1 = np.array(train_data['1']['pose']).astype(np.float32)
        pose2 = np.array(train_data['2']['pose']).astype(np.float32)

        scan_name0 = np.array(train_data['0']['scan_name'])
        scan_name1 = np.array(train_data['1']['scan_name'])
        scan_name2 = np.array(train_data['2']['scan_name'])

        gar_vert0 = np.array(train_data['0']['pose_upper'])[:, :num_gar_point, :].astype(np.float32)
        gar_vert1 = np.array(train_data['1']['pose_upper'])[:, :num_gar_point, :].astype(np.float32)
        gar_vert2 = np.array(train_data['2']['pose_upper'])[:, :num_gar_point, :].astype(np.float32)

        size1 = np.array(train_data['0']['size']).reshape(gar_vert0.shape[0], 1)
        size2 = np.array(train_data['1']['size']).reshape(gar_vert0.shape[0], 1)
        size3 = np.array(train_data['2']['size']).reshape(gar_vert0.shape[0], 1)

        size_label_hot1 = np.array([one_hot(i, 4) for i in size1]).astype(np.float32)
        size_label_hot2 = np.array([one_hot(i, 4) for i in size2]).astype(np.float32)
        size_label_hot3 = np.array([one_hot(i, 4) for i in size3]).astype(np.float32)


        if split == "train":
            num_train = len(data_name) - 8
            chosen_idx = range(num_train)
            gar_vert0 = gar_vert0[chosen_idx]
            gar_vert1 = gar_vert1[chosen_idx]
            gar_vert2 = gar_vert2[chosen_idx]
            betas0 = betas0[chosen_idx]

            pose0 = pose0[chosen_idx]
            pose1 = pose1[chosen_idx]
            pose2 = pose2[chosen_idx]
            trans0 = trans0[chosen_idx]
            trans1 = trans1[chosen_idx]
            trans2 = trans2[chosen_idx]
            size_label_hot1 = size_label_hot1[chosen_idx]
            size_label_hot2 = size_label_hot2[chosen_idx]
            size_label_hot3 = size_label_hot3[chosen_idx]
            # self.chosen_idx = chosen_idx

        if split == "test":
            num_train = len(data_name) - 8
            gar_vert0 = gar_vert0[num_train:]
            gar_vert1 = gar_vert1[num_train:]
            gar_vert2 = gar_vert2[num_train:]
            betas0 = betas0[num_train:]

            pose0 = pose0[num_train:]
            pose1 = pose1[num_train:]
            pose2 = pose2[num_train:]
            trans0 = trans0[num_train:]
            trans1 = trans1[num_train:]
            trans2 = trans2[num_train:]
            size_label_hot1 = size_label_hot1[num_train:]
            size_label_hot2 = size_label_hot2[num_train:]
            size_label_hot3 = size_label_hot3[num_train:]

            # self.chosen_idx = chosen_idx

        self.gar_vert0 = torch.from_numpy(gar_vert0)
        self.gar_vert1 = torch.from_numpy(gar_vert1)
        self.gar_vert2 = torch.from_numpy(gar_vert2)
        self.betas0 = torch.from_numpy(betas0)

        self.pose0 = torch.from_numpy(pose0)
        self.pose1 = torch.from_numpy(pose1)
        self.pose2 = torch.from_numpy(pose2)
        self.trans0 = torch.from_numpy(trans0)
        self.trans1 = torch.from_numpy(trans1)
        self.trans2 = torch.from_numpy(trans2)
        self.size_label_hot1 = torch.from_numpy(size_label_hot1)
        self.size_label_hot2 = torch.from_numpy(size_label_hot2)
        self.size_label_hot3 = torch.from_numpy(size_label_hot3)

        self.smoothing = None
        self.smpl = None

    def __len__(self):
        return self.gar_vert0.shape[0]

    def __getitem__(self, item):
        # print(item, self.style_idx, self.shape_idx, self.split)

        gar_vert0, gar_vert1, gar_vert2, betas0, pose0, pose1, pose2, trans0, trans1, \
        trans2, size0, size1, size2 = self.gar_vert0[item], self.gar_vert1[item], self.gar_vert2[item], self.betas0[item], \
                                         self.pose0[item], self.pose1[item], self.pose2[item], \
                                      self.trans0[item], self.trans1[item], self.trans2[item], self.size_label_hot1[item], \
                                      self.size_label_hot2[item], self.size_label_hot3[item]

        return gar_vert0, gar_vert1, gar_vert2, betas0, pose0, pose1, pose2, trans0, trans1, \
        trans2, size0, size1, size2, item