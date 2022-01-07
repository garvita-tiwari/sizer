from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
import ipdb
# todo: create a base trainer
from models.network import net_modules
from models.loss import lap_loss, interp_loss, data_loss, normal_loss, verts_dist
from models.torch_smpl4garment import TorchSMPL4Garment
from data.geomtery import get_res_vert, get_vid
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes

# todo: create a base trainer


class SizerNet(object):

    def __init__(self, train_dataset, val_dataset, opt):
        self.device = opt['train']['device']

        ### garment data from experiment params
        self.garment_class = opt['experiment']['garment_class']
        self.garment_layer = opt['experiment']['garment_layer']
        self.res = opt['experiment']['resolution']
        self.gender = opt['experiment']['gender']
        self.feat = opt['experiment']['feat']
        self.num_neigh = opt['experiment']['num_neigh']

        ##create smpl layer from TailorNet, you can use any SMPL pytorch implementation(TailorNet has hres SMPL also)
        self.smpl = TorchSMPL4Garment(gender=self.gender).to(self.device)

        # load training parameters etc
        self.layer_size, _ = get_res_vert(self.garment_class, self.res, self.garment_layer)

        input_dim = self.layer_size * 3
        if self.feat == 'vn':
            input_dim = self.layer_size * 6
        output_dim = self.layer_size * 3

        mesh = load_objs_as_meshes([os.path.join(opt['data']['meta_data'],
                                                 "{}/{}_{}.obj".format(self.garment_class, self.garment_layer,
                                                                       self.res))], device=self.device)
        mesh_verts, mesh_faces = mesh.get_mesh_verts_faces(0)
        self.garment_f_torch = mesh_faces

        # geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))  todo: do we need this???
        self.d_tol = 0.002

        # create exp name based on experiment params
        self.loss_weight = {'wgt': opt['train']['wgt_wgt'], 'data': opt['train']['data_wgt'],
                            'spr_wgt': opt['train']['spr_wgt']}

        self.exp_name = '{}_{}_{}_{}_{}_{}_{}'.format(self.loss_weight['wgt'], self.loss_weight['data'],
                                                      self.loss_weight['spr_wgt'], self.garment_layer,
                                                      self.garment_class, self.feat, self.num_neigh)
        self.exp_path = '{}/{}/'.format(opt['experiment']['root_dir'], self.exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format(self.exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(self.exp_name))

        self.val_min = None
        self.train_min = None
        self.loss = opt['train']['loss_type']
        self.n_part = opt['experiment']['num_part']
        self.loss_mse = torch.nn.MSELoss()
        self.batch_size = opt['train']['batch_size']
        # weight initialiser

        ## train and val dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        ### load model and optimizer
        self.model = getattr(net_modules, opt['model']['name'])
        latent_dim = 100
        self.model = self.model(opt['model'], input_dim, latent_dim, output_dim).to(self.device)
        self.optimizer = getattr(optim, opt['train']['optimizer'])
        self.optimizer = self.optimizer(self.model.parameters(), opt['train']['optimizer_param'])

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()

    def train_step(self, batch, ep=None):

        self.model.train()
        self.optimizer.zero_grad()

        loss, loss_dict = self.compute_loss(batch, ep)
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_dict


    def compute_loss(self, batch, ep=None):
        gar_vert0 = batch.get('gar_vert0').to(self.device)
        gar_vert1 = batch.get('gar_vert1').to(self.device)
        gar_vert2 = batch.get('gar_vert2').to(self.device)

        betas0 = batch.get('betas0').to(self.device)

        pose0 = batch.get('pose0').to(self.device)
        pose1 = batch.get('pose1').to(self.device)
        pose2 = batch.get('pose2').to(self.device)

        trans0 = batch.get('trans0').to(self.device)
        trans1 = batch.get('trans1').to(self.device)
        trans2 = batch.get('trans2').to(self.device)

        size0 = batch.get('size0').to(self.device)
        size1 = batch.get('size1').to(self.device)
        size2 = batch.get('size2').to(self.device)
        inp_gar = torch.cat([gar_vert0, gar_vert0, gar_vert0, gar_vert1,gar_vert1, gar_vert1, gar_vert2, gar_vert2, gar_vert2], dim=0)
        size_inp = torch.cat([size0, size0, size0, size1,size1, size1, size2, size2, size2], dim=0)
        size_des = torch.cat([size0, size1, size2,size0, size1, size2,size0, size1, size2], dim=0)
        pose_all = torch.cat([pose0, pose1, pose2,pose0, pose1, pose2,pose0, pose1, pose2], dim=0)
        trans_all = torch.cat([trans0, trans1, trans2,trans0, trans1, trans2,trans0, trans1, trans2], dim=0)
        betas_feat = torch.cat([betas0, betas0, betas0,betas0, betas0, betas0,betas0, betas0, betas0], dim=0)
        all_dist = self.model(inp_gar, size_inp, size_des, betas_feat)
        #todo change this to displacement in unposed space , not really because of wrong correspondence
        _, pred_verts = self.smpl.forward(beta=betas_feat, theta=pose_all, trans=trans_all, garment_class='t-shirt',
                                          garment_d=all_dist)
        gt_verts = torch.cat([gar_vert0, gar_vert1, gar_vert2, gar_vert0, gar_vert1, gar_vert2, gar_vert0, gar_vert1, gar_vert2], dim=0)
        pred_mesh = Meshes(verts=pred_verts, faces=self.garment_f_torch.unsqueeze(0).repeat(self.batch_size*4, 1, 1))
        gt_mesh = Meshes(verts=gt_verts, faces=self.garment_f_torch.unsqueeze(0).repeat(self.batch_size*4, 1, 1))
        loss_data, _ = chamfer_distance(pred_verts, gt_verts)
        loss_lap = mesh_laplacian_smoothing(pred_mesh, method='uniform')
        loss_dict = {}
        loss = loss_data + 100. * loss_lap
        return loss, loss_dict

    def train_model(self, epochs, eval=True):
        loss = 0
        start = self.load_checkpoint()
        for epoch in range(start, epochs):
            sum_loss = 0
            loss_terms = {'wgt': 0, 'sdf': 0, 'disp': 0, 'diff_can': 0, 'spr_wgt': 0}
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 100 == 0:
                self.save_checkpoint(epoch)

            for batch in train_data_loader:
                loss, loss_dict = self.train_step(batch, epoch)
                for k in loss_dict.keys():
                    loss_terms[k] += self.loss_weight[k] * loss_dict[k].item()
                    print("Current loss: {} {}  ".format(k, loss_dict[k].item()))

                sum_loss += loss
            batch_loss = sum_loss / len(train_data_loader)
            print("Current batch_loss: {} {}  ".format(epoch, batch_loss))

            for k in loss_dict.keys():
                loss_terms[k] = loss_dict[k] / len(train_data_loader)
            if self.train_min is None:
                self.train_min = batch_loss
            if batch_loss < self.train_min:
                self.save_checkpoint(epoch)
                for path in glob(self.exp_path + 'train_min=*'):
                    os.remove(path)
                np.save(self.exp_path + 'train_min={}'.format(epoch), [epoch, batch_loss])

            if eval:
                val_loss = self.compute_val_loss(epoch)
                print('validation loss:   ', val_loss)
                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    self.save_checkpoint(epoch)
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, batch_loss])
                self.writer.add_scalar('val loss batch avg', val_loss, epoch)

            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', batch_loss, epoch)
            for k in loss_dict.keys():
                self.writer.add_scalar('training loss {} avg'.format(k), loss_terms[k], epoch)

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path,
                       _use_new_zipfile_serialization=False)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self, ep):

        self.model.eval()
        sum_val_loss = 0

        val_data_loader = self.val_dataset.get_loader()
        for batch in val_data_loader:
            loss, _ = self.compute_loss(batch, ep)
            sum_val_loss += loss.item()
        return sum_val_loss / len(val_data_loader)

