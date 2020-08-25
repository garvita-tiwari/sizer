import os
import sys
import tensorboardX
import argparse
import torch
from torch.utils.data import DataLoader
from psbody.mesh import Mesh
import numpy as np
import json
import pickle
import ipdb
import torch.nn as nn

from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.point import  SidedDistance

from models import network_layers
from models.loss import lap_loss, interp_loss, data_loss, normal_loss, verts_dist
from models.torch_smpl4garment import TorchSMPL4Garment


from data_utils.sizer_data import  SizerData
from data_utils.geomtery import  get_res_vert, get_vid

DATA_DIR = '/scratch/BS/pool1/garvita/parser/meta_data'

device = torch.device("cuda:0")

class Trainer(object):
    def __init__(self, params):
        self.device = device
        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']
        self.bs = params['batch_size']
        self.garment_layer = params['garment_layer']
        self.res_name = params['res']
        self.hres = True
        if self.res_name == 'lres':
            self.hres = False

        # log and backup
        LOG_DIR = params['log_dir']
        self.model_name = "EncDec_{}".format(self.res_name)

        log_name = os.path.join(self.garment_class, '{}_{}'.format(self.garment_layer, self.res_name))
        self.log_dir = os.path.join(LOG_DIR, log_name)
        if not os.path.exists(self.log_dir):
            print('making %s' % self.log_dir)
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir , "params.json"), 'w') as f:
            json.dump(params, f)

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        #load smpl data
        self.layer_size, self.smpl_size = get_res_vert(params['garment_class'],self.hres, params['garment_layer'] )

        # get active vert id
        input_dim = self.layer_size * 3
        output_dim = input_dim

        self.vert_indices = get_vid(self.garment_layer, self.garment_class,self.hres)
        self.vert_indices = torch.tensor(self.vert_indices.astype(np.long)).long().cuda()

        # dataset and dataloader
        self.train_dataset = SizerData(garment_class=self.garment_class, garment_layer=self.garment_layer,
                                       mode='train', batch_size=self.bs, res='hres', gender='male')
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                       drop_last=True if len(self.train_dataset) > self.bs else False)

        self.val_dataset = SizerData(garment_class=self.garment_class, garment_layer=self.garment_layer,
                                       mode='val', batch_size=self.bs, res='hres', gender='male')
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bs, num_workers=12, shuffle=False,
                                      drop_last=False)

        #create smpl
        self.smpl = TorchSMPL4Garment(gender=self.gender).to(device)
        self.smpl_faces_np = self.smpl.faces
        self.smpl_faces = torch.tensor(self.smpl_faces_np.astype('float32'), dtype=torch.long).cuda()

        #interpenetraion loss term
        self.body_f_np = self.smpl.faces
        self.garment_f_np = Mesh(filename=os.path.join(DATA_DIR,
                                                       'real_{}_{}_{}.obj'.format(self.garment_class, self.res_name,
                                                                                  self.garment_layer))).f

        self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)
        # models and optimizer
        latent_dim = 50
        self.model = getattr(network_layers, self.model_name)(input_size=input_dim, latent_size=latent_dim, output_size=output_dim)

        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=1e-6)
        self.out_layer = torch.nn.Softmax(dim=2)
        if params['checkpoint']:
            ckpt_path = params['checkpoint']
            print('loading ckpt from {}'.format(ckpt_path))
            state_dict = torch.load(os.path.join(ckpt_path, 'lin.pth.tar'))
            self.model.load_state_dict(state_dict)
            state_dict = torch.load(os.path.join(ckpt_path, 'optimizer.pth.tar'))
            self.optimizer.load_state_dict(state_dict)


        self.best_error = np.inf
        self.best_epoch = -1
        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir))
        self.val_min = None
        self.d_tol = 0.002


    def train(self, batch):
        gar_vert0 = batch.get('gar_vert0').to(device)
        gar_vert1 = batch.get('gar_vert1').to(device)
        gar_vert2 = batch.get('gar_vert2').to(device)

        betas0 = batch.get('betas0').to(device)

        pose0 = batch.get('pose0').to(device)
        pose1 = batch.get('pose1').to(device)
        pose2 = batch.get('pose2').to(device)

        trans0 = batch.get('trans0').to(device)
        trans1 = batch.get('trans1').to(device)
        trans2 = batch.get('trans2').to(device)

        size0 = batch.get('size0').to(device)
        size1 = batch.get('size1').to(device)
        size2 = batch.get('size2').to(device)

        self.optimizer.zero_grad()

        # encode the displacemnt
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

        pred_meshes = [tm.from_tensors(vertices=v,
                                     faces=self.garment_f_torch) for v in pred_verts]

        gt_meshes = [tm.from_tensors(vertices=v,
                                   faces=self.garment_f_torch) for v in gt_verts]


        loss_lap = lap_loss(pred_meshes, gt_meshes)
        loss_data = data_loss(self.garment_layer, pred_verts, gt_verts)

        loss = loss_data + 100.*loss_lap
        loss.backward()
        self.optimizer.step()

        return loss


    def train_epoch(self, epoch, pretrain=False, train=True):
        if train:
            self.model.train()
            loss_total = 0.0
            for batch in self.train_loader:
                train_loss = self.train(batch)
                loss_total += train_loss.item()
                self.logger.add_scalar("train/loss", train_loss.item(), self.iter_nums)
                print("Iter {}, loss: {:.8f}".format(self.iter_nums, train_loss.item()))
                self.iter_nums += 1
            self.logger.add_scalar("train_epoch/loss", loss_total/len(batch), epoch)
        else:  #validation
            self._save_ckpt(epoch)
            val_loss, val_dist = self.validation(epoch)
            print("epoch {}, loss: {:.8f} dist {:8f}".format(epoch, val_loss, val_dist))
            if self.val_min is None:
                self.val_min = val_loss

            if val_loss < self.val_min:
                self.val_min = val_loss
                with open(os.path.join(self.log_dir, 'best_epoch'), 'w') as f:
                    f.write("{:04d}".format(epoch))

            self.logger.add_scalar("val/loss", val_loss, epoch)
            self.logger.add_scalar("val/dist", val_dist, epoch)

    def validation(self, epoch):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_loader.__iter__()
                batch = self.val_data_iterator.next()
            gar_vert0 = batch.get('gar_vert0').to(device)
            gar_vert1 = batch.get('gar_vert1').to(device)
            gar_vert2 = batch.get('gar_vert2').to(device)

            betas0 = batch.get('betas0').to(device)

            pose0 = batch.get('pose0').to(device)
            pose1 = batch.get('pose1').to(device)
            pose2 = batch.get('pose2').to(device)

            trans0 = batch.get('trans0').to(device)
            trans1 = batch.get('trans1').to(device)
            trans2 = batch.get('trans2').to(device)

            size0 = batch.get('size0').to(device)
            size1 = batch.get('size1').to(device)
            size2 = batch.get('size2').to(device)

            self.optimizer.zero_grad()
            inp_gar = torch.cat(
                [gar_vert0, gar_vert0, gar_vert0, gar_vert1, gar_vert1, gar_vert1, gar_vert2, gar_vert2, gar_vert2],
                dim=0)
            size_inp = torch.cat([size0, size0, size0, size1, size1, size1, size2, size2, size2], dim=0)
            size_des = torch.cat([size0, size1, size2, size0, size1, size2, size0, size1, size2], dim=0)
            pose_all = torch.cat([pose0, pose1, pose2, pose0, pose1, pose2, pose0, pose1, pose2], dim=0)
            trans_all = torch.cat([trans0, trans1, trans2, trans0, trans1, trans2, trans0, trans1, trans2], dim=0)
            betas_feat = torch.cat([betas0, betas0, betas0, betas0, betas0, betas0, betas0, betas0, betas0], dim=0)
            gt_verts = torch.cat(
                [gar_vert0, gar_vert1, gar_vert2, gar_vert0, gar_vert1, gar_vert2, gar_vert0, gar_vert1, gar_vert2],
                dim=0)

            all_dist = self.model(inp_gar, size_inp, size_des, betas_feat)
            _, pred_verts = self.smpl.forward(beta=betas_feat, theta=pose_all, trans=trans_all, garment_class='t-shirt',
                                              garment_d=all_dist)

            sum_val_loss += data_loss(self.garment_layer, pred_verts, gt_verts).item()
        return sum_val_loss, sum_val_loss


    def _save_ckpt(self, epoch):
        save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'lin.pth.tar'))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth.tar"))



def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--garment_class', default="g5")
    parser.add_argument('--garment_layer', default="UpperClothes")
    parser.add_argument('--log_dir', default="")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--res', default="hres")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epoch', default=2000, type=int)
    parser.add_argument('--checkpoint', default="")
    parser.add_argument('--dropout', default=0.3)

    args = parser.parse_args()

    params = args.__dict__

    return params


if __name__ == '__main__':
    params = parse_argument()
    trainer = Trainer(params)
    for i in range(params['epoch']):
        print("epoch: {}".format(i))
        trainer.train_epoch(i)
        if i % 200 == 0:
            trainer.train_epoch(i, train=False)

