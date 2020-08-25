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

from data_utils.parser_data import ParserData
from data_utils.geomtery import get_res_vert, get_vid

device = torch.device("cuda:0")

DATA_DIR = '/scratch/BS/pool1/garvita/parser/meta_data'

class Trainer(object):
    def __init__(self, params):
        self.device= device
        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']
        self.bs = params['batch_size']
        self.garment_layer = params['garment_layer']
        self.res_name = params['res']
        self.num_neigh = params['num_neigh']
        self.feat = params['feat']
        self.hres = True
        if self.res_name == 'lres':
            self.hres = False
        # log
        LOG_DIR = params['log_dir']

        self.model_name = 'FC_correspondence_{}'.format(self.res_name)
        self.note = "FC_corr_{}_{}_{}".format(self.garment_class, self.garment_layer, self.res_name)
        log_name = os.path.join(self.garment_class, '{}_{}_{}_{}'.format(self.garment_layer, self.feat, self.num_neigh, self.res_name))

        self.log_dir = os.path.join(LOG_DIR, log_name)
        if not os.path.exists(self.log_dir):
            print('making %s' % self.log_dir)
            os.makedirs(self.log_dir)

        with open(os.path.join(self.log_dir , "params.json"), 'w') as f:
            json.dump(params, f)

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        #load smpl and garment data

        self.layer_size, self.smpl_size = get_res_vert(params['garment_class'],self.hres, params['garment_layer'] )
        if self.garment_layer == 'Body':
            self.layer_size = 4448
        # get active vert id
        input_dim = self.smpl_size * 3
        if self.feat == 'vn':
            input_dim = self.smpl_size * 6
        output_dim = self.layer_size *  self.num_neigh

        layer_neigh = np.array(np.load(os.path.join(DATA_DIR, "real_{}_neighborheuristics_{}_{}_{}_gar_order2.npy".format(self.garment_class, self.res_name, self.garment_layer, self.num_neigh))))
        self.layer_neigh = torch.from_numpy(layer_neigh).cuda()

        #separate for body layer
        body_vert = range(self.smpl_size)
        vert_id_upper = get_vid('UpperClothes', self.garment_class, False)
        vert_id_lower = get_vid('Pants', self.garment_class, False)
        body_vert2 = [i for i in body_vert if i not in vert_id_upper]
        body_vert2 = [i for i in body_vert2 if i not in vert_id_lower]
        self.body_vert = body_vert2

        all_neighbors = np.array([[vid] for k in layer_neigh for vid in k])
        self.neigh_id2 = all_neighbors
        if self.garment_layer == 'Body':
            self.idx2 = torch.from_numpy(self.neigh_id2).view(len(self.body_vert), self.num_neigh).cuda()
        else:
            self.idx2 = torch.from_numpy(self.neigh_id2).view(self.layer_size, self.num_neigh).cuda()

        #get vert indixed of layer
        self.vert_indices = get_vid(self.garment_layer, self.garment_class,False)
        self.vert_indices = torch.tensor(self.vert_indices.astype(np.long)).long().cuda()

        # dataset and dataloader
        self.train_dataset = ParserData(garment_class=self.garment_class, garment_layer=self.garment_layer,
                                        mode='train', batch_size= self.bs, res=self.res_name, gender=self.gender, feat=self.feat)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                       drop_last=True if len(self.train_dataset) > self.bs else False)

        self.val_dataset = ParserData(garment_class=self.garment_class, garment_layer=self.garment_layer,
                                        mode='val', batch_size= self.bs,  res=self.res_name, gender=self.gender, feat=self.feat)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                      drop_last=False)

        #create smpl
        self.smpl = TorchSMPL4Garment(gender=self.gender).to(device)
        self.smpl_faces_np = self.smpl.faces
        self.smpl_faces = torch.tensor(self.smpl_faces_np.astype('float32'), dtype=torch.long).cuda()

        if self.garment_layer == 'Body':
            self.garment_f_np = self.body_f_np
            self.garment_f_torch = self.smpl_faces
        else:
            self.garment_f_np = Mesh(filename=os.path.join(DATA_DIR,'real_{}_{}_{}.obj'.format(self.garment_class,self.res_name, self.garment_layer))).f
            self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)

        self.num_faces = len(self.garment_f_np)


        self.model = getattr(network_layers, self.model_name)(input_size=input_dim, output_size=output_dim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=1e-6)
        self.out_layer = torch.nn.Softmax(dim=2)
        if params['checkpoint']:
            ckpt_path = params['checkpoint']
            print('loading ckpt from {}'.format(ckpt_path))
            state_dict = torch.load(os.path.join(ckpt_path, 'lin.pth.tar'))
            self.model.load_state_dict(state_dict)
            state_dict = torch.load(os.path.join(ckpt_path, 'optimizer.pth.tar'))
            self.optimizer.load_state_dict(state_dict)

        geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))
        self.geo_weights = torch.tensor(geo_weights[body_vert2].astype(np.float32)).cuda()
        self.best_error = np.inf
        self.best_epoch = -1
        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir))
        self.val_min = None
        self.d_tol = 0.002

        self.sideddistance =  SidedDistance()
        self.relu = nn.ReLU()
        #weight initialiser
        vert_id = self.vert_indices.cpu().numpy()
        init_weights = torch.from_numpy(np.array([layer_neigh[i] ==vert_id[i] for i in range(self.layer_size)]).astype('int64'))
        self.init_weight = torch.stack([init_weights for _ in range(self.bs)]).cuda()


    def train(self, batch, pretrain=False,):
        inp = batch.get('inp').to(self.device)
        gt_verts = batch.get('gt_verts').to(self.device)
        betas = batch.get('betas').to(self.device)
        pose = batch.get('pose').to(self.device)
        trans = batch.get('trans').to(self.device)

        self.optimizer.zero_grad()
        weights_from_net = self.model(inp).view(self.bs, self.layer_size, self.num_neigh)
        weights_from_net = self.out_layer(weights_from_net)

        if pretrain:
            loss = (weights_from_net - self.init_weight).abs().sum(-1).mean()
        else:
            input_copy = inp[:, self.idx2, :3]
            pred_x = weights_from_net * input_copy[:, :, :, 0]
            pred_y = weights_from_net * input_copy[:, :, :, 1]
            pred_z = weights_from_net * input_copy[:, :, :, 2]

            pred_verts = torch.sum(torch.stack((pred_x, pred_y, pred_z), axis=3), axis=2)

            # local neighbourhood regulaiser
            current_argmax = torch.argmax(weights_from_net, axis=2)
            idx = torch.stack([torch.index_select(self.layer_neigh, 1, current_argmax[i])[0] for i in range(self.bs)])
            current_argmax_verts = torch.stack([torch.index_select(inp[i, :, :3], 0, idx[i]) for i in range(self.bs)])
            current_argmax_verts = torch.stack([current_argmax_verts for i in range(self.num_neigh)], dim=2)
            dist_from_max = current_argmax_verts - input_copy  # todo: should it be input copy??

            dist_from_max = torch.sqrt(torch.sum(dist_from_max * dist_from_max, dim=3))
            local_regu = torch.sum(dist_from_max * weights_from_net) / (self.bs * self.num_neigh * self.layer_size)

            body_tmp = self.smpl.forward(beta=betas, theta=pose, trans=trans)
            body_mesh = [tm.from_tensors(vertices=v,
                                         faces=self.smpl_faces) for v in body_tmp]

            if self.garment_layer == 'Body':
                # update body verts with prediction
                body_tmp[:, self.vert_indices, :] = pred_verts
                # get skin cutout
                loss_data = data_loss(self.garment_layer, pred_verts, inp[:, self.vert_indices, :], self.geo_weights)
            else:
                loss_data = data_loss(self.garment_layer, pred_verts, gt_verts)

            # create mesh for predicted and smpl mesh
            pred_mesh = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in pred_verts]
            gt_mesh = [tm.from_tensors(vertices=v,
                                       faces=self.garment_f_torch) for v in gt_verts]

            loss_lap = lap_loss(pred_mesh, gt_mesh)

            # calculate normal for gt, pred and body
            loss_norm, body_normals, pred_normals = normal_loss(self.bs, pred_mesh, gt_mesh, body_mesh, self.num_faces)

            # interpenetration loss
            loss_interp = interp_loss(self.sideddistance, self.relu, pred_verts, gt_verts,  body_tmp, body_normals, self.layer_size, d_tol =self.d_tol)

            loss = loss_data + 100. * loss_lap + local_regu + loss_interp  # loss_norm

        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self, epoch, pretrain=False, train=True):
        if train:
            self.model.train()
            loss_total = 0.0
            for batch in self.train_loader:
                train_loss = self.train(batch, pretrain)
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
            inp = batch.get('inp').to(self.device)
            gt_verts = batch.get('gt_verts').to(self.device)
            betas = batch.get('betas').to(self.device)
            pose = batch.get('pose').to(self.device)
            trans = batch.get('trans').to(self.device)
            self.optimizer.zero_grad()
            bs = inp.shape[0]
            # pred_verts = self.models(torch.cat((thetas, betas, gammas), dim=1)).view(gt_verts.shape) + linear_pred
            weights_from_net = self.model(inp).view(bs, self.layer_size, self.num_neigh)
            weights_from_net = self.out_layer(weights_from_net)

            input_copy = inp[:, self.idx2, :3]
            pred_x = weights_from_net * input_copy[:, :, :, 0]
            pred_y = weights_from_net * input_copy[:, :, :, 1]
            pred_z = weights_from_net * input_copy[:, :, :, 2]
            pred_verts = torch.sum(torch.stack((pred_x, pred_y, pred_z), axis=3), axis=2)

            sum_val_loss += data_loss(self.garment_layer, pred_verts, gt_verts).item()
        return sum_val_loss, sum_val_loss

    def _save_ckpt(self, epoch):
        save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'lin.pth.tar'))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth.tar"))



def parse_argument():
    parser = argparse.ArgumentParser(description='Training ParserNet')

    parser.add_argument('--garment_class', default="g5")
    parser.add_argument('--garment_layer', default="UpperClothes")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--log_dir', default="")
    parser.add_argument('--res', default="hres")
    parser.add_argument('--feat', default="vn")  #vc_vn
    parser.add_argument('--num_neigh', default=20, type=int)
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
    pretrain = True
    for i in range(params['epoch']):
        print("epoch: {}".format(i))
        trainer.train_epoch(i)
        if i % 200 == 0:
            trainer.train_epoch(i, train=False)

