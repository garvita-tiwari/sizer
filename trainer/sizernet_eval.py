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
from utils.global_var import DATA_DIR

class Runner(object):
    def __init__(self, ckpt, params):
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
        LOG_DIR = '/scratch/BS/pool1/garvita/sizer'
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
        self.test_dataset = SizerData(garment_class=self.garment_class, garment_layer=self.garment_layer,
                                       mode='test', batch_size=self.bs, res='hres', gender='male')
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                       drop_last=True if len(self.train_dataset) > self.bs else False)

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

        print("loading {}".format(ckpt))
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, ):
         pass

    def eval_test(self):

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
            body_verts, pred_verts = self.smpl.forward(beta=betas_feat, theta=pose_all, trans=trans_all, garment_class='t-shirt',
                                              garment_d=all_dist)

            sum_val_loss += data_loss(self.garment_layer, pred_verts, gt_verts).item()
        return sum_val_loss, sum_val_loss, gt_verts.detach().cpu().numpy(), pred_verts.detach().cpu().numpy(), body_verts.detach().cpu().numpy(), self.garment_f_np, self.body_f_np


    def cuda(self):
        self.model.cuda()

    def to(self, device):
        self.model.to(device)


def get_model(log_dir, epoch_num=None):
    ckpt_dir = log_dir
    with open(os.path.join(ckpt_dir, 'params.json')) as jf:
        params = json.load(jf)

    if epoch_num is None:
        with open(os.path.join(ckpt_dir, 'best_epoch')) as f:
            best_epoch = int(f.read().strip())
    else:
        best_epoch = epoch_num
    ckpt_path = os.path.join(ckpt_dir, "{:04d}".format(best_epoch), 'lin.pth.tar')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, 'lin.pth.tar')

    runner = Runner(ckpt_path, params)
    return runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating ParserNet')

    parser.add_argument('--log_dir', default="")
    args = parser.parse_args()

    log_dir = args.log_dir
    runner = get_model(log_dir)
    _, dist, gt, pred, body, gar_faces, body_faces = runner.eval_test()
    ipdb.set_trace()
    print('average v2v error: ', dist)
    for i in range(len(body)):
        gt_mesh = Mesh(v=gt[i], f=gar_faces)
        pred_mesh = Mesh(v=pred[i], f=gar_faces)
        inp_mesh = Mesh(v=body[i], f=body_faces)
        gt_mesh.write_obj(os.path.join(log_dir, 'gt_{}.obj'.format(i)))
        pred_mesh.write_obj(os.path.join(log_dir, 'pred_{}.obj'.format(i)))
        inp_mesh.write_obj(os.path.join(log_dir, 'ip_{}.obj'.format(i)))
