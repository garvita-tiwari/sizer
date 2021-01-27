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
from utils.global_var import DATA_DIR

device = torch.device("cuda:0")
class Runner(object):
    def __init__(self, ckpt, params):
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
        self.model_name = 'FC_correspondence_{}'.format(self.res_name)
        self.layer_size, self.smpl_size = get_res_vert(params['garment_class'], self.hres, params['garment_layer'])

        layer_neigh = np.array(np.load(os.path.join(DATA_DIR, "real_{}_neighborheuristics_{}_{}_{}_gar_order2.npy".format(self.garment_class, self.res_name, self.garment_layer, self.num_neigh))))

        all_neighbors = np.array([[vid] for k in layer_neigh for vid in k])
        self.neigh_id2 = all_neighbors
        if self.garment_layer == 'Body':
            self.idx2 = torch.from_numpy(self.neigh_id2).view(len(self.body_vert), self.num_neigh).cuda()
        else:
            self.idx2 = torch.from_numpy(self.neigh_id2).view(self.layer_size, self.num_neigh).cuda()

        self.test_dataset = ParserData(garment_class=self.garment_class, garment_layer=self.garment_layer,
                                        mode='test', batch_size= self.bs,  res=self.res_name, gender=self.gender, feat=self.feat)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                      drop_last=False)
        # #create smpl
        self.smpl = TorchSMPL4Garment(gender=self.gender).to(device)
        self.smpl_faces_np = self.smpl.faces
        self.smpl_faces = torch.tensor(self.smpl_faces_np.astype('float32'), dtype=torch.long).cuda()

        if self.garment_layer == 'Body':
            self.garment_f_np = self.body_f_np
            self.garment_f_torch = self.smpl_faces
        else:
            self.garment_f_np = Mesh(filename=os.path.join(DATA_DIR,'real_{}_{}_{}.obj'.format(self.garment_class,self.res_name, self.garment_layer))).f
            self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)

        self.out_layer = torch.nn.Softmax(dim=2)
        input_dim = self.smpl_size * 3
        if self.feat == 'vn':
            input_dim = self.smpl_size * 6
        output_dim = self.layer_size *  self.num_neigh

        self.model = getattr(network_layers, self.model_name)(input_size=input_dim, output_size=output_dim)
        self.model.to(self.device)
        print("loading {}".format(ckpt))
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, inp, betas, pose, trans=None, gt=None):
        bs = inp.shape[0]
        ipdb.set_trace()
        weights_from_net = self.model(inp)
        weights_from_net = weights_from_net.view(bs, self.layer_size, self.num_neigh)
        weights_from_net = self.out_layer(weights_from_net)

        # make a copy of neighbour for each vertex
        input_copy = inp[:, self.idx2, :3]
        pred_x = weights_from_net * input_copy[:, :, :, 0]
        pred_y = weights_from_net * input_copy[:, :, :, 1]
        pred_z = weights_from_net * input_copy[:, :, :, 2]

        pred_verts = torch.sum(torch.stack((pred_x, pred_y, pred_z), axis=3), axis=2)

        if trans is None:
            trans =  torch.zeros((self.bs, 3))

        smpl_verts = self.smpl.forward(beta=betas, theta=pose, trans=trans)

        dist = None
        if gt is not None:
            dist = verts_dist(gt, pred_verts, dim =1) * 1000.

        return pred_verts.detach().cpu().numpy(), smpl_verts.detach().cpu().numpy(), self.garment_f_np, self.smpl_faces_np, dist.detach().cpu().numpy()

    def eval_test(self):

        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                batch = self.test_data_iterator.next()
            except:
                self.test_data_iterator = self.test_loader.__iter__()
                batch = self.test_data_iterator.next()
            inp = batch.get('inp').to(self.device)
            gt_verts = batch.get('gt_verts').to(self.device)
            betas = batch.get('betas').to(self.device)
            pose = batch.get('pose').to(self.device)
            trans = batch.get('trans').to(self.device)
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
        return sum_val_loss, sum_val_loss, gt_verts.detach().cpu().numpy(), pred_verts.detach().cpu().numpy(), inp[:,:, :3].detach().cpu().numpy(), self.garment_f_np, self.smpl_faces_np

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
    _, dist, gt, pred, inp, gar_faces, body_faces = runner.eval_test()
    print('average v2v error: ', dist)
    for i in range(len(inp)):
        gt_mesh = Mesh(v=gt[i], f=gar_faces)
        pred_mesh = Mesh(v=pred[i], f=gar_faces)
        inp_mesh = Mesh(v=inp[i], f=body_faces)
        gt_mesh.write_obj(os.path.join(log_dir, 'gt_{}.obj'.format(i)))
        pred_mesh.write_obj(os.path.join(log_dir, 'pred_{}.obj'.format(i)))
        inp_mesh.write_obj(os.path.join(log_dir, 'ip_{}.obj'.format(i)))
