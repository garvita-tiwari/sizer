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

from kaolin.metrics.mesh import point_to_surface, laplacian_loss, TriangleDistance
from kaolin.metrics.point import  SidedDistance
from kaolin.rep import TriangleMesh as tm

sys.path.append('/BS/garvita/work/code/sizer')

from models import network_layers
from models.eval import AverageMeter, verts_dist, batched_index_select
from data_utils.parser_data import ParserData
from data_utils.geomtery import get_vertex_normals, nearest_map, get_res_vert, get_vid
sys.path.append('/BS/garvita/work/libs/core_gt')
from log_utils import sio

from models.torch_smpl4garment import TorchSMPL4Garment

device = torch.device("cuda:0")

class Trainer(object):
    def __init__(self, params):
        self.device= device
        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']
        self.bs = params['batch_size']
        self.vis_freq = params['vis_freq']
        self.garment_layer = params['garment_layer']
        self.hres = params['res']
        self.num_neigh = params['num_neigh']

        self.vc = params['vc']
        self.vn = params['vn']
        # log and backup
        LOG_DIR = '/scratch/BS/pool1/garvita/parser'
        ROOT_DIR = '/scratch/BS/pool1/garvita/parser'
        #log_name = os.path.join(params['log_name'], self.garment_class)
        self.feat = 'None'
        if self.vc:
            self.feat = 'vc_vn'
        if self.vn:
            self.feat = 'vn'

        if self.hres:
            self.res_name = 'hres'
        else:
            self.res_name = 'lres'

        #self.model_name = "{}_{}_{}_{}".format(self.garment_class, self.layer, self.res_name, self.feat)
        self.model_name = 'FC_correspondence_{}'.format(self.res_name)
        self.note = "FC_corr_{}_{}_{}".format(self.garment_class, self.garment_layer, self.res_name)
        log_name = os.path.join(self.garment_class, '{}_{}_{}_{}'.format(self.garment_layer, self.feat, self.num_neigh, self.res_name))

        self.log_dir = sio.prepare_log_dir(LOG_DIR, ROOT_DIR,log_name)
        sio.save_params(self.log_dir, params, save_name='params')

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        #load smpl data

        self.layer_size, self.smpl_size = get_res_vert(params['garment_class'],self.hres, params['garment_layer'] )
        if self.garment_layer == 'Body':
            self.layer_size = 4448

        # get active vert id
        input_dim = self.smpl_size * 3
        if self.vn:
            input_dim = self.smpl_size * 6

        output_dim = self.layer_size *  self.num_neigh
        layer_neigh = np.array(np.load("/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_neighborheuristics_{}_{}_{}_gar_order2.npy".format(self.garment_class, self.res_name, self.garment_layer, self.num_neigh)))
        self.layer_neigh = torch.from_numpy(layer_neigh).cuda()

        #separate for body layer
        body_vert = range(27554)
        vert_id_upper = get_vid('UpperClothes', self.garment_class, False)
        all_vid_upper = np.array([[i, vid] for i in range(self.bs) for vid in vert_id_upper])

        vert_id_lower = get_vid('Pants', self.garment_class, False)
        all_vid_lower = np.array([[i, vid] for i in range(self.bs) for vid in vert_id_lower])

        body_vert2 = [i for i in body_vert if i not in vert_id_upper]
        body_vert2 = [i for i in body_vert2 if i not in vert_id_lower]
        self.body_vert = body_vert2
        self.all_vid_body = torch.from_numpy(np.array([[vid] for i in range(self.bs)  for vid in self.body_vert])).cuda()


        #all_neighbors = np.array([[i, vid] for i in range(self.bs) for k in layer_neigh for vid in k])
        all_neighbors = np.array([[i, vid] for i in range(self.bs) for k in layer_neigh for vid in k])
        all_neighbors = np.array([[vid] for i in range(self.bs) for k in layer_neigh for vid in k])

        self.neigh_id = all_neighbors
        self.idx = torch.from_numpy(self.neigh_id).cuda()

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
        #self.train_dataset = ParserData(params['garment_class'], split='train',
        #                                     gender=self.gender, vn=self.vn, layer_size=self.layer_size, body_size= self.smpl_size)
        self.train_dataset = ParserData(garment_class=self.garment_class, garment_layer=self.garment_layer,
                                        mode='train', batch_size= self.bs, res=self.res_name, gender=self.gender, vc = self.vc,
                                        vn= self.vn)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                       drop_last=True if len(self.train_dataset) > self.bs else False)

        self.val_dataset = ParserData(garment_class=self.garment_class, garment_layer=self.garment_layer,
                                        mode='val', batch_size= self.bs,  res=self.res_name, gender=self.gender, vc = self.vc,
                                        vn= self.vn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                      drop_last=False)

        #create smpl
        self.smpl = TorchSMPL4Garment(gender=self.gender).to(device)

        self.smpl_faces_np = self.smpl.faces
        self.smpl_faces = torch.tensor(self.smpl_faces_np.astype('float32'), dtype=torch.long).cuda()

        #interpenetraion loss term
        self.nearest_pt = TriangleDistance()
        self.body_f_np = self.smpl.faces

        if self.garment_layer == 'Body':
            self.garment_f_np = self.body_f_np
            self.garment_f_torch = self.smpl_faces
        else:
            self.garment_f_np = Mesh(filename='/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_{}_{}.obj'.format(self.garment_class,self.res_name, self.garment_layer)).f

            self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)
        # models and optimizer
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

        geo_weights = np.load('/BS/garvita2/static00/ClothSize_data/gcn_assets/weights/real_g5_geo_weights.npy')
        self.geo_weights = torch.tensor(geo_weights[body_vert2].astype(np.float32)).cuda()
        self.best_error = np.inf
        self.best_epoch = -1
        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir))
        self.val_min = None
        self.d_tol = 0.002

        #self.csv_logger = PSS2GLogger()
        self.sideddistance =  SidedDistance()
        self.relu = nn.ReLU()
        #weight initialiser
        vert_id = self.vert_indices.cpu().numpy()
        #todo: wrong init, check this for neighbourhood
        init_weights = torch.from_numpy(np.array([layer_neigh[i] ==vert_id[i] for i in range(self.layer_size)]).astype('int64'))
        self.init_weight = torch.stack([init_weights for i in range(self.bs)]).cuda()

    def data_loss(self, pred_verts, gt_verts):

        if self.garment_layer == 'Body':
            data_loss = (pred_verts - gt_verts).abs().sum(-1) * self.geo_weights
            data_loss = data_loss.mean()

        else:
            data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()

        return data_loss

    def lap_loss(self, pred_mesh, gt_mesh):

        return  torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh, gt_mesh)]).mean()

    def normal_loss(self, pred_mesh, gt_mesh, body_mesh):
        body_normals = []
        gt_normals = []
        pred_normals = []
        for i in range(self.bs):
            b_normal = body_mesh[i].compute_face_normals()
            body_normals.append(b_normal)
            gt_nromal = gt_mesh[i].compute_face_normals()
            pred_normal = pred_mesh[i].compute_face_normals()
            gt_normals.append(gt_nromal)
            pred_normals.append(pred_normal)

        body_normals = torch.stack(body_normals)
        gt_normals = torch.stack(gt_normals)
        pred_normals = torch.stack(pred_normals)
        loss_norm = torch.sum(torch.sum((1 - gt_normals) * pred_normals, dim=2).abs()) / (self.bs * self.num_faces)

        return loss_norm, body_normals, pred_normals

    def interp_loss(self,pred_verts, gt_verts,  body_tmp, body_normals ):
        dist1 = self.sideddistance(pred_verts, body_tmp)
        nearest_body_verts = torch.stack([body_tmp[i][dist1[i]] for i in range(self.bs)])
        # todo: change this to normal
        nearest_body_normals = torch.stack([body_normals[i][dist1[i]] for i in range(self.bs)])
        # body normal
        # select vertices which are near to gt
        vert_dist = pred_verts - gt_verts
        vert_dist = torch.sqrt(torch.sum(vert_dist * vert_dist, dim=2))
        # find activet indices, which are near to gt prediction
        active_id = vert_dist < self.d_tol

        # calculate interp loss #todo: check this sign
        loss_interp = self.relu(torch.sum((nearest_body_verts - pred_verts) * nearest_body_normals, dim=2))
        loss_interp = torch.sum(active_id * loss_interp) / (self.bs * self.layer_size)

        return loss_interp

    def train(self, batch, pretrain=False,):
        inp = batch.get('inp').to(self.device)
        gt_verts = batch.get('gt_verts').to(self.device)
        betas = batch.get('betas').to(self.device)
        pose = batch.get('pose').to(self.device)
        trans = batch.get('trans').to(self.device)

        self.optimizer.zero_grad()
        # pred_verts = self.models(torch.cat((thetas, betas, gammas), dim=1)).view(gt_verts.shape) + linear_pred
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
            # todo: change this to attention
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
            # todo: find quick way for this

            if self.garment_layer == 'Body':
                # update body verts with prediction
                body_tmp[:, self.vert_indices, :] = pred_verts
                # get skin cutout
                loss_data = self.data_loss(pred_verts, inp[:, self.vert_indices, :])
            else:
                loss_data = self.data_loss(pred_verts, gt_verts)

            # create mesh for predicted and smpl mesh
            pred_mesh = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in pred_verts]
            gt_mesh = [tm.from_tensors(vertices=v,
                                       faces=self.garment_f_torch) for v in gt_verts]

            loss_lap = self.lap_loss(pred_mesh, gt_mesh)

            # calculate normal for gt, pred and body
            loss_norm, body_normals, pred_normals = self.normal_loss(pred_mesh, gt_mesh, body_mesh)

            # interpenetration loss
            loss_interp = self.interp_loss(pred_verts, gt_verts,  body_tmp, body_normals)

            loss = loss_data + 100. * loss_lap + local_regu + loss_interp  # todo: it gives error +loss_norm

        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self, epoch, pretrain=False, train=True):
        if train:
            epoch_loss = AverageMeter()
            self.model.train()
            loss_total = 0.0
            for batch in self.train_loader:
                train_loss = self.train(batch, pretrain)
                loss_total += train_loss.item()
                self.logger.add_scalar("train/loss", train_loss.item(), self.iter_nums)
                print("Iter {}, loss: {:.8f}".format(self.iter_nums, train_loss.item()))
                epoch_loss.update(train_loss, self.bs)
                self.iter_nums += 1
            self.logger.add_scalar("train_epoch/loss", epoch_loss.avg, epoch)
        else:  #validation
            self._save_ckpt(epoch)
            val_loss, val_dist = self.validation(epoch)

            if self.val_min is None:
                self.val_min = val_loss

            if val_loss < self.val_min:
                self.val_min = val_loss
                with open(os.path.join(self.log_dir, 'best_epoch'), 'w') as f:
                    f.write("{:04d}".format(epoch))

            self.writer.add_scalar('val loss batch avg', val_loss, epoch)
            self.writer.add_scalar('val dist batch avg', val_dist, epoch)

    def validation(self, epoch):
        val_loss = AverageMeter()
        val_dist = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (batch) in enumerate(self.val_loader):
                inp = batch.get('inp').to(self.device)
                gt_verts = batch.get('gt_verts').to(self.device)
                betas = batch.get('betas').to(self.device)
                pose = batch.get('pose').to(self.device)
                trans = batch.get('trans').to(self.device)
                self.optimizer.zero_grad()
                # pred_verts = self.models(torch.cat((thetas, betas, gammas), dim=1)).view(gt_verts.shape) + linear_pred
                weights_from_net = self.model(inp).view(self.bs, self.layer_size, self.num_neigh)
                weights_from_net = self.out_layer(weights_from_net)

                # make a copy of neighbour for each vertex
                input_copy = inp[:, self.idx2, :3]
                pred_x = weights_from_net * input_copy[:, :, :, 0]
                pred_y = weights_from_net * input_copy[:, :, :, 1]
                pred_z = weights_from_net * input_copy[:, :, :, 2]

                pred_verts = torch.sum(torch.stack((pred_x, pred_y, pred_z), axis=3), axis=2)
                smpl_verts = self.smpl.forward(beta=betas, theta=pose, trans=trans)

                if self.layer == 'Body':
                    loss = self.loss_data(pred_verts, inp[:, self.vert_indices, :])
                    dist = verts_dist(pred_verts, inp[:, self.vert_indices, :]) * 1000.
                    val_loss.update(loss.item(), gt_verts.shape[0])
                    val_dist.update(dist.item(), gt_verts.shape[0])
                else:
                    loss = self.loss_data(pred_verts, gt_verts)
                    dist = verts_dist(gt_verts, pred_verts) * 1000.
                    val_loss.update(loss.item(), gt_verts.shape[0])
                    val_dist.update(dist.item(), gt_verts.shape[0])

                for lidx, idx in enumerate(idxs):
                    #ipdb.set_trace()
                    #if idx % self.vis_freq != 0:
                    #    continue
                    pred_vert = pred_verts[lidx].cpu().numpy()
                    gt_vert = gt_verts[lidx].cpu().numpy()
                    body_vert = smpl_verts[lidx].cpu().numpy()
                    body_mesh = Mesh(v=body_vert,f=self.smpl_faces_np)

                    if self.garment_layer == 'Body':
                        body_tmp = smpl_verts[lidx].cpu().numpy()
                        # update body verts with prediction
                        body_tmp[self.vert_indices.cpu().numpy(), :] = pred_vert
                        pred_vert = body_tmp
                        gt_vert = inp[lidx].cpu().numpy()

                    pred_m = Mesh(v=pred_vert, f=self.garment_f_np)
                    gt_m = Mesh(v=gt_vert, f=self.garment_f_np)
                    save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
                    pred_m.write_ply(os.path.join(save_dir, "pred_{}.ply".format(idx)))
                    gt_m.write_ply(os.path.join(save_dir, "gt_{}.ply".format(idx)))
                    body_mesh.write_ply(os.path.join(save_dir, "smpl_{}.ply".format(idx)))

        self.logger.add_scalar("val/loss", val_loss.avg, epoch)
        self.logger.add_scalar("val/dist", val_dist.avg, epoch)
        print("VALIDATION epoch {}, loss: {:.4f}, dist: {:.4f} mm".format(epoch, val_loss.avg, val_dist.avg))
        return val_loss.avg, val_dist.avg

    def write_log(self):
        if self.best_epoch >= 0:
            self.csv_logger.add_item(best_error=self.best_error, best_epoch=self.best_epoch, **self.params)

    def _save_ckpt(self, epoch):
        save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'lin.pth.tar'))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth.tar"))


class Runner(object):
    def __init__(self, ckpt, params):

        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']
        self.bs = params['batch_size']
        self.vis_freq = params['vis_freq']
        self.model_name = params['model_name']
        self.note = params['note']

        params['garment_layer'] = 'UpperClothes'
        #load smpl data
        self.smpl = SmplPaths(gender=self.gender)
        body_vert = range(6890)
        self.smpl_size = len(body_vert)
        input_dim = self.smpl_size * 6
        self.layer_size = get_res_vert(params['garment_class'],False, params['garment_layer'] )[0]
        self.layer =params['garment_layer']
        # get active vert id
        self.num_neigh = 20
        output_dim = self.layer_size *  self.num_neigh

        layer_neigh = np.array(np.load("/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_neighborheuristics_{}_{}_{}_gar_order2.npy".format(self.garment_class, res_name, self.layer, self.num_neigh)))
        self.layer_neigh = torch.from_numpy(layer_neigh).cuda()
        #all_neighbors = np.array([[i, vid] for i in range(self.bs) for k in layer_neigh for vid in k])
        all_neighbors = np.array([[i, vid] for i in range(self.bs) for k in layer_neigh for vid in k])
        all_neighbors = np.array([[vid] for i in range(self.bs) for k in layer_neigh for vid in k])

        self.neigh_id = all_neighbors
        self.idx = torch.from_numpy(self.neigh_id).cuda()

        all_neighbors = np.array([[vid] for k in layer_neigh for vid in k])

        self.neigh_id2 = all_neighbors
        self.idx2 = torch.from_numpy(self.neigh_id2).view(self.layer_size, self.num_neigh).cuda()
        #class_info = pickle.load(open(os.path.join(DATA_DIR, 'garment_class_info.pkl'), 'rb'), encoding="latin1")

        self.body_f_np = self.smpl.get_faces()

        self.garment_f_np = Mesh(
            filename='/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_lres_{}.obj'.format(self.garment_class, self.layer)).f

        self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)
        self.vert_indices = None


        #create smpl
        sp = SmplPaths(gender="male")
        smpl_faces = sp.get_faces()
        self.smpl_faces_np = smpl_faces
        self.smpl_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).cuda()

        #interpenetraion loss term
        self.nearest_pt = TriangleDistance()

        # models and optimizer
        self.model = getattr(network_layers, self.model_name)(input_size=input_dim, output_size=output_dim)

        self.model.to(device)
        self.out_layer = torch.nn.Softmax(dim=2)

        print("loading {}".format(ckpt))
        if torch.cuda.is_available():
            self.model.cuda()
            # self.linear_model.cuda()
            state_dict = torch.load(ckpt)
        else:
            state_dict = torch.load(ckpt, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, inp, betas, pose, trans=None, gt=None):
        bs = inp.shape[0]
        ipdb.set_trace()

        weights_from_net = self.model(inp)

        weights_from_net =weights_from_net.view(bs, self.layer_size, self.num_neigh)
        # normalise weights
        # weight_norm = torch.norm(weights_from_net, p=2, dim=2, keepdim=True)
        # weights_from_net = weights_from_net.div(weight_norm.expand_as(weights_from_net))
        weights_from_net = self.out_layer(weights_from_net)

        # make a copy of neighbour for each vertex
        input_copy = inp[:, self.idx2, :]
        pred_x = weights_from_net * input_copy[:, :, :, 0]
        pred_y = weights_from_net * input_copy[:, :, :, 1]
        pred_z = weights_from_net * input_copy[:, :, :, 2]

        pred_verts = torch.sum(torch.stack((pred_x, pred_y, pred_z), axis=3), axis=2)

        if trans is None:
            trans =  torch.zeros((self.bs, 3))

        smpl = th_batch_SMPL(bs, betas, pose, trans, faces=self.smpl_faces).cuda()
        smpl_verts, _, _, _ = smpl()

        dist = None
        if gt is not None:
            dist = verts_dist(gt, pred_verts, dim =1) * 1000.

        return pred_verts.detach().cpu().numpy(), smpl_verts.detach().cpu().numpy(), self.garment_f_np, self.smpl_faces_np, dist.detach().cpu().numpy()

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



def parse_argument():
    parser = argparse.ArgumentParser(description='Training ParserNet')

    parser.add_argument('--garment_class', default="g5")
    parser.add_argument('--garment_layer', default="UpperClothes")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--res', default=True, type=bool)
    parser.add_argument('--vc', default=False, type=bool)
    #parser.add_argument('--vn', default=True, type=bool)
    parser.add_argument('--vn', dest='vn', action='store_true')
    parser.add_argument('--no-vn', dest='vn', action='store_false')
    parser.set_defaults(feature=True)
    parser.add_argument('--num_neigh', default=20, type=int)

    parser.add_argument('--vis_freq', default=16, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epoch', default=2000, type=int)
    parser.add_argument('--checkpoint', default="")

    #parser.add_argument('--num_layers', default=3)
    parser.add_argument('--dropout', default=0.3)

    args = parser.parse_args()

    params = args.__dict__
    #
    # if os.path.exists(params['local_config']):
    #     print("loading config from {}".format(params['local_config']))
    #     with open(params['local_config']) as f:
    #         lc = json.load(f)
    #     for k, v in lc.items():
    #         params[k] = v
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

    trainer.write_log()
