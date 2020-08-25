import os
import sys
import tensorboardX
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from psbody.mesh import Mesh
import numpy as np
import json
import pickle
import ipdb

from kaolin.metrics.mesh import point_to_surface, laplacian_loss, TriangleDistance
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.point import  SidedDistance

sys.path.append('/BS/garvita/work/code/sizer')

from models import network_layers
from models.eval import AverageMeter, verts_dist
from data_utils.mesh_dataloader import  SizerData
from data_utils.geomtery import get_vertex_normals, nearest_map, get_res_vert, get_vid
sys.path.append('/BS/garvita/work/libs/core_gt')
from log_utils import sio

from models.torch_smpl4garment import TorchSMPL4Garment


device = torch.device("cuda:0")

class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']
        self.bs = params['batch_size']
        self.vis_freq = params['vis_freq']
        self.layer = params['garment_layer']
        self.res = params['res']

        self.vc = params['vc']
        self.vn = params['vn']
        # log and backup
        LOG_DIR = '/scratch/BS/pool1/garvita/sizer'
        ROOT_DIR = '/scratch/BS/pool1/garvita/sizer'
        #log_name = os.path.join(params['log_name'], self.garment_class)
        self.feat = 'None'
        if self.vn:
            self.feat = 'vn'
        if self.vc:
            self.feat = 'vc_vn'
        self.res_name = 'hres'

        
        self.model_name = "EncDec_{}".format(self.res_name)

        self.note = "FC_size_{}_{}_{}_{}".format(self.garment_class, self.layer, self.res_name, self.res)
        log_name = os.path.join(self.garment_class, '{}_{}_{}'.format(self.layer, self.feat, self.res_name))

        self.log_dir = sio.prepare_log_dir(LOG_DIR, ROOT_DIR,log_name)
        sio.save_params(self.log_dir, params, save_name='params')

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        #load smpl data

        self.layer_size, self.smpl_size = get_res_vert(params['garment_class'],self.res, params['garment_layer'] )
        if self.layer == 'Body':
            self.layer_size = 4448

        # get active vert id
        self.num_neigh = params['num_neigh']
        input_dim = self.layer_size * 3
        if self.vn:
            input_dim = self.layer_size * 6

        output_dim = input_dim

        self.vert_indices = get_vid(self.layer, self.garment_class,self.res)

        self.vert_indices = torch.tensor(self.vert_indices.astype(np.long)).long().cuda()
        # dataset and dataloader
        self.train_dataset = SizerData(params['garment_class'], split='train',
                                             gender=self.gender, vn=self.vn)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                       drop_last=True if len(self.train_dataset) > self.bs else False)

        self.test_dataset = SizerData(params['garment_class'], split='test', vn=self.vn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, num_workers=12, shuffle=False,
                                      drop_last=False)

        #create smpl
        self.smpl = TorchSMPL4Garment(gender=self.gender).to(device)

        self.smpl_faces_np = self.smpl.faces
        self.smpl_faces = torch.tensor(self.smpl_faces_np.astype('float32'), dtype=torch.long).cuda()

        #interpenetraion loss term
        self.nearest_pt = TriangleDistance()
        self.body_f_np = self.smpl.faces
        if self.layer == 'Body':
            self.garment_f_np = self.body_f_np
            self.garment_f_torch = self.smpl_faces
        else:
            self.garment_f_np = Mesh(
                filename='/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_hres_{}.obj'.format(self.garment_class, self.layer)).f

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

        #geo_weights = np.load('/BS/garvita2/static00/ClothSize_data/gcn_assets/weights/real_g5_geo_weights.npy')
        #self.geo_weights = torch.tensor(geo_weights[body_vert2].astype(np.float32)).cuda()
        self.best_error = np.inf
        self.best_epoch = -1
        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir))
        self.val_min = None
        self.d_tol = 0.002

        #self.csv_logger = PSS2GLogger()
        self.sideddistance =  SidedDistance()
        self.relu = nn.ReLU()
        #self.csv_logger = PSS2GLogger()

    def get_loss(self, pred_verts, gt_verts):

        if self.layer == 'Body':
            data_loss = (pred_verts - gt_verts).abs().sum(-1) * self.geo_weights
            data_loss = data_loss.mean()

        else:
            data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()

        return (data_loss
                # + interp_loss
                # + 10*lapl_loss
                # + normal_loss * ( np.e ** (epoch/10-10) )
                )

    def train(self, epoch):
        epoch_loss = AverageMeter()
        self.model.train()
        for i, (gar_vert0, gar_vert1, gar_vert2, betas0, pose0, pose1, pose2, trans0, trans1, \
            trans2, size0, size1, size2, item) in enumerate(self.train_loader):


            gar_vert0 = gar_vert0.to(device)
            gar_vert1 = gar_vert1.to(device)
            gar_vert2 = gar_vert2.to(device)

            betas0 = betas0.to(device)

            pose0 = pose0.to(device)
            pose1 = pose1.to(device)
            pose2 = pose2.to(device)

            trans0 = trans0.to(device)
            trans1 = trans1.to(device)
            trans2 = trans2.to(device)

            size0 = size0.to(device)
            size1 = size1.to(device)
            size2 = size2.to(device)

            self.optimizer.zero_grad()

            # encode the displacemnt
            #todo change this to displacement in unposed space , not really because of wrong correspondence
            dis_out00 = self.model(gar_vert0, size0, size0, betas0)
            dis_out01 = self.model(gar_vert0, size0, size1, betas0)
            dis_out02 = self.model(gar_vert0, size0, size2, betas0)

            dis_out10 = self.model(gar_vert1, size1, size0, betas0)
            dis_out11 = self.model(gar_vert1, size1, size1, betas0)
            dis_out12 = self.model(gar_vert1, size1, size2, betas0)

            dis_out20 = self.model(gar_vert2, size2, size0, betas0)
            dis_out21 = self.model(gar_vert2, size2, size1, betas0)
            dis_out22 = self.model(gar_vert2, size2, size2, betas0)

            #create posed garment
            _,gar_vert00 = self.smpl.forward(beta=betas0, theta=pose0, trans=trans0, garment_class ='t-shirt', garment_d = dis_out00)
            _,gar_vert01 = self.smpl.forward(beta=betas0, theta=pose1, trans=trans1, garment_class ='t-shirt', garment_d = dis_out01)
            _,gar_vert02 = self.smpl.forward(beta=betas0, theta=pose2, trans=trans2, garment_class ='t-shirt', garment_d = dis_out02)

            _,gar_vert10 = self.smpl.forward(beta=betas0, theta=pose0, trans=trans0, garment_class='t-shirt',
                                           garment_d=dis_out10)
            _,gar_vert11 = self.smpl.forward(beta=betas0, theta=pose1, trans=trans1, garment_class='t-shirt',
                                           garment_d=dis_out11)
            _,gar_vert12 = self.smpl.forward(beta=betas0, theta=pose2, trans=trans2, garment_class='t-shirt',
                                           garment_d=dis_out12)

            _,gar_vert20 = self.smpl.forward(beta=betas0, theta=pose0, trans=trans0, garment_class='t-shirt',
                                           garment_d=dis_out20)
            _,gar_vert21 = self.smpl.forward(beta=betas0, theta=pose1, trans=trans1, garment_class='t-shirt',
                                           garment_d=dis_out21)
            _,gar_vert22 = self.smpl.forward(beta=betas0, theta=pose2, trans=trans2, garment_class='t-shirt',
                                           garment_d=dis_out22)

            pred_mesh00 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert00]
            pred_mesh01 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert01]
            pred_mesh02 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert02]

            gt_mesh0 = [tm.from_tensors(vertices=v,
                                       faces=self.garment_f_torch) for v in gar_vert0]

            pred_mesh10 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert10]
            pred_mesh11 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert11]
            pred_mesh12 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert12]

            gt_mesh1 = [tm.from_tensors(vertices=v,
                                       faces=self.garment_f_torch) for v in gar_vert1]

            pred_mesh20 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert20]
            pred_mesh21 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert21]
            pred_mesh22 = [tm.from_tensors(vertices=v,
                                         faces=self.garment_f_torch) for v in gar_vert22]

            gt_mesh2 = [tm.from_tensors(vertices=v,
                                       faces=self.garment_f_torch) for v in gar_vert2]

            lap_loss = (torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh00, gt_mesh0)]).mean() + \
                       torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh01, gt_mesh1)]).mean() + \
                       torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh02, gt_mesh2)]).mean() + \
                       torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh10, gt_mesh0)]).mean() + \
                       torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh11, gt_mesh1)]).mean() + \
                       torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh12, gt_mesh2)]).mean() + \
                       torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh20, gt_mesh0)]).mean() + \
                       torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh21, gt_mesh1)]).mean() + \
                       torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh22, gt_mesh2)]).mean() )/9.


            loss_data = (self.get_loss(gar_vert00, gar_vert0) + self.get_loss(gar_vert10, gar_vert0) +
                         self.get_loss(gar_vert20, gar_vert0) + self.get_loss(gar_vert01, gar_vert1) +
                         self.get_loss(gar_vert11, gar_vert1) +  self.get_loss(gar_vert21, gar_vert1) +
                         self.get_loss(gar_vert02, gar_vert2) +  self.get_loss(gar_vert12, gar_vert2) +
                         self.get_loss(gar_vert22, gar_vert2)
                         ) /9.

            loss = loss_data + 100.*lap_loss
            loss.backward()
            self.optimizer.step()

            self.logger.add_scalar("train/loss", loss.item(), self.iter_nums)
            print("Iter {}, loss: {:.8f}".format(self.iter_nums, loss.item()))
            epoch_loss.update(loss, gar_vert2.shape[0])
            self.iter_nums += 1

        self.logger.add_scalar("train_epoch/loss", epoch_loss.avg, epoch)
        # self._save_ckpt(epoch)

    def validate(self, epoch):
        val_loss = AverageMeter()
        val_dist = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (gar_vert0, gar_vert1, gar_vert2, betas0, pose0, pose1, pose2, trans0, trans1, \
                    trans2, size0, size1, size2, idxs) in enumerate(self.test_loader):
                gar_vert0 = gar_vert0.to(device)
                gar_vert1 = gar_vert1.to(device)
                gar_vert2 = gar_vert2.to(device)

                betas0 = betas0.to(device)

                pose0 = pose0.to(device)
                pose1 = pose1.to(device)
                pose2 = pose2.to(device)

                trans0 = trans0.to(device)
                trans1 = trans1.to(device)
                trans2 = trans2.to(device)

                size0 = size0.to(device)
                size1 = size1.to(device)
                size2 = size2.to(device)

                self.optimizer.zero_grad()

                # create smpl from params
                smpl = th_batch_SMPL(self.bs, betas0, pose0, trans0, faces=self.smpl_faces, gender=self.gender).cuda()
                smpl_verts0, _, _, _ = smpl()
                smpl = th_batch_SMPL(self.bs, betas1, pose1, trans1, faces=self.smpl_faces, gender=self.gender).cuda()
                smpl_verts1, _, _, _ = smpl()
                smpl = th_batch_SMPL(self.bs, betas2, pose2, trans2, faces=self.smpl_faces, gender=self.gender).cuda()
                smpl_verts2, _, _, _ = smpl()

                # get smpl ciutout for given garment classs
                naked0 = smpl_verts0[:, self.vert_indices, :]
                naked1 = smpl_verts1[:, self.vert_indices, :]
                naked2 = smpl_verts2[:, self.vert_indices, :]
                dis0 = gar_vert0 - naked0
                dis1 = gar_vert1 - naked1
                dis2 = gar_vert2 - naked2

                # encode the displacemnt
                dis_out00 = self.model(dis0, size0, size0)
                dis_out01 = self.model(dis0, size0, size1)
                dis_out02 = self.model(dis0, size0, size2)


                gar_out00 = naked0 + dis_out00
                gar_out01 = naked1 + dis_out01
                gar_out02 = naked2 + dis_out02

                #encode dis1

                dis_out10 = self.model(dis1, size1, size0)
                dis_out11 = self.model(dis1, size1, size1)
                dis_out12 = self.model(dis1, size1, size2)

                gar_out10 = naked0 + dis_out10
                gar_out11 = naked1 + dis_out11
                gar_out12 = naked2 + dis_out12

                #enocde dis2
                dis_out20 = self.model(dis2, size2, size0)
                dis_out21 = self.model(dis2, size2, size1)
                dis_out22 = self.model(dis2, size2, size2)

                gar_out20 = naked0 + dis_out20
                gar_out21 = naked1 + dis_out21
                gar_out22 = naked2 + dis_out22


                loss = (self.get_loss(gar_out00, gar_vert0, "val", epoch) +  self.get_loss(gar_out01, gar_vert1, "val", epoch) +
                        self.get_loss(gar_out02, gar_vert2, "val", epoch) + self.get_loss(gar_out10, gar_vert0, "val", epoch) +
                        self.get_loss(gar_out11, gar_vert1, "val", epoch) +  self.get_loss(gar_out12, gar_vert2, "val", epoch) +
                        self.get_loss(gar_out20, gar_vert0, "val", epoch) + self.get_loss(gar_out21, gar_vert1, "val",  epoch) +
                        self.get_loss(gar_out22, gar_vert2, "val", epoch) )/9.
                dist = (verts_dist(gar_out00, gar_vert0) * 1000. + verts_dist(gar_out01, gar_vert1) * 1000.
                        +verts_dist(gar_out02, gar_vert2) * 1000. + verts_dist(gar_out10, gar_vert0) * 1000. + verts_dist(gar_out11, gar_vert1) * 1000.
                        +verts_dist(gar_out12, gar_vert2) * 1000. +verts_dist(gar_out20, gar_vert0) * 1000. + verts_dist(gar_out21, gar_vert1) * 1000.
                        +verts_dist(gar_out22, gar_vert2) * 1000.)/9.
                val_loss.update(loss.item(), gar_out00.shape[0])
                val_dist.update(dist.item(), gar_out00.shape[0])


                for lidx, idx in enumerate(idxs):
                    #ipdb.set_trace()
                    #if idx % self.vis_freq != 0:
                    #    continue
                    pred_vert = gar_out00[lidx].cpu().numpy()
                    gt_vert = gar_vert0[lidx].cpu().numpy()
                    body_vert = smpl_verts0[lidx].cpu().numpy()
                    body_mesh = Mesh(v=body_vert,f=self.smpl_faces_np)

                    pred_m = Mesh(v=pred_vert, f=self.garment_f_np)
                    gt_m = Mesh(v=gt_vert, f=self.garment_f_np)
                    save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
                    pred_m.write_ply(os.path.join(save_dir, "pred_{}_{}.ply".format(idx, size0[lidx].cpu().detach().numpy())))
                    gt_m.write_ply(os.path.join(save_dir, "gt_{}_{}.ply".format(idx,size0[lidx].cpu().detach().numpy())))
                    body_mesh.write_ply(os.path.join(save_dir, "smpl_{}_{}.ply".format(idx,size0[lidx].cpu().detach().numpy())))

                    #    continue
                    pred_vert = gar_out01[lidx].cpu().numpy()
                    gt_vert = gar_vert1[lidx].cpu().numpy()
                    body_vert = smpl_verts1[lidx].cpu().numpy()
                    body_mesh = Mesh(v=body_vert, f=self.smpl_faces_np)

                    pred_m = Mesh(v=pred_vert, f=self.garment_f_np)
                    gt_m = Mesh(v=gt_vert, f=self.garment_f_np)
                    save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
                    pred_m.write_ply(os.path.join(save_dir, "pred_{}_{}.ply".format(idx, size1[lidx].cpu().detach().numpy())))
                    gt_m.write_ply(os.path.join(save_dir, "gt_{}_{}.ply".format(idx,size1[lidx].cpu().detach().numpy())))
                    body_mesh.write_ply(os.path.join(save_dir, "smpl_{}_{}.ply".format(idx,size1[lidx].cpu().detach().numpy())))

                    #    continue
                    pred_vert = gar_out02[lidx].cpu().numpy()
                    gt_vert = gar_vert2[lidx].cpu().numpy()
                    body_vert = smpl_verts2[lidx].cpu().numpy()
                    body_mesh = Mesh(v=body_vert, f=self.smpl_faces_np)

                    pred_m = Mesh(v=pred_vert, f=self.garment_f_np)
                    gt_m = Mesh(v=gt_vert, f=self.garment_f_np)
                    save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
                    pred_m.write_ply(os.path.join(save_dir, "pred_{}_{}.ply".format(idx, size2[lidx].cpu().detach().numpy())))
                    gt_m.write_ply(os.path.join(save_dir, "gt_{}_{}.ply".format(idx,size2[lidx].cpu().detach().numpy())))
                    body_mesh.write_ply(os.path.join(save_dir, "smpl_{}_{}.ply".format(idx,size2[lidx].cpu().detach().numpy())))
                    # linear_m.write_ply(os.path.join(save_dir, "linear_pred_{}.ply".format(idx)))
                    #body_m.write_ply(os.path.join(save_dir, "body_{}.ply".format(idx)))

        self.logger.add_scalar("val/loss", val_loss.avg, epoch)
        self.logger.add_scalar("val/dist", val_dist.avg, epoch)
        print("VALIDATION")
        print("Epoch {}, loss: {:.4f}, dist: {:.4f} mm".format(epoch, val_loss.avg, val_dist.avg))
        self._save_ckpt(epoch)

        if val_dist.avg < self.best_error:
            self.best_error = val_dist.avg
            self.best_epoch = epoch
            with open(os.path.join(self.log_dir, 'best_epoch'), 'w') as f:
                f.write("{:04d}".format(epoch))

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
        self.layer_size = get_res_vert(params['garment_class'],self.res, params['garment_layer'] )[0]
        self.layer =params['garment_layer']
        # get active vert id
        self.num_neigh = 20
        output_dim = self.layer_size *  self.num_neigh

        res_name = 'lres'
        layer_neigh = np.array(np.load("/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_neighborheuristics_{}_{}_{}_gar_order2.npy".format(self.garment_class, res_name, self.layer, self.num_neigh)))
        self.layer_neigh = layer_neigh
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
        weights_from_net = self.model(inp).view(bs, self.layer_size, self.num_neigh)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--garment_class', default="g5")
    parser.add_argument('--garment_layer', default="UpperClothes")
    parser.add_argument('--gender', default="male")
    parser.add_argument('--res', default=True, type=bool)
    parser.add_argument('--vc', default=False, type=bool)
    parser.add_argument('--vn', default=False, type=bool)
    parser.add_argument('--num_neigh', default=20, type=int)

    parser.add_argument('--vis_freq', default=16, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--dropout', default=0.3)

    args = parser.parse_args()

    params = args.__dict__

    if os.path.exists(params['local_config']):
        print("loading config from {}".format(params['local_config']))
        with open(params['local_config']) as f:
            lc = json.load(f)
        for k, v in lc.items():
            params[k] = v
    return params


if __name__ == '__main__':
    params = parse_argument()
    start_epoch = params['start_epoch']
    trainer = Trainer(params)
    try:
        # if True:
        for i in range(start_epoch, params['max_epoch']):
            print("epoch: {}".format(i))
            trainer.train(i)
            if i % 200 == 0:
                trainer.validate(i)
    finally:
        # else:
        trainer.write_log()
        print("safely quit!")
