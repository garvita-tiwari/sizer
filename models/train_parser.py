from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
import ipdb
#todo: create a base trainer
from models.network import  net_modules
from models.loss import lap_loss, interp_loss, data_loss, normal_loss, verts_dist
from models.torch_smpl4garment import TorchSMPL4Garment
from data.geomtery import get_res_vert, get_vid
#from psbody.mesh import Mesh
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes
class ParserNet(object):

    def __init__(self,  train_dataset, val_dataset, opt):
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
        self.smpl_faces_np = self.smpl.faces
        self.smpl_faces = torch.tensor(self.smpl_faces_np.astype('float32'), dtype=torch.long).cuda()


        # load training parameters etc
        self.layer_size, self.smpl_size = get_res_vert(self.garment_class,self.res, self.garment_layer )
        if self.garment_layer == 'Body':  #todo: move this in the function
            self.layer_size = 4448
        # get active vert id
        input_dim = self.smpl_size * 3
        if self.feat == 'vn':
            input_dim = self.smpl_size * 6
        output_dim = self.layer_size *  self.num_neigh


        layer_neigh = np.array(np.load(os.path.join(opt['data']['meta_data'], "{}/{}_{}_{}_gar_order.npy".format(self.garment_class,  self.garment_layer, self.res,self.num_neigh))))
        self.layer_neigh = torch.from_numpy(layer_neigh).cuda()

        #separate for body layer
        body_vert = range(self.smpl_size)
        vert_id_upper = get_vid(opt['data']['meta_data'],'UpperClothes', self.garment_class, self.res)
        vert_id_lower = get_vid(opt['data']['meta_data'],'Pants', self.garment_class, self.res)
        body_vert2 = [i for i in body_vert if i not in vert_id_upper]
        body_vert2 = [i for i in body_vert2 if i not in vert_id_lower]
        self.body_vert = body_vert2

        all_neighbors = np.array([[vid] for k in layer_neigh for vid in k])
        self.neigh_id2 = all_neighbors
        if self.garment_layer == 'Body':
            self.idx2 = torch.from_numpy(self.neigh_id2).view(len(self.body_vert), self.num_neigh).cuda()
        else:
            self.idx2 = torch.from_numpy(self.neigh_id2).view(self.layer_size, self.num_neigh).cuda()

        self.vert_indices = get_vid(opt['data']['meta_data'],self.garment_layer, self.garment_class,self.res)
        self.vert_indices = torch.tensor(self.vert_indices.astype(np.long)).long().cuda()


        if self.garment_layer == 'Body':
            #self.garment_f_np = self.body_f_np
            #self.garment_f_np = Mesh(filename='/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_lres_{}.obj'.format(garment, layer)).f
            self.garment_f_torch = self.smpl_faces
        else:
            mesh = load_objs_as_meshes([os.path.join(opt['data']['meta_data'], "{}/{}_{}.obj".format(self.garment_class,  self.garment_layer, self.res))], device=self.device)
            mesh_verts, mesh_faces = mesh.get_mesh_verts_faces(0)
            self.garment_f_torch = mesh_faces

        self.num_faces = len(self.garment_f_torch)
        self.out_layer = torch.nn.Softmax(dim=2)
        #geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))  todo: do we need this???
        self.d_tol = 0.002


        #create exp name based on experiment params
        self.loss_weight = {'wgt': opt['train']['wgt_wgt'], 'data':opt['train']['data_wgt'], 'spr_wgt': opt['train']['spr_wgt']}

        self.exp_name = '{}_{}_{}_{}_{}_{}_{}'.format(self.loss_weight['wgt'], self.loss_weight['data'], self.loss_weight['spr_wgt'], self.garment_layer, self.garment_class, self.feat, self.num_neigh )
        self.exp_path = '{}/{}/'.format( opt['experiment']['root_dir'], self.exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( self.exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(self.exp_name))

        self.val_min = None
        self.train_min = None
        self.loss = opt['train']['loss_type']
        self.n_part = opt['experiment']['num_part']
        self.loss_mse = torch.nn.MSELoss()

        self.batch_size=  opt['train']['batch_size']
        
        self.relu = nn.ReLU()
        #weight initialiser
        vert_id = self.vert_indices.cpu().numpy()
        init_weights = torch.from_numpy(np.array([layer_neigh[i] ==vert_id[i] for i in range(self.layer_size)]).astype('int64'))
        self.init_weight = torch.stack([init_weights for _ in range(self.batch_size)]).cuda()
        ######endddd####################

        ## train and val dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        ### load model and optimizer
        self.model = getattr(net_modules, opt['model']['name'])
        self.model = self.model(opt['model'], input_dim, output_dim).to(self.device)
        self.optimizer = getattr(optim, opt['train']['optimizer'])
        self.optimizer = self.optimizer(self.model.parameters(), opt['train']['optimizer_param'])




        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()



    def train_step(self,batch, ep=None):

        self.model.train()
        self.optimizer.zero_grad()

        loss, loss_dict = self.compute_loss(batch, ep)
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_dict

    def tranform_pts(self, pts, transform, W, trans=None):
        if trans is not None:
            pts = pts - trans.unsqueeze(1)

        #transform = torch.from_numpy(transform.astype(np.float32)).unsqueeze(0).cuda()
        #W = torch.from_numpy(weight_pred.astype(np.float32)).unsqueeze(0).cuda()
        T = torch.matmul(W, transform.view(transform.shape[0], 24, 16)).view(transform.shape[0], -1, 4, 4)
        Tinv = torch.inverse(T)
        verts_homo = torch.cat([pts, torch.ones(pts.shape[0], pts.shape[1], 1).cuda()], dim=2)
        transformed_pts = torch.matmul(Tinv, verts_homo.unsqueeze(-1))[:, :, :3, 0]

        return transformed_pts

    def compute_loss(self,batch,ep=None):
        inp = batch.get('inp').to(self.device)
        gt_verts = batch.get('gt_verts').to(self.device)
        betas = batch.get('betas').to(self.device)
        pose = batch.get('pose').to(self.device)
        trans = batch.get('trans').to(self.device)
        weights_from_net = self.model(inp).view(self.batch_size, self.layer_size, self.num_neigh)
        weights_from_net = self.out_layer(weights_from_net)


        loss_dict = {}
        pretrain = False
        if ep < 16:
            pretrain= True
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
            idx = torch.stack([torch.index_select(self.layer_neigh, 1, current_argmax[i])[0] for i in range(self.batch_size)])
            current_argmax_verts = torch.stack([torch.index_select(inp[i, :, :3], 0, idx[i]) for i in range(self.batch_size)])
            current_argmax_verts = torch.stack([current_argmax_verts for i in range(self.num_neigh)], dim=2)
            dist_from_max = current_argmax_verts - input_copy  # todo: should it be input copy??

            dist_from_max = torch.sqrt(torch.sum(dist_from_max * dist_from_max, dim=3))
            local_regu = torch.sum(dist_from_max * weights_from_net) / (self.batch_size * self.num_neigh * self.layer_size)

            body_tmp = self.smpl.forward(beta=betas, theta=pose, trans=trans)
            # body_mesh = [tm.from_tensors(vertices=v,
            #                              faces=self.smpl_faces) for v in body_tmp]

            if self.garment_layer == 'Body':
                # update body verts with prediction
                body_tmp[:, self.vert_indices, :] = pred_verts
                # get skin cutout
                loss_data = data_loss(self.garment_layer, pred_verts, inp[:, self.vert_indices, :], self.geo_weights)
            else:
                #loss_data = data_loss(self.garment_layer, pred_verts, gt_verts)
                loss_data, _ = chamfer_distance(pred_verts, gt_verts)
            # create mesh for predicted and smpl mesh
            #pred_mesh = Meshes(verts=[pred_verts], faces=[self.garment_f_torch.unsqueeze(0).repeat(self.batch_size,1,1)])
            pred_mesh = Meshes(verts=pred_verts, faces=self.garment_f_torch.unsqueeze(0).repeat(self.batch_size,1,1))
            # pred_mesh = [tm.from_tensors(vertices=v,
            #                              faces=self.garment_f_torch) for v in pred_verts]
            # gt_mesh = [tm.from_tensors(vertices=v,
            #                            faces=self.garment_f_torch) for v in gt_verts]

            #loss_lap = lap_loss(pred_mesh, gt_mesh)
            loss_lap = mesh_laplacian_smoothing(pred_mesh, method='uniform')
            # calculate normal for gt, pred and body
            #loss_norm, body_normals, pred_normals = normal_loss(self.batch_size, pred_mesh, gt_mesh, body_mesh, self.num_faces)
            #loss_edge = mesh_edge_loss(smpl_mesh_deformed)
            # interpenetration loss
            # loss_interp = interp_loss(self.sideddistance, self.relu, pred_verts, gt_verts, body_tmp, body_normals,
            #                           self.layer_size, d_tol=self.d_tol)

            loss = loss_data + 100. * loss_lap + local_regu #+ loss_interp  # loss_norm

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
                    loss_terms[k] += self.loss_weight[k]*loss_dict[k].item()
                    print("Current loss: {} {}  ".format(k, loss_dict[k].item()))

                sum_loss += loss
            batch_loss = sum_loss / len(train_data_loader)
            print("Current batch_loss: {} {}  ".format(epoch, batch_loss))

            for k in loss_dict.keys():
                loss_terms[k] = loss_dict[k]/ len(train_data_loader)
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
                self.writer.add_scalar('training loss {} avg'.format(k), loss_terms[k] , epoch)

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch':epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path,  _use_new_zipfile_serialization=False)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
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
            loss, _= self.compute_loss(batch, ep)
            sum_val_loss += loss.item()
        return sum_val_loss /len(val_data_loader)

