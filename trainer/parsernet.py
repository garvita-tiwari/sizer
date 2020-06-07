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

from kaolin.metrics.mesh import point_to_surface, laplacian_loss, TriangleDistance
from kaolin.rep import TriangleMesh as tm

sys.path.append('/BS/garvita/work/code/sizer')

from models import network_layers
from models.eval import AverageMeter, verts_dist
from data_utils.mesh_dataloader import ParserData
from data_utils.geomtery import get_vertex_normals, nearest_map, get_res_vert, get_vid
sys.path.append('/BS/garvita/work/libs/core_gt')
from log_utils import sio

sys.path.append('/BS/garvita/work/code/3D_humans_dataset')
from lib.smpl_paths import SmplPaths
from lib.th_SMPL import th_batch_SMPL

device = torch.device("cuda:0")

class Trainer(object):
    def __init__(self, params):
        self.params = params
        self.gender = params['gender']
        self.garment_class = params['garment_class']
        self.bs = params['batch_size']
        self.vis_freq = params['vis_freq']
        self.model_name = params['model_name']
        self.note = params['note']
        self.layer = params['garment_layer']
        self.vc = False
        self.vn = True
        # log and backup
        LOG_DIR = '/scratch/BS/pool1/garvita/lap'
        ROOT_DIR = '/scratch/BS/pool1/garvita/lap'
        #log_name = os.path.join(params['log_name'], self.garment_class)
        feat = 'vn'
        if self.vc:
            feat = 'vc_vn'
        log_name = os.path.join(self.garment_class, '{}_{}'.format(self.layer, feat))

        self.log_dir = sio.prepare_log_dir(LOG_DIR, ROOT_DIR,log_name)
        sio.save_params(self.log_dir, params, save_name='params')

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        #load smpl data
        self.smpl = SmplPaths(gender=self.gender)

        self.layer_size, self.smpl_size = get_res_vert(params['garment_class'],False, params['garment_layer'] )
        if self.layer == 'Body':
            self.layer_size = 4448

        # get active vert id
        self.num_neigh = 20
        input_dim = self.smpl_size * 3
        if self.vn:
            input_dim = self.smpl_size * 6

        output_dim = self.layer_size *  self.num_neigh

        res_name = 'lres'
        layer_neigh = np.array(np.load("/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_neighborheuristics_{}_{}_{}_gar_order2.npy".format(self.garment_class, res_name, self.layer, self.num_neigh)))
        self.layer_neigh = layer_neigh

        #separate for body layer
        body_vert = range(6890)
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
        if self.layer == 'Body':
            self.idx2 = torch.from_numpy(self.neigh_id2).view(len(self.body_vert), self.num_neigh).cuda()
        else:
            self.idx2 = torch.from_numpy(self.neigh_id2).view(self.layer_size, self.num_neigh).cuda()



        #get vert indixed of layer

        self.vert_indices = get_vid(self.layer, self.garment_class,False)

        self.vert_indices = torch.tensor(self.vert_indices.astype(np.long)).long().cuda()
        # dataset and dataloader
        self.train_dataset = ParserData(params['garment_class'], split='train',
                                             gender=self.gender, vn=self.vn)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                       drop_last=True if len(self.train_dataset) > self.bs else False)

        self.test_dataset = ParserData(params['garment_class'], split='test', vn=self.vn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, num_workers=12, shuffle=False,
                                      drop_last=False)

        #create smpl
        sp = SmplPaths(gender="male")
        smpl_faces = sp.get_faces()
        self.smpl_faces_np = smpl_faces
        self.smpl_faces = torch.tensor(smpl_faces.astype('float32'), dtype=torch.long).cuda()

        #interpenetraion loss term
        self.nearest_pt = TriangleDistance()
        self.body_f_np = self.smpl.get_faces()

        if self.layer == 'Body':
            self.garment_f_np = self.body_f_np
            self.garment_f_torch = self.smpl_faces
        else:
            self.garment_f_np = Mesh(
                filename='/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_lres_{}.obj'.format(self.garment_class, self.layer)).f

            self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)
        # models and optimizer
        self.model = getattr(network_layers, self.model_name)(input_size=input_dim, output_size=output_dim)

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

        geo_weights = np.load('/BS/garvita2/static00/ClothSize_data/gcn_assets/weights/real_g5_geo_weights.npy')
        self.geo_weights = torch.tensor(geo_weights[body_vert2].astype(np.float32)).cuda()
        self.best_error = np.inf
        self.best_epoch = -1

        self.logger = tensorboardX.SummaryWriter(os.path.join(self.log_dir))


        #self.csv_logger = PSS2GLogger()

    def get_loss(self, pred_verts, gt_verts, tttype, epoch):


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
        for i, (inp, gt_verts, betas, pose, trans, _) in enumerate(self.train_loader):
            gt_verts = gt_verts.to(device)
            inp = inp.to(device)
            self.optimizer.zero_grad()
            # pred_verts = self.models(torch.cat((thetas, betas, gammas), dim=1)).view(gt_verts.shape) + linear_pred
            weights_from_net = self.model(inp).view(self.bs, self.layer_size, self.num_neigh)
            weights_from_net = self.out_layer(weights_from_net)
            #normalise weights
            #weight_norm = torch.norm(weights_from_net, p=2, dim=2, keepdim=True)
            #weights_from_net = weights_from_net.div(weight_norm.expand_as(weights_from_net))
            #make a copy of neighbour for each vertex
            input_copy = inp[:,self.idx2,:]
            pred_x = weights_from_net * input_copy[:,:,:,0]
            pred_y = weights_from_net * input_copy[:,:,:,1]
            pred_z = weights_from_net * input_copy[:,:,:,2]

            pred_verts = torch.sum(torch.stack((pred_x, pred_y, pred_z), axis=3), axis=2)

            #local neighbourhood regulaiser
            #todo: change this to attention
            #ipdb.set_trace()
            #current_argmax = torch.argmax(weights_from_net, axis=2)
            # create smpl for finding interpenetration
            smpl = th_batch_SMPL(self.bs, betas, pose,trans, faces=self.smpl_faces, gender=self.gender).cuda()
            smpl_verts, _, _, _ = smpl()

            if self.layer == 'Body':
                body_tmp = smpl_verts
                #update body verts with prediction
                body_tmp[:,self.vert_indices,:] = pred_verts
                #get skin cutout

                loss_data = self.get_loss(pred_verts, inp[:, self.vert_indices, :], "train", epoch)


                # create mesh for predicted and smpl mesh
                pred_mesh = [tm.from_tensors(vertices=v,
                                             faces=self.smpl_faces) for v in body_tmp]

                gt_mesh = [tm.from_tensors(vertices=v,
                                           faces=self.smpl_faces) for v in smpl_verts]

                lap_loss = torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh, gt_mesh)]).mean()
            else:
                loss_data = self.get_loss(pred_verts, gt_verts, "train", epoch)
                pred_mesh = [tm.from_tensors(vertices=v,
                                                  faces=self.garment_f_torch) for v in pred_verts]

                gt_mesh = [tm.from_tensors(vertices=v,
                                                  faces=self.garment_f_torch) for v in gt_verts]

                lap_loss = torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh, gt_mesh)]).mean()
            loss = loss_data + 100.*lap_loss
            loss.backward()
            self.optimizer.step()

            self.logger.add_scalar("train/loss", loss.item(), self.iter_nums)
            print("Iter {}, loss: {:.8f}".format(self.iter_nums, loss.item()))
            epoch_loss.update(loss, gt_verts.shape[0])
            self.iter_nums += 1

        self.logger.add_scalar("train_epoch/loss", epoch_loss.avg, epoch)
        # self._save_ckpt(epoch)

    def validate(self, epoch):
        val_loss = AverageMeter()
        val_dist = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (inp, gt_verts, betas, pose, trans, idxs) in enumerate(self.test_loader):
                gt_verts = gt_verts.to(device)
                inp = inp.to(device)
                idxs = idxs.numpy()
                weights_from_net = self.model(inp).view(self.bs, self.layer_size, self.num_neigh)
                # normalise weights
                #weight_norm = torch.norm(weights_from_net, p=2, dim=2, keepdim=True)
                #weights_from_net = weights_from_net.div(weight_norm.expand_as(weights_from_net))
                weights_from_net = self.out_layer(weights_from_net)

                # make a copy of neighbour for each vertex
                input_copy = inp[:, self.idx2, :]
                pred_x = weights_from_net * input_copy[:, :, :, 0]
                pred_y = weights_from_net * input_copy[:, :, :, 1]
                pred_z = weights_from_net * input_copy[:, :, :, 2]

                pred_verts = torch.sum(torch.stack((pred_x, pred_y, pred_z), axis=3), axis=2)

                # create smpl for finding interpenetration
                smpl = th_batch_SMPL(self.bs, betas, pose, trans, faces=self.smpl_faces, gender=self.gender).cuda()
                smpl_verts, _, _, _ = smpl()

                if self.layer == 'Body':
                    loss = self.get_loss(pred_verts, inp[:, self.vert_indices, :], "val", epoch)
                    dist = verts_dist(pred_verts, inp[:, self.vert_indices, :]) * 1000.
                    val_loss.update(loss.item(), gt_verts.shape[0])
                    val_dist.update(dist.item(), gt_verts.shape[0])
                else:
                    loss = self.get_loss(pred_verts, gt_verts, "val", epoch)
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

                    if self.layer == 'Body':
                        body_tmp = smpl_verts[lidx].cpu().numpy()
                        # update body verts with prediction
                        body_tmp[self.vert_indices.cpu().numpy(), :] = pred_vert
                        pred_vert = body_tmp
                        gt_vert = inp[lidx].cpu().numpy()

                    # # linear_vert = linear_pred[lidx].cpu().numpy()
                    #
                    # body_m, pred_m = self.smpl.run(theta=theta, garment_d=pred_vert, beta=beta,
                    #                                garment_class=self.garment_class)
                    # _, gt_m = self.smpl.run(theta=theta, garment_d=gt_vert, beta=beta,
                    #                         garment_class=self.garment_class)
                    # # _, linear_m = self.smpl.run(theta=theta, garment_d=linear_vert, beta=beta,
                    # # _, linear_m = self.smpl.run(theta=self.apose, garment_d=linear_vert, beta=beta,
                    # #                             garment_class=self.garment_class)
                    pred_m = Mesh(v=pred_vert, f=self.garment_f_np)
                    gt_m = Mesh(v=gt_vert, f=self.garment_f_np)
                    save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
                    pred_m.write_ply(os.path.join(save_dir, "pred_{}.ply".format(idx)))
                    gt_m.write_ply(os.path.join(save_dir, "gt_{}.ply".format(idx)))
                    body_mesh.write_ply(os.path.join(save_dir, "smpl_{}.ply".format(idx)))
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
        self.layer_size = get_res_vert(params['garment_class'],False, params['garment_layer'] )[0]
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

    parser.add_argument('--garment_class', default="real_g5")
    parser.add_argument('--garment_layer', default="UpperClothes")
    parser.add_argument('--gender', default="male")

    parser.add_argument('--vis_freq', default=16, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    parser.add_argument('--model_name', default="FC_correspondence_lres")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--note', default="parsernet fc correspondence lres")

    #parser.add_argument('--log_name', default="real_g5_parsernet")
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
