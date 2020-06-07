import os
import sys
import tensorboardX
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle
import ipdb

sys.path.append('/BS/garvita/work/code/sizer')



from models import network_layers
from models.eval import AverageMeter, verts_dist
from data_utils.mesh_dataloader import ParserData

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
        self.layer_siz = 1000
        self.bs = params['batch_size']
        self.vis_freq = params['vis_freq']
        self.model_name = params['model_name']
        self.note = params['note']

        # log and backup
        LOG_DIR = '/scratch/BS/pool1/garvita/tmp'
        ROOT_DIR = '/scratch/BS/pool1/garvita/tmp'
        log_name = os.path.join(params['log_name'], self.garment_class)
        self.log_dir = sio.prepare_log_dir(LOG_DIR, ROOT_DIR,log_name)
        sio.save_params(self.log_dir, params, save_name='params')

        self.iter_nums = 0 if 'iter_nums' not in params else params['iter_nums']

        #load smpl data
        self.smpl = SmplPaths(gender=self.gender)
        body_vert = range(6890)
        self.smpl_size = len(body_vert)
        input_dim = self.smpl_size * 3
        self.layer_size = 1000
        self.layer = 'UpperClothes'
        output_dim = self.layer_size * 3
        # get active vert id
        self.num_neigh = 20
        res_name = 'lres'
        layer_neigh = np.array(np.load("/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_neighborheuristics_{}_{}_{}_gar_order2.npy".format(self.garment_class, res_name, self.layer, self.num_neigh)))
        self.layer_neigh = layer_neigh
        #class_info = pickle.load(open(os.path.join(DATA_DIR, 'garment_class_info.pkl'), 'rb'), encoding="latin1")

        self.body_f_np = self.smpl.get_faces()
        self.garment_f_np = np.random.rand(1000,3)
        self.garment_f_torch = torch.tensor(self.garment_f_np.astype(np.long)).long().to(device)
        self.vert_indices = None

        # dataset and dataloader
        self.train_dataset = ParserData(params['garment_class'], split='train',
                                             gender=self.gender)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, num_workers=12, shuffle=True,
                                       drop_last=True if len(self.train_dataset) > self.bs else False)

        self.test_dataset = ParserData(params['garment_class'], split='test')
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, num_workers=12, shuffle=False,
                                      drop_last=False)

        print(len(self.train_dataset))
        print(len(self.test_dataset))

        # models and optimizer
        self.model = getattr(network_layers, self.model_name)(input_size=input_dim, output_size=output_dim)

        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=1e-6)

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
        #self.csv_logger = PSS2GLogger()

    def get_loss(self, pred_verts, gt_verts, tttype, epoch):

        # data_loss = ((pred_verts - gt_verts) ** 2).sum(-1).mean()
        data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()

        return (data_loss
                # + interp_loss
                # + 10*lapl_loss
                # + normal_loss * ( np.e ** (epoch/10-10) )
                )

    def train(self, epoch):
        epoch_loss = AverageMeter()
        self.model.train()
        for i, (inp, gt_verts, _) in enumerate(self.train_loader):
            gt_verts = gt_verts.to(device)
            inp = inp.to(device)
            self.optimizer.zero_grad()
            # pred_verts = self.models(torch.cat((thetas, betas, gammas), dim=1)).view(gt_verts.shape) + linear_pred
            pred_verts = self.model(inp).view(gt_verts.shape)
            loss = self.get_loss(pred_verts, gt_verts, "train", epoch)
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
            for i, (inp, gt_verts,  idxs) in enumerate(self.test_loader):
                gt_verts = gt_verts.to(device)
                inp = inp.to(device)
                idxs = idxs.numpy()

                pred_verts = self.model(inp).view(gt_verts.shape)

                loss = self.get_loss(pred_verts, gt_verts, "val", epoch)
                dist = verts_dist(gt_verts, pred_verts) * 1000.
                val_loss.update(loss.item(), gt_verts.shape[0])
                val_dist.update(dist.item(), gt_verts.shape[0])

                # for lidx, idx in enumerate(idxs):
                #     if idx % self.vis_freq != 0:
                #         continue
                #     pred_vert = pred_verts[lidx].cpu().numpy()
                #     gt_vert = gt_verts[lidx].cpu().numpy()
                #     # linear_vert = linear_pred[lidx].cpu().numpy()
                #
                #     body_m, pred_m = self.smpl.run(theta=theta, garment_d=pred_vert, beta=beta,
                #                                    garment_class=self.garment_class)
                #     _, gt_m = self.smpl.run(theta=theta, garment_d=gt_vert, beta=beta,
                #                             garment_class=self.garment_class)
                #     # _, linear_m = self.smpl.run(theta=theta, garment_d=linear_vert, beta=beta,
                #     # _, linear_m = self.smpl.run(theta=self.apose, garment_d=linear_vert, beta=beta,
                #     #                             garment_class=self.garment_class)
                #
                #     save_dir = os.path.join(self.log_dir, "{:04d}".format(epoch))
                #     pred_m.write_ply(os.path.join(save_dir, "pred_{}.ply".format(idx)))
                #     gt_m.write_ply(os.path.join(save_dir, "gt_{}.ply".format(idx)))
                #     # linear_m.write_ply(os.path.join(save_dir, "linear_pred_{}.ply".format(idx)))
                #     body_m.write_ply(os.path.join(save_dir, "body_{}.ply".format(idx)))

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

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_config', default='')

    parser.add_argument('--garment_class', default="real_g5")
    parser.add_argument('--gender', default="female")

    parser.add_argument('--vis_freq', default=16, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--max_epoch', default=800, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--checkpoint', default="")

    parser.add_argument('--model_name', default="FC_correspondence_lres")
    parser.add_argument('--num_layers', default=3)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--note', default="parsernet fc correspondence lres")

    parser.add_argument('--log_name', default="real_g5_parsernet")
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
            if i % 10 == 0:
                trainer.validate(i)
    finally:
        # else:
        trainer.write_log()
        print("safely quit!")
