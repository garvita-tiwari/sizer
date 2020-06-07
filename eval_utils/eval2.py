import torch
import numpy as np
import sys
import pickle as pkl
import ipdb
from kaolin.metrics.mesh import point_to_surface, laplacian_loss, TriangleDistance
from psbody.mesh import Mesh, MeshViewer
import os
sys.path.append('/BS/garvita/work/code/sizer')

from models import network_layers
from models.eval import AverageMeter, verts_dist
from data_utils.mesh_dataloader import ParserData
from data_utils.geomtery import get_vertex_normals, nearest_map
sys.path.append('/BS/garvita/work/libs/core_gt')
from log_utils import sio

sys.path.append('/BS/garvita/work/code/3D_humans_dataset')
from lib.smpl_paths import SmplPaths
from lib.th_SMPL import th_batch_SMPL

from globar_var import body_points, num_gar_point


device = torch.device("cuda:0")

def eval_real_data():
    from trainer.parsernet import Runner, get_model

    LOG_DIR = '/scratch/BS/pool1/garvita/lap/real_g5/UpperClothes_vn'
    eval_runner = get_model(LOG_DIR)

    #read data
    rp_data = pkl.load(open('/BS/garvita2/static00/sizer_eval/tshirt_apose.pkl', 'rb'),
                encoding="latin1")
    reg = np.array(rp_data['reg'])

    #getvertex normals
    

    theta_gt = np.array(rp_data['pose']).astype(np.float32)
    beta_gt = np.array(rp_data['betas'])[:,:10].astype(np.float32)
    trans_gt = np.array(rp_data['trans']).astype(np.float32)
    reg =  np.array(rp_data['reg']).astype(np.float32)[:, :body_points, :]

    layer_mesh = np.array(rp_data['UpperClothes']).astype(np.float32)[:, :num_gar_point, :]
    data_names = rp_data['data_name']

    theta_torch = torch.from_numpy(theta_gt).cuda()
    beta_torch = torch.from_numpy(beta_gt).cuda()
    trans_torch = torch.from_numpy(trans_gt).cuda()
    reg =  torch.from_numpy(reg).cuda()
    layer_mesh =  torch.from_numpy(layer_mesh).cuda()
    #get vertex normal for input mesh
    vert_nrm = get_vertex_normals(reg.cpu(), eval_runner.smpl_faces.cpu()).cuda()
    inp = torch.cat((reg,vert_nrm),2)
    gar, smpl_verts, gar_faces, smpl_faces, dist = eval_runner.forward(inp, beta_torch, theta_torch, trans_torch, layer_mesh)
    #print(dist)
    layer_mesh = np.array(rp_data['UpperClothes']).astype(np.float32)[:, :num_gar_point, :]

    #create resulting mesh
    for i in range(theta_gt.shape[0]):
        gar_mesh = Mesh(v=gar[i],f=gar_faces)
        gar_mesh.write_ply(os.path.join(LOG_DIR, 'eval/{}pred.ply'.format(data_names[i])))

        gar_mesh = Mesh(v=layer_mesh[i],f=gar_faces)
        gar_mesh.write_ply(os.path.join(LOG_DIR, 'eval/{}_gt.ply'.format(data_names[i])))

        gar_mesh = Mesh(v=smpl_verts[i],f=smpl_faces)
        gar_mesh.write_ply(os.path.join(LOG_DIR, 'eval/{}_smpl.ply'.format(data_names[i])))
        print(data_names[i], dist[i])
if __name__ == "__main__":
    eval_real_data()
