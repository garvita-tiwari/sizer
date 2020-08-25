import numpy as np
from psbody.mesh import Mesh
import torch
import ipdb
import sys
import pickle as pkl
import os
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import point_to_surface, laplacian_loss, TriangleDistance

sys.path.append('/BS/garvita/work/libs/core_gt/pytorch_utils')
from mesh_dist import point_to_surface_vec

def get_face_normals(verts, faces):
    num_batch = verts.size(0)
    num_faces = faces.size(0)

    # faces by vertices
    fbv = torch.index_select(verts, 1, faces.view(-1)).view(num_batch, num_faces, 3, 3)
    normals = torch.cross(fbv[:, :, 1] - fbv[:, :, 0], fbv[:, :, 2] - fbv[:, :, 0], dim=2)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1.e-10)
    return normals

def get_vertex_normals(verts, faces, ret_face_normals=False):
    num_faces = faces.size(0)
    num_verts = verts.size(1)
    face_normals = get_face_normals(verts, faces)

    FID = torch.arange(num_faces).unsqueeze(1).repeat(1, 3).view(-1)
    VID = faces.view(-1)
    data = torch.ones_like(FID, dtype=torch.float32)

    mat = torch.sparse_coo_tensor(
        indices=torch.stack((VID, FID)),
        values=data,
        size=(num_verts, num_faces)
    )
    degree = torch.sparse.sum(mat, dim=1).to_dense()
    vertex_normals = torch.stack((
        torch.sparse.mm(mat, face_normals[:, :, 0].t()),
        torch.sparse.mm(mat, face_normals[:, :, 1].t()),
        torch.sparse.mm(mat, face_normals[:, :, 2].t()),
    ), dim=-1)
    vertex_normals = vertex_normals.transpose(1, 0) / degree.unsqueeze(0).unsqueeze(-1)
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=-1, keepdim=True) + 1.e-10)

    if ret_face_normals:
        return vertex_normals, face_normals
    else:
        return vertex_normals


def extract_tri_from_mesh(vertices, faces):
    v1 = torch.index_select(vertices.clone(), 1, faces[:, 0])
    v2 = torch.index_select(vertices.clone(), 1, faces[:, 1])
    v3 = torch.index_select(vertices.clone(), 1, faces[:, 2])

    return v1, v2, v3

def nearest_map(body_verts, gar_verts, gt_verts, body_faces, gar_faces):
    v1 = torch.index_select(body_verts.clone(), 1, body_faces[:, 0])
    v2 = torch.index_select(body_verts.clone(), 1, body_faces[:, 1])
    v3 = torch.index_select(body_verts.clone(), 1, body_faces[:, 2])

    tri_minimum_dist = TriangleDistance()
    distance, indx, dist_type = tri_minimum_dist(gar_verts, v1, v2, v3)
    ipdb.set_trace()
    return v1, v2, v3


def loss_interp(body_verts, gar_verts, gt_verts, body_faces, gar_faces):

    bs, n, _ = gar_verts.size()

    th_smpl_meshes = [tm.from_tensors(vertices=v,
                                      faces=body_faces)for v in body_verts]

    #calculate distance of each point in garment mesh to body mesh distance
    gar2body = torch.stack([point_to_surface_vec(gar_verts[b, i, :],th_smpl_meshes[b] ) for b in range(bs) for i in range(n)])

    ipdb.set_trace()

    gar2body = gar2body.view(bs,n)

    ipdb.set_trace()
    #random index for  testig
    nearest_idx= torch.LongTensor(bs,n)
    body_normals = get_vertex_normals(body_verts, body_faces)
    #todo: tolerence based indexing
    print("here", gar2body.shape)



def get_res_vert(garment='ShirtNoCoat', hres=True, garment_type='UpperClothes'):
    res= 'lres'
    body_vert = 6890
    if hres:
        res ='hres'
        body_vert = 27554

    num_verts = {'ShirtNoCoat_hres': 10464, 'ShirtNoCoat_lres':2618, 'TShirtNoCoat_hres': 8168, 'TShirtNoCoat_lres': 2038,
                 'g1__hres':9723, 'g5_hres_UpperClothes': 7702,  'g5_lres_UpperClothes': 1924, 'g5_lres_Pants': 2710,
                 'g5_hres_Pants': 2710, 'g5_lres_Body': 6890,  'g1_lres_UpperClothes': 2438,
                 'g6_hres_Pants': 2710, 'g6_lres_Body': 6890,   'g6_lres_UpperClothes': 1924, 'g6_lres_Pants': 2710,
                 'g1_lres_Pants': 4718,'g3_lres_Pants': 4718, 'g3_lres_UpperClothes': 2530,
                 'g4_lres_Pants': 4718, 'g4_lres_UpperClothes': 2530,
                 'g7_lres_Pants': 2710, 'g7_lres_UpperClothes': 4758,
                 'g7_hres_Pants': 2710, 'g7_hres_UpperClothes': 4758,
                 'g1_lres_Body': 6890,  'g2_lres_Body': 6890,  'g3_lres_Body': 6890,
                 'g4_lres_Body': 6890,  'g7_lres_Body': 6890,  'g1__lres':2438, 'Pants_hres': 4041}
    return num_verts['{}_{}_{}'.format(garment, res,garment_type)], body_vert

def get_vid(garment, gar, hres=True):

    #default SHirtNoCoat
    vert_id = pkl.load(open(
        '/BS/bharat-2/static00/renderings/renderpeople_rigged/rp_kumar_rigged_002_zup_a/temp16/UpperClothes.pkl', 'rb'), encoding="latin1")[
        'vert_indices']  # high res v_id

    res='lres'
    if hres:
        res = 'hres'
    if gar == 'real_g6':
        gar = 'real_g5'
    vert_difile = '/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_{}_vertid_{}.npy'.format(gar, res, garment)
    if os.path.exists(vert_difile):
        vert_id = np.load(vert_difile)

    return vert_id