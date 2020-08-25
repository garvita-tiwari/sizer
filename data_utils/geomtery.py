import numpy as np
from psbody.mesh import Mesh
import torch
import ipdb
import sys
import pickle as pkl
import os
from kaolin.rep import TriangleMesh as tm
from kaolin.metrics.mesh import point_to_surface, laplacian_loss, TriangleDistance


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