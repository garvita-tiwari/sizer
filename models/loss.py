import torch.nn as nn
import torch


def data_loss(garment_layer, pred_verts, gt_verts, geo_weights=None):

    if garment_layer == 'Body':
        data_loss = (pred_verts - gt_verts).abs().sum(-1) * geo_weights
        data_loss = data_loss.mean()
    else:
        data_loss = (pred_verts - gt_verts).abs().sum(-1).mean()

    return data_loss


def verts_dist(v1, v2, dim=None):
    """
    distance between two point sets
    v1 and v2 shape: NxVx3
    """
    x = torch.pow(v2 - v1, 2)
    x = torch.sum(x, -1)
    x = torch.sqrt(x)
    if dim == -1:
        return x
    elif dim is None:
        return torch.mean(x)
    else:
        return torch.mean(x, dim=dim)


def lap_loss(pred_mesh, gt_mesh):

    return  torch.stack([laplacian_loss(sc, sm) for sc, sm in zip(pred_mesh, gt_mesh)]).mean()

def normal_loss(bs, pred_mesh, gt_mesh, body_mesh, num_faces):
    body_normals = []
    gt_normals = []
    pred_normals = []
    for i in range(bs):
        b_normal = body_mesh[i].compute_face_normals()
        body_normals.append(b_normal)
        gt_nromal = gt_mesh[i].compute_face_normals()
        pred_normal = pred_mesh[i].compute_face_normals()
        gt_normals.append(gt_nromal)
        pred_normals.append(pred_normal)

    body_normals = torch.stack(body_normals)
    gt_normals = torch.stack(gt_normals)
    pred_normals = torch.stack(pred_normals)
    loss_norm = torch.sum(torch.sum((1 - gt_normals) * pred_normals, dim=2).abs()) / (bs * num_faces)

    return loss_norm, body_normals, pred_normals

def interp_loss(sideddistance, relu, pred_verts, gt_verts,  body_tmp, body_normals, layer_size, d_tol=0.001 ):
    bs = len(pred_verts)
    dist1 = sideddistance(pred_verts, body_tmp)
    nearest_body_verts = torch.stack([body_tmp[i][dist1[i]] for i in range(bs)])
    nearest_body_normals = torch.stack([body_normals[i][dist1[i]] for i in range(bs)])
    # body normal
    # select vertices which are near to gt
    vert_dist = pred_verts - gt_verts
    vert_dist = torch.sqrt(torch.sum(vert_dist * vert_dist, dim=2))
    # find active indices, which are near to gt prediction
    active_id = vert_dist < d_tol

    # calculate interp loss
    loss_interp = relu(torch.sum((nearest_body_verts - pred_verts) * nearest_body_normals, dim=2))
    loss_interp = torch.sum(active_id * loss_interp) / (bs * layer_size)

    return loss_interp