import torch
import ipdb
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def batched_index_select(inp, dimn, idx):
    ipdb.set_trace()
    views = [inp.shape[0]] + [1 if i != dimn else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dimn] = -1
    idx = idx.view(views).expand(expanse)
    return torch.gather(inp, dimn, idx)


def nearest_neighbour(pt1, pt2):
    # get the product x * y
    # here, y = x.t()
    r = torch.mm(mat, mat.t())
    # get the diagonal elements
    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)
    # compute the distance matrix
    D = diag + diag.t() - 2*r
    return D.sqrt()