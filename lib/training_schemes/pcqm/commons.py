import json
import os.path as osp
import torch
import torch.nn.functional as F

@torch.jit.script
def coords2dist(coords):
    return torch.norm(coords.unsqueeze(-2) - coords.unsqueeze(-3), dim=-1)

def add_coords_noise(coords, edge_mask, noise_level, noise_smoothing):
    noise = coords.new(coords.size()).normal_(0, noise_level)
    dist_mat = coords2dist(coords).add_((1-edge_mask.float())*1e9)
    smooth_mat = torch.softmax(-dist_mat/noise_smoothing, -1)
    noise = torch.matmul(smooth_mat, noise)
    coords = coords + noise
    return coords


def discrete_dist(dist, num_bins, range_bins):
    dist = dist * ((num_bins - 1) / range_bins)
    dist = dist.long().clamp(0, num_bins - 1)
    return dist


class DiscreteDistLoss:
    def __init__(self, num_bins, range_bins):
        self.num_bins = num_bins
        self.range_bins = range_bins
    
    def __call__(self, dist_logits, dist_targ, mask, reduce=True):
        num_bins = self.num_bins
        range_bins = self.range_bins
        bsize = dist_logits.size(0)
        
        dist_targ = discrete_dist(dist_targ, num_bins, range_bins)
        dist_logits = dist_logits.view(-1, num_bins)
        dist_targ = dist_targ.view(-1)
        xent = F.cross_entropy(dist_logits, dist_targ, reduction='none')
        
        xent = xent.view(bsize, -1)
        mask = mask.to(xent.dtype).view(bsize, -1)
        
        if reduce:
            xent = (xent * mask).sum() / (mask.sum() + 1e-9)
        else:
            xent = (xent * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        
        return xent



class BinsProcessor:
    def __init__(self, path, 
                 shift_half=True,
                 zero_diag=True,):
        self.path = path
        self.shift_half = shift_half
        self.zero_diag = zero_diag
        
        self.data_path = osp.join(path, 'data')
        self.meta_path = osp.join(path, 'meta.json')
        
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
        
        self.num_samples = self.meta['num_samples']
        self.num_bins = self.meta['num_bins']
        self.range_bins = self.meta['range_bins']
        
        self.bin_size = self.range_bins / (self.num_bins-1)
    
    def bins2dist(self, bins):
        bins = bins.float()
        if self.shift_half:
            bins = bins + 0.5
        dist = bins * self.bin_size
        dist = dist + dist.transpose(-2,-1)
        if self.zero_diag:
            dist = dist * (1 - torch.eye(dist.size(-1),
                                         dtype=dist.dtype,
                                         device=dist.device))
        return dist
