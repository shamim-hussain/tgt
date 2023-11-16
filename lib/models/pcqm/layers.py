
import math

import torch
from torch import nn
import torch.nn.functional as F

from . import consts as C
from lib.egt import Graph

class EmbedInput(nn.Module):
    def __init__(self,
                 node_width,
                 edge_width,
                 upto_hop            = 32        ,
                 embed_3d_type       = 'gaussian',
                 num_3d_kernels      = 128       ,
                 ):
        super().__init__()
        
        self.node_width         = node_width
        self.edge_width         = edge_width
        self.upto_hop           = upto_hop
        self.num_3d_kernels     = num_3d_kernels
        self.embed_3d_type      = embed_3d_type
        
        self.nodef_embed = nn.Embedding(C.NUM_NODE_FEATURES*C.NODE_FEATURES_OFFSET+1,
                                        self.node_width, padding_idx=0)
        
        self.dist_embed = nn.Embedding(self.upto_hop+2, self.edge_width)
        self.featm_embed = nn.Embedding(C.NUM_EDGE_FEATURES*C.EDGE_FEATURES_OFFSET+1,
                                        self.edge_width, padding_idx=0)
        
        if self.embed_3d_type == 'gaussian':
            self.m3d_embed = Gaussian3DEmbed(self.edge_width,
                                            2*C.NODE_FEATURES_OFFSET+1,
                                            self.num_3d_kernels)
            self._node_j_offset = C.NODE_FEATURES_OFFSET
        elif self.embed_3d_type == 'fourier':
            self.m3d_embed = Fourier3DEmbed(self.edge_width,
                                            self.num_3d_kernels)
        elif self.embed_3d_type != 'none':
            raise ValueError('Invalid 3D embedding type')
        
        self._uses_3d = (self.embed_3d_type != 'none')
        

    def embed_3d_dist(self, dist_input, nodef):
        if self.embed_3d_type == 'gaussian':
            num_nodes = nodef.size(1)
            nodes_i = nodef[:,:,0]                                  # (b,i)
            nodes_j = nodes_i + self._node_j_offset
            nodes_i = nodes_i.unsqueeze(2).expand(-1,-1,num_nodes)  # (b,i,j)
            nodes_j = nodes_j.unsqueeze(1).expand(-1,num_nodes,-1)  # (b,i,j)
            nodes_ij = torch.stack([nodes_i,nodes_j], dim=-1)       # (b,i,j,2)
            return self.m3d_embed(dist_input, nodes_ij)
        elif self.embed_3d_type == 'fourier':
            return self.m3d_embed(dist_input)
        else:
            raise ValueError('Invalid 3D embedding type')
            
    def forward(self, inputs):
        g = Graph(inputs)
        
        nodef = g.node_features.long()              # (b,i,f)
        h = self.nodef_embed(nodef).sum(dim=2)      # (b,i,w,h) -> (b,i,h)
        
        dm0 = g.distance_matrix                     # (b,i,j)
        dm = dm0.long().clamp(max=self.upto_hop+1)  # (b,i,j)
        featm = g.feature_matrix.long()             # (b,i,j,f)
        
        e = self.dist_embed(dm) \
                + self.featm_embed(featm).sum(dim=-2)  # (b,i,j,f,e) -> (b,i,j,e)
        
        if self._uses_3d:
            e = e + self.embed_3d_dist(g.dist_input, nodef)
        
        mask_dtype = e.dtype
        edge_mask = g.edge_mask.unsqueeze(-1).to(mask_dtype)
        mask = (1-edge_mask)*torch.finfo(mask_dtype).min
        
        g.h, g.e, g.mask = h, e, mask
        return g


class Fourier3DEmbed(nn.Module):
    def __init__(self, num_heads, num_kernel,
                 min_dist=0.01, max_dist=20):
        assert num_kernel % 2 == 0
        
        super().__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel
        self.min_dist = min_dist
        self.max_dist = max_dist
        
        wave_lengths = torch.exp(torch.linspace(math.log(2*min_dist),
                                                math.log(2*max_dist),
                                                num_kernel // 2))
        angular_freqs = 2 * math.pi / wave_lengths
        self.register_buffer('angular_freqs', angular_freqs)
        
        self.proj = nn.Linear(num_kernel, num_heads)
    
    def forward(self, dist):
        phase = dist.unsqueeze(-1) * self.angular_freqs
        sinusoids = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        out = self.proj(sinusoids)
        return out


class Gaussian3DEmbed(nn.Module):
    def __init__(self, num_heads, num_edges, num_kernel):
        super(Gaussian3DEmbed, self).__init__()
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.num_kernel = num_kernel

        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.gbf_proj = NonLinear(self.num_kernel, self.num_heads)


    def forward(self, dist, node_type_edge):
        edge_feature = self.gbf(dist, node_type_edge.long())
        gbf_result = self.gbf_proj(edge_feature)
        return gbf_result



@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512*3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

