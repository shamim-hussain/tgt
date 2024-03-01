import torch
from torch import nn
import torch.nn.functional as F

from lib.tgt import TGT_Encoder

from . import layers

class TGT_Distance(nn.Module):
    def __init__(self,
                 model_height,
                 layer_multiplier    = 1         ,
                 upto_hop            = 32        ,
                 embed_3d_type       = 'gaussian',
                 num_3d_kernels      = 128       ,
                 num_dist_bins       = 128       ,
                 **layer_configs
                 ):
        super().__init__()
        
        self.model_height        = model_height
        self.layer_multiplier    = layer_multiplier
        self.upto_hop            = upto_hop
        self.embed_3d_type       = embed_3d_type
        self.num_3d_kernels      = num_3d_kernels
        self.num_dist_bins       = num_dist_bins
        
        self.node_width          = layer_configs['node_width']
        self.edge_width          = layer_configs['edge_width']
        
        self.layer_configs = layer_configs
        self.encoder = TGT_Encoder(model_height     = self.model_height      ,
                                   layer_multiplier = self.layer_multiplier  ,
                                   node_ended       = False                  ,
                                   edge_ended       = True                   ,
                                   egt_simple       = False                  ,
                                   **self.layer_configs)
        
        self.input_embed = layers.EmbedInput(node_width      = self.node_width     ,
                                             edge_width      = self.edge_width     ,
                                             upto_hop        = self.upto_hop       ,
                                             embed_3d_type   = self.embed_3d_type  ,
                                             num_3d_kernels  = self.num_3d_kernels )
        
        self.final_ln_edge = nn.LayerNorm(self.edge_width)
        self.dist_pred = nn.Linear(self.edge_width, self.num_dist_bins)
    
    def forward(self, inputs):
        g = self.input_embed(inputs)
        g = self.encoder(g)
        
        e = g.e
        e = self.final_ln_edge(e)
        e = self.dist_pred(e)
        return e
