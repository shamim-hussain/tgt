
import torch
from torch import nn
import torch.nn.functional as F
from .layers import TGT_Layer

class Graph(dict):
    def __dir__(self):
        return super().__dir__() + list(self.keys())
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError('No such attribute: '+key)
        
    def __setattr__(self, key, value):
        self[key]=value
        
    def copy(self):
        return self.__class__(self)


class TGT_Encoder(nn.Module):
    class IndivConfig(list): pass
    
    def __init__(self,
                 model_height        = 4         ,
                 layer_multiplier    = 1         ,
                 node_ended          = True      ,
                 edge_ended          = True      ,
                 egt_simple          = False     ,
                 **layer_configs
                 ):
        super().__init__()
        self.model_height        = model_height
        self.layer_multiplier    = layer_multiplier
        self.node_ended          = node_ended
        self.edge_ended          = edge_ended
        self.egt_simple          = egt_simple
        self.layer_configs       = layer_configs
        for k,v in layer_configs.items():
            setattr(self, k, v)
        
        assert (self.node_ended or self.edge_ended),\
                'At least one of node_ended and edge_ended must be True'
        
        self.TGT_layers = nn.ModuleList([TGT_Layer(**self.get_layer_kwargs(i))
                                                          for i in range(self.model_height)])
     
    
    def get_layer_kwargs(self, i):
        layer_kwargs = {}
        for k,v in self.layer_configs.items():
            if isinstance(v, self.IndivConfig):
                layer_kwargs[k] = v[i]
            elif k == 'drop_path':
                layer_kwargs[k] = v * i / (self.model_height - 1)
            else:
                layer_kwargs[k] = v
        
        if (i == self.model_height - 1)\
                              and (not self.node_ended):
            node_update = False
        else:
            node_update = True
        layer_kwargs['node_update'] = node_update
        
        if self.egt_simple:
            edge_update = False
        elif (i == self.model_height - 1)\
                              and (not self.edge_ended):
            edge_update = False
        else:
            edge_update = True
        layer_kwargs['edge_update'] = edge_update
        
        return layer_kwargs
    
    def apply_layer(self, layer_idx, graph):
        layer = self.TGT_layers[layer_idx]
        for _ in range(self.layer_multiplier):
            graph = layer(graph)
        return graph
    
    def forward(self, inputs):
        g = Graph(inputs)
        for i in range(self.model_height):
            g = self.apply_layer(i, g)
        return g
