
import torch
from torch import nn
import torch.nn.functional as F
from .activations import get_activation
from .triangle import get_triangle_layer

@torch.jit.script
def form_degree_scalers(gates: torch.Tensor):
    degrees = torch.sum(gates, dim=2, keepdim=True)
    degree_scalers = torch.log(1+degrees)
    return degree_scalers


class EGT_Attention(nn.Module):
    def __init__(self,
                 node_width            ,
                 edge_width            ,
                 num_heads             ,
                 source_dropout = 0    ,
                 scale_degree   = True ,
                 edge_update    = True ,
                 ):
        super().__init__()
        self.node_width          = node_width
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.source_dropout      = source_dropout
        self.scale_degree        = scale_degree
        self.edge_update         = edge_update
        
        assert not (self.node_width % self.num_heads),\
                'node_width must be divisible by num_heads'
        self._dot_dim = self.node_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.mha_ln_h   = nn.LayerNorm(self.node_width)
        self.mha_ln_e   = nn.LayerNorm(self.edge_width)
        self.lin_QKV    = nn.Linear(self.node_width, self.node_width*3)
        self.lin_EG     = nn.Linear(self.edge_width, self.num_heads*2)
        
        self.lin_O_h    = nn.Linear(self.node_width, self.node_width)
        if self.edge_update:
            self.lin_O_e    = nn.Linear(self.num_heads, self.edge_width)
    
    def forward(self, h, e, mask):
        bsize, num_nodes, embed_dim = h.shape
        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)
        
        # Projections
        Q, K, V = self.lin_QKV(h_ln).chunk(3, dim=-1)
        E, G = self.lin_EG(e_ln).chunk(2, dim=-1)
        
        if self.source_dropout > 0 and self.training:
            rmask = h.new_empty(size=[bsize,1,num_nodes,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask = mask + rmask
            
        # Multi-head attention
        Q = Q.view(bsize,num_nodes,self._dot_dim,self.num_heads)
        K = K.view(bsize,num_nodes,self._dot_dim,self.num_heads)
        V = V.view(bsize,num_nodes,self._dot_dim,self.num_heads)
        
        Q = Q * self._scale_factor
        
        gates = torch.sigmoid(G+mask)
        H_hat = torch.einsum('bldh,bmdh->blmh', Q, K) + E
        A_tild = F.softmax(H_hat+mask, dim=2) * gates
        V_att = torch.einsum('blmh,bmkh->blkh', A_tild, V)
        
        if self.scale_degree:
            degree_scalers = form_degree_scalers(gates)
            V_att = V_att * degree_scalers
        
        V_att = V_att.reshape(bsize,num_nodes,embed_dim)
        
        # Update
        h = self.lin_O_h(V_att)
        if self.edge_update:
            e = self.lin_O_e(H_hat)
        
        return h, e


class EdgeUpdate(nn.Module):
    def __init__(self,
                 node_width            ,
                 edge_width            ,
                 num_heads             ,
                 ):
        super().__init__()
        self.node_width          = node_width
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        
        assert not (self.node_width % self.num_heads),\
                'node_width must be divisible by num_heads'
        self._dot_dim = self.node_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.mha_ln_h   = nn.LayerNorm(self.node_width)
        self.mha_ln_e   = nn.LayerNorm(self.edge_width)
        self.lin_QK     = nn.Linear(self.node_width, self.node_width*2)
        self.lin_E      = nn.Linear(self.edge_width, self.num_heads)
        
        self.lin_O_e    = nn.Linear(self.num_heads, self.edge_width)
    
    def forward(self, h, e, mask):
        bsize, num_nodes, _ = h.shape
        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)
        
        # Projections
        Q, K = self.lin_QK(h_ln).chunk(2, dim=-1)
        E = self.lin_E(e_ln)
        
            
        # Multi-head attention
        Q = Q.view(bsize,num_nodes,self._dot_dim,self.num_heads)
        K = K.view(bsize,num_nodes,self._dot_dim,self.num_heads)
        
        Q = Q * self._scale_factor
        H_hat = torch.einsum('bldh,bmdh->blmh', Q, K) + E
        
        # Update
        e = self.lin_O_e(H_hat)
        
        return h, e



class FFN(nn.Module):
    def __init__(self,
                 width,
                 multiplier = 1.,
                 act_dropout = 0.,
                 activation = 'gelu',
                 ):
        super().__init__()
        self.width = width
        self.multiplier = multiplier
        self.act_dropout = act_dropout
        self.activation = activation
        
        self.ffn_fn, self.act_mul = get_activation(activation)
        inner_dim = round(self.width*self.multiplier)
        
        self.ffn_ln = nn.LayerNorm(self.width)
        self.lin_W1  = nn.Linear(self.width, inner_dim*self.act_mul)
        self.lin_W2  = nn.Linear(inner_dim, self.width)
        self.dropout = nn.Dropout(self.act_dropout)
    
    def forward(self, x):
        x_ln = self.ffn_ln(x)
        x = self.ffn_fn(self.lin_W1(x_ln))
        x = self.dropout(x)
        x = self.lin_W2(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_path=0.):
        super().__init__()
        self.drop_path = drop_path
        self._keep_prob = 1 - self.drop_path
    
    def forward(self, x):
        if self.drop_path > 0 and self.training:
            mask_shape = [x.size(0)] + [1]*(x.ndim-1)
            mask = x.new_empty(size=mask_shape).bernoulli_(self._keep_prob)
            x = x.div(self._keep_prob) * mask
        return x
    
    def __repr__(self):
        return f'{self.__class__.__name__}(drop_path={self.drop_path})' 


class EGT_Layer(nn.Module):
    def __init__(self,
                 node_width                      ,
                 edge_width                      ,
                 num_heads                       ,
                 activation          = 'gelu'    ,
                 scale_degree        = True      ,
                 node_update         = True      ,
                 edge_update         = True      ,
                 triangle_heads      = 0         ,
                 triangle_type       = 'update'  ,
                 triangle_dropout    = 0         ,
                 node_ffn_multiplier = 1.        ,
                 edge_ffn_multiplier = 1.        ,
                 source_dropout      = 0         ,
                 drop_path           = 0         ,
                 node_act_dropout    = 0         ,
                 edge_act_dropout    = 0         ,
                 ):
        super().__init__()
        self.node_width          = node_width
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.activation          = activation
        self.node_ffn_multiplier = node_ffn_multiplier
        self.edge_ffn_multiplier = edge_ffn_multiplier
        self.node_act_dropout    = node_act_dropout
        self.edge_act_dropout    = edge_act_dropout
        self.source_dropout      = source_dropout
        self.drop_path           = drop_path
        self.scale_degree        = scale_degree
        self.node_update         = node_update
        self.edge_update         = edge_update
        self.triangle_heads      = triangle_heads
        self.triangle_type       = triangle_type
        self.triangle_dropout    = triangle_dropout
        
        self._triangle_update     = self.triangle_heads > 0
        
        if self.node_update:
            self.update = EGT_Attention(
                node_width      = self.node_width,
                edge_width      = self.edge_width,
                num_heads       = self.num_heads,
                source_dropout  = self.source_dropout,
                scale_degree    = self.scale_degree,
                edge_update     = self.edge_update,
                )
        elif self.edge_update:
            self.update = EdgeUpdate(
                node_width      = self.node_width,
                edge_width      = self.edge_width,
                num_heads       = self.num_heads,
                )
        else:
            raise ValueError('At least one of node_update and edge_update must be True')
        
        if self.node_update:
            self.node_ffn = FFN(
                width           = self.node_width,
                multiplier      = self.node_ffn_multiplier,
                act_dropout     = self.node_act_dropout,
                activation      = self.activation,
                )
        if self.edge_update:
            if self._triangle_update:
                TriangleLayer = get_triangle_layer(self.triangle_type)
                self.tria = TriangleLayer(
                    edge_width      = self.edge_width,
                    num_heads       = self.triangle_heads,
                    source_dropout  = self.triangle_dropout,
                )
            
            self.edge_ffn = FFN(
                width           = self.edge_width,
                multiplier      = self.edge_ffn_multiplier,
                act_dropout     = self.edge_act_dropout,
                activation      = self.activation,
                )
        
        self.drop_path = DropPath(self.drop_path)
    
    def forward(self, g):
        h, e, mask = g.h, g.e, g.mask
        
        h_r1, e_r1 = h, e
        h, e = self.update(h, e, mask)
        
        if self.node_update:
            h = self.drop_path(h)
            h.add_(h_r1)
            
            h_r2 = h
            h = self.node_ffn(h)
            h = self.drop_path(h)
            h.add_(h_r2)
        
        if self.edge_update:
            e = self.drop_path(e)
            e.add_(e_r1)
            
            if self._triangle_update:
                e_rt = e
                e = self.tria(e, mask)
                e = self.drop_path(e)
                e.add_(e_rt)
            
            e_r2 = e
            e = self.edge_ffn(e)
            e = self.drop_path(e)
            e.add_(e_r2)
        
        g = g.copy()
        g.h, g.e = h, e
        return g
    
    def __repr__(self):
        rep = super().__repr__()
        rep = (rep + ' ('
                   + f'activation: {self.activation}, '
                   + f'source_dropout: {self.source_dropout}'
                   +')')
        return rep

