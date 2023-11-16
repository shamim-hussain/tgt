
import torch
from torch import nn

def get_triangle_layer(layer_type):
    if layer_type == 'update':
        return TriangleUpdate
    elif layer_type == 'update_ungated':
        return TriangleUpdateUngated
    elif layer_type == 'attention':
        return TriangleAttention
    elif layer_type == 'attention_ungated':
        return TriangleAttentionUngated
    elif layer_type == 'attention_mq':
        return TriangleAttentionMQ
    else:
        raise ValueError(f'Invalid layer_type: {layer_type}')

class TriangleUpdate(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 source_dropout = 0    ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.source_dropout      = source_dropout
        
        assert not (self.edge_width % self.num_heads),\
                'edge_width must be divisible by num_heads'
        self._dot_dim = self.edge_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_V   = nn.Linear(self.edge_width, self.edge_width*2)
        self.lin_EG  = nn.Linear(self.edge_width, self.num_heads*4)
        
        self.lin_O  = nn.Linear(self.edge_width*2, self.edge_width)
    
    def forward(self, e, mask):
        bsize, num_edges, _, embed_dim = e.shape
        e_ln = self.tri_ln_e(e)
        
        # Projections
        V_in, V_out = self.lin_V(e_ln).chunk(2, dim=-1)
        E_in, G_in, E_out, G_out = self.lin_EG(e_ln).chunk(4, dim=-1)
        
        V_in = V_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        mask_in = mask
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,1,num_edges,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_in = mask_in + rmask
        gates_in = torch.sigmoid(G_in+mask_in)
        A_in = torch.softmax(E_in+mask_in, dim=2) * gates_in
        Va_in = torch.einsum('bikh,bjkdh->bijdh', A_in, V_in)
        
        mask_out = mask
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,num_edges,1,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_out = mask_out + rmask
        gates_out = torch.sigmoid(G_out+mask_out)
        A_out = torch.softmax(E_out+mask_out, dim=1) * gates_out
        Va_out = torch.einsum('bkih,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        
        e = self.lin_O(Va)
        return e



class TriangleUpdateUngated(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 source_dropout = 0    ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.source_dropout      = source_dropout
        
        assert not (self.edge_width % self.num_heads),\
                'edge_width must be divisible by num_heads'
        self._dot_dim = self.edge_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_V   = nn.Linear(self.edge_width, self.edge_width*2)
        self.lin_E  = nn.Linear(self.edge_width, self.num_heads*2)
        
        self.lin_O  = nn.Linear(self.edge_width*2, self.edge_width)
    
    def forward(self, e, mask):
        bsize, num_edges, _, embed_dim = e.shape
        e_ln = self.tri_ln_e(e)
        
        # Projections
        V_in, V_out = self.lin_V(e_ln).chunk(2, dim=-1)
        E_in, E_out = self.lin_E(e_ln).chunk(2, dim=-1)
        
        V_in = V_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        mask_in = mask
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,1,num_edges,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_in = mask_in + rmask
        
        A_in = torch.softmax(E_in+mask_in, dim=2)
        Va_in = torch.einsum('bikh,bjkdh->bijdh', A_in, V_in)
        
        mask_out = mask
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,num_edges,1,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_out = mask_out + rmask
        
        A_out = torch.softmax(E_out+mask_out, dim=1)
        Va_out = torch.einsum('bkih,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1)\
                  .view(bsize,num_edges,num_edges,embed_dim*2)
        
        e = self.lin_O(Va)
        return e



class TriangleAttention(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 source_dropout = 0    ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.source_dropout      = source_dropout
        
        assert not (self.edge_width % self.num_heads),\
                'edge_width must be divisible by num_heads'
        self._dot_dim = self.edge_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_QKV_in = nn.Linear(self.edge_width, self.edge_width*3)
        self.lin_EG_in  = nn.Linear(self.edge_width, self.num_heads*2)
        
        self.lin_QKV_out = nn.Linear(self.edge_width, self.edge_width*3)
        self.lin_EG_out  = nn.Linear(self.edge_width, self.num_heads*2)
        
        self.lin_O  = nn.Linear(self.edge_width*2, self.edge_width)
    
    def forward(self, e, mask):
        bsize, num_edges, _, embed_dim = e.shape
        e_ln = self.tri_ln_e(e)
        
        # Projections
        Q_in, K_in, V_in = self.lin_QKV_in(e_ln).chunk(3, dim=-1)
        E_in, G_in = self.lin_EG_in(e_ln).unsqueeze(2).chunk(2, dim=-1) # bi1kh
        
        Q_in = Q_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_in = K_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_in = V_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_in = Q_in * self._scale_factor
        H_in = torch.einsum('bijdh,bjkdh->bijkh', Q_in, K_in) + E_in
        
        mask_in = mask.unsqueeze(2)
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,1,num_edges,num_edges,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_in = mask_in + rmask
        gates_in = torch.sigmoid(G_in+mask_in)
        A_in = torch.softmax(H_in+mask_in, dim=3) * gates_in
        
        Va_in = torch.einsum('bijkh,bjkdh->bijdh', A_in, V_in)
        
        Q_out, K_out, V_out = self.lin_QKV_out(e_ln).chunk(3, dim=-1)
        E_out, G_out = self.lin_EG_out(e_ln).unsqueeze(3).chunk(2, dim=-1) # bki1h
        
        Q_out = Q_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_out = K_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_out = Q_out * self._scale_factor
        H_out = torch.einsum('bijdh,bkjdh->bkijh', Q_out, K_out) + E_out
        
        mask_out = mask.unsqueeze(3)
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,num_edges,1,num_edges,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_out = mask_out + rmask
        gates_out = torch.sigmoid(G_out+mask_out)
        A_out = torch.softmax(H_out+mask_out, dim=1) * gates_out
        
        Va_out = torch.einsum('bkijh,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        e = self.lin_O(Va)
        return e


class TriangleAttentionUngated(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 source_dropout = 0    ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.source_dropout      = source_dropout
        
        assert not (self.edge_width % self.num_heads),\
                'edge_width must be divisible by num_heads'
        self._dot_dim = self.edge_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_QKV_in = nn.Linear(self.edge_width, self.edge_width*3)
        self.lin_E_in  = nn.Linear(self.edge_width, self.num_heads)
        
        self.lin_QKV_out = nn.Linear(self.edge_width, self.edge_width*3)
        self.lin_E_out  = nn.Linear(self.edge_width, self.num_heads)
        
        self.lin_O  = nn.Linear(self.edge_width*2, self.edge_width)
    
    def forward(self, e, mask):
        bsize, num_edges, _, embed_dim = e.shape
        e_ln = self.tri_ln_e(e)
        
        # Projections
        Q_in, K_in, V_in = self.lin_QKV_in(e_ln).chunk(3, dim=-1)
        E_in = self.lin_E_in(e_ln).unsqueeze(2) # bi1kh
        
        Q_in = Q_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_in = K_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_in = V_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_in = Q_in * self._scale_factor
        H_in = torch.einsum('bijdh,bjkdh->bijkh', Q_in, K_in) + E_in
        
        mask_in = mask.unsqueeze(2)
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,1,num_edges,num_edges,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_in = mask_in + rmask
        
        A_in = torch.softmax(H_in+mask_in, dim=3)
        
        Va_in = torch.einsum('bijkh,bjkdh->bijdh', A_in, V_in)
        
        Q_out, K_out, V_out = self.lin_QKV_out(e_ln).chunk(3, dim=-1)
        E_out = self.lin_E_out(e_ln).unsqueeze(3) # bki1h
        
        Q_out = Q_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_out = K_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_out = Q_out * self._scale_factor
        H_out = torch.einsum('bijdh,bkjdh->bkijh', Q_out, K_out) + E_out
        
        mask_out = mask.unsqueeze(3)
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,num_edges,1,num_edges,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_out = mask_out + rmask
        
        A_out = torch.softmax(H_out+mask_out, dim=1)
        
        Va_out = torch.einsum('bkijh,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        e = self.lin_O(Va)
        return e




class TriangleAttentionMQ(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 source_dropout = 0    ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.source_dropout      = source_dropout
        
        assert not (self.edge_width % self.num_heads),\
                'edge_width must be divisible by num_heads'
        self._dot_dim = self.edge_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_Q_in = nn.Linear(self.edge_width, self.edge_width)
        self.lin_KV_in = nn.Linear(self.edge_width, self._dot_dim*2)
        self.lin_EG_in  = nn.Linear(self.edge_width, self.num_heads*2)
        
        self.lin_Q_out = nn.Linear(self.edge_width, self.edge_width)
        self.lin_KV_out = nn.Linear(self.edge_width, self._dot_dim*2)
        self.lin_EG_out  = nn.Linear(self.edge_width, self.num_heads*2)
        
        self.lin_O  = nn.Linear(self.edge_width*2, self.edge_width)
    
    def forward(self, e, mask):
        bsize, num_edges, _, embed_dim = e.shape
        e_ln = self.tri_ln_e(e)
        
        # Projections
        Q_in = self.lin_Q_in(e_ln)
        K_in, V_in = self.lin_KV_in(e_ln).chunk(2, dim=-1)
        E_in, G_in = self.lin_EG_in(e_ln).unsqueeze(2).chunk(2, dim=-1) # bi1kh
        
        Q_in = Q_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_in = K_in.view(bsize,num_edges,num_edges,self._dot_dim)
        V_in = V_in.view(bsize,num_edges,num_edges,self._dot_dim)
        
        Q_in = Q_in * self._scale_factor
        H_in = torch.einsum('bijdh,bjkd->bijkh', Q_in, K_in) + E_in
        
        mask_in = mask.unsqueeze(2)
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,1,num_edges,num_edges,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_in = mask_in + rmask
        gates_in = torch.sigmoid(G_in+mask_in)
        A_in = torch.softmax(H_in+mask_in, dim=3) * gates_in
        
        Va_in = torch.einsum('bijkh,bjkd->bijdh', A_in, V_in)
        
        Q_out = self.lin_Q_out(e_ln)
        K_out, V_out = self.lin_KV_out(e_ln).chunk(2, dim=-1)
        E_out, G_out = self.lin_EG_out(e_ln).unsqueeze(3).chunk(2, dim=-1) # bki1h
        
        Q_out = Q_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_out = K_out.view(bsize,num_edges,num_edges,self._dot_dim)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim)
        
        Q_out = Q_out * self._scale_factor
        H_out = torch.einsum('bijdh,bkjd->bkijh', Q_out, K_out) + E_out
        
        mask_out = mask.unsqueeze(3)
        if self.source_dropout > 0 and self.training:
            rmask = e.new_empty(size=[bsize,num_edges,1,num_edges,1])\
                     .bernoulli_(self.source_dropout)\
                         * torch.finfo(mask.dtype).min
            mask_out = mask_out + rmask
        gates_out = torch.sigmoid(G_out+mask_out)
        A_out = torch.softmax(H_out+mask_out, dim=1) * gates_out
        
        Va_out = torch.einsum('bkijh,bkjd->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        e = self.lin_O(Va)
        return e
