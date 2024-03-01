
import torch
from torch import nn
import torch.nn.functional as F

def get_triplet_layer(layer_type):
    if layer_type == 'aggregate':
        return TripletAggregate
    elif layer_type == 'aggregate_ungated':
        return TripletAggregateUngated
    elif layer_type == 'attention':
        return TripletAttention
    elif layer_type == 'attention_ungated':
        return TripletAttentionUngated
    elif layer_type == 'tiangular_update':
        return TriangularUpdate
    elif layer_type == 'axial_attention':
        return AxialAttention
    else:
        raise ValueError(f'Invalid layer_type: {layer_type}')

class TripletAggregate(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
        
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
        
        gates_in = torch.sigmoid(G_in+mask)
        A_in = torch.softmax(E_in+mask, dim=2) * gates_in
        if self.attention_dropout > 0:
            A_in = F.dropout(A_in, p=self.attention_dropout,
                             training=self.training, inplace=True)
        Va_in = torch.einsum('bikh,bjkdh->bijdh', A_in, V_in)
        
        gates_out = torch.sigmoid(G_out)
        A_out = torch.softmax(E_out, dim=1) * gates_out
        if self.attention_dropout > 0:
            A_out = F.dropout(A_out, p=self.attention_dropout,
                              training=self.training, inplace=True)
        Va_out = torch.einsum('bkih,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        
        e = self.lin_O(Va)
        return e



class TripletAggregateUngated(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
        
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
        
        A_in = torch.softmax(E_in+mask, dim=2)
        if self.attention_dropout > 0:
            A_in = F.dropout(A_in, p=self.attention_dropout,
                             training=self.training, inplace=True)
        Va_in = torch.einsum('bikh,bjkdh->bijdh', A_in, V_in)
        
        A_out = torch.softmax(E_out+mask, dim=1)
        if self.attention_dropout > 0:
            A_out = F.dropout(A_out, p=self.attention_dropout,
                              training=self.training, inplace=True)
        Va_out = torch.einsum('bkih,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1)\
                  .view(bsize,num_edges,num_edges,embed_dim*2)
        
        e = self.lin_O(Va)
        return e


@torch.jit.script
def siglin(gates, lins):
    return torch.sigmoid(gates) * lins

class TriangularUpdate(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
        
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        
        self.lin_V   = nn.Linear(self.edge_width, self.num_heads*4)
        self.lin_E  = nn.Linear(self.edge_width, self.num_heads*4)
        
        self.lin_O  = nn.Linear(self.num_heads*2, self.edge_width*2)
    
    def forward(self, e, mask):
        e_ln = self.tri_ln_e(e)
        
        # Projections
        V_in_g, V_in_l, V_out_g, V_out_l = self.lin_V(e_ln).chunk(4, dim=-1)
        E_in_g, E_in_l, E_out_g, E_out_l = self.lin_E(e_ln).chunk(4, dim=-1)
        
        E_in_g = E_in_g + mask
        E_out_g = E_out_g + mask
        V_in_g = V_in_g + mask
        V_out_g = V_out_g + mask
        
        V_in = siglin(V_in_g, V_in_l)
        V_out = siglin(V_out_g, V_out_l)
        E_in = siglin(E_in_g, E_in_l)
        E_out = siglin(E_out_g, E_out_l)
        
        Va_in = torch.einsum('bikh,bjkh->bijh', E_in, V_in)
        Va_out = torch.einsum('bkih,bkjh->bijh', E_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1)
        
        e_g, e_l = self.lin_O(Va).chunk(2, dim=-1)
        e = siglin(e_g, e_l)
        return e


class TripletAttention(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
        
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
        gates_in = torch.sigmoid(G_in+mask_in)
        A_in = torch.softmax(H_in+mask_in, dim=3) * gates_in
        if self.attention_dropout > 0:
            A_in = F.dropout(A_in, p=self.attention_dropout,
                             training=self.training, inplace=True)
        
        Va_in = torch.einsum('bijkh,bjkdh->bijdh', A_in, V_in)
        
        Q_out, K_out, V_out = self.lin_QKV_out(e_ln).chunk(3, dim=-1)
        E_out, G_out = self.lin_EG_out(e_ln).unsqueeze(3).chunk(2, dim=-1) # bki1h
        
        Q_out = Q_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_out = K_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_out = Q_out * self._scale_factor
        H_out = torch.einsum('bijdh,bkjdh->bkijh', Q_out, K_out) + E_out
        
        mask_out = mask.unsqueeze(3)
        gates_out = torch.sigmoid(G_out+mask_out)
        A_out = torch.softmax(H_out+mask_out, dim=1) * gates_out
        if self.attention_dropout > 0:
            A_out = F.dropout(A_out, p=self.attention_dropout,
                              training=self.training, inplace=True)
        
        Va_out = torch.einsum('bkijh,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        e = self.lin_O(Va)
        return e


class TripletAttentionUngated(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
        
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
        A_in = torch.softmax(H_in+mask_in, dim=3)
        if self.attention_dropout > 0:
            A_in = F.dropout(A_in, p=self.attention_dropout,
                             training=self.training, inplace=True)
        
        Va_in = torch.einsum('bijkh,bjkdh->bijdh', A_in, V_in)
        
        Q_out, K_out, V_out = self.lin_QKV_out(e_ln).chunk(3, dim=-1)
        E_out = self.lin_E_out(e_ln).unsqueeze(3) # bki1h
        
        Q_out = Q_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_out = K_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_out = Q_out * self._scale_factor
        H_out = torch.einsum('bijdh,bkjdh->bkijh', Q_out, K_out) + E_out
        
        mask_out = mask.unsqueeze(3)
        A_out = torch.softmax(H_out+mask_out, dim=1)
        if self.attention_dropout > 0:
            A_out = F.dropout(A_out, p=self.attention_dropout,
                              training=self.training, inplace=True)
        
        Va_out = torch.einsum('bkijh,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        e = self.lin_O(Va)
        return e


class AxialAttention(nn.Module):
    def __init__(self,
                 edge_width            ,
                 num_heads             ,
                 attention_dropout = 0 ,
                 ):
        super().__init__()
        self.edge_width          = edge_width
        self.num_heads           = num_heads
        self.attention_dropout   = attention_dropout
        
        assert not (self.edge_width % self.num_heads),\
                'edge_width must be divisible by num_heads'
        self._dot_dim = self.edge_width//self.num_heads
        self._scale_factor = self._dot_dim ** -0.5
        
        self.tri_ln_e   = nn.LayerNorm(self.edge_width)
        self.lin_QKV_in = nn.Linear(self.edge_width, self.edge_width*3)
        self.lin_QKV_out = nn.Linear(self.edge_width, self.edge_width*3)
        self.lin_O  = nn.Linear(self.edge_width*2, self.edge_width)
    
    def forward(self, e, mask):
        bsize, num_edges, _, embed_dim = e.shape
        e_ln = self.tri_ln_e(e)
        
        # Projections
        Q_in, K_in, V_in = self.lin_QKV_in(e_ln).chunk(3, dim=-1)
        
        Q_in = Q_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_in = K_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_in = V_in.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_in = Q_in * self._scale_factor
        H_in = torch.einsum('bijdh,bjkdh->bijkh', Q_in, K_in)
        
        mask_in = mask.unsqueeze(2)
        A_in = torch.softmax(H_in+mask_in, dim=3)
        if self.attention_dropout > 0:
            A_in = F.dropout(A_in, p=self.attention_dropout,
                             training=self.training, inplace=True)
        
        Va_in = torch.einsum('bijkh,bjkdh->bijdh', A_in, V_in)
        
        Q_out, K_out, V_out = self.lin_QKV_out(e_ln).chunk(3, dim=-1)
        
        Q_out = Q_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        K_out = K_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        V_out = V_out.view(bsize,num_edges,num_edges,self._dot_dim,self.num_heads)
        
        Q_out = Q_out * self._scale_factor
        H_out = torch.einsum('bijdh,bkjdh->bkijh', Q_out, K_out)
        
        mask_out = mask.unsqueeze(3)
        A_out = torch.softmax(H_out+mask_out, dim=1)
        if self.attention_dropout > 0:
            A_out = F.dropout(A_out, p=self.attention_dropout,
                              training=self.training, inplace=True)
        
        Va_out = torch.einsum('bkijh,bkjdh->bijdh', A_out, V_out)
        
        Va = torch.cat([Va_in, Va_out], dim=-1).view(bsize,num_edges,num_edges,embed_dim*2)
        e = self.lin_O(Va)
        return e

