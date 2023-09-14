import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim) # if True, need another projection layer
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.layernorm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias = False) # (b, n, dim) -> (b, n, heads * dim_head * 3)
        self.project_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    
    def forward(self, x):
        x = self.layernorm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1) # (b, n, heads * dim_head * 3) -> 3 * [(b, n, heads * dim_head)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # (b, n, heads * dim_head) -> (b, heads, n, dim_head)
        
        dots = (q @ k.transpose(-2, -1)) * self.scale # (b, heads, n, dim_head) @ (b, heads, dim_head, n) = (b, heads, n, n)
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = attn @ v # (b, heads, n, n) @ (b, heads, n, dim_head) = (b, heads, n, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)') # (b, heads, n, dim_head) -> (b, n, heads * dim_head)
        return self.project_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.layers = []
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            
        return self.layernorm(x)
