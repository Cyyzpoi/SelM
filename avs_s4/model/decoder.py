from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import copy




def l2norm(X, dim=-1, eps=1e-12):
    """
    L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.GroupNorm(32,out_dim), nn.ReLU(True))

def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, dim: int = -2) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau 
    y_soft = gumbels.softmax(dim)

    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

class Fusion(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, out_dim, bias=False) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim_1+in_dim_2, out_dim, 3, padding=1, bias=bias),
            nn.GroupNorm(32,out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=bias),
            nn.GroupNorm(32,out_dim),
            nn.ReLU(),
        )

    def forward(self, in_1, in_2):
        if in_1.shape[-1] < in_2.shape[-1]:
            in_1 = F.interpolate(in_1, size=in_2.shape[-2:], mode='bilinear', align_corners=True)
        elif in_1.shape[-1] > in_2.shape[-1]:
            in_2 = F.interpolate(in_2, size=in_1.shape[-2:], mode='bilinear', align_corners=True)

        x = torch.cat((in_1, in_2), dim=1)
        x = self.fusion(x)
        return x

class DProjector(nn.Module):
    def __init__(self, audio_dim=512, in_dim=512, kernel_size=1,backbone='resnet'):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        if backbone.lower()=='pvt':
            self.vis = nn.Sequential(  
                # nn.Upsample(scale_factor=2, mode='bilinear'),
                # conv_layer(in_dim, in_dim, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                conv_layer(in_dim, in_dim, 3, padding=1),
                nn.Conv2d(in_dim, in_dim, 1))
        else:
            self.vis = nn.Sequential(  
                nn.Upsample(scale_factor=2, mode='bilinear'),
                conv_layer(in_dim, in_dim, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                conv_layer(in_dim, in_dim, 3, padding=1),
                nn.Conv2d(in_dim, in_dim, 1))

        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.audio_linear = nn.Linear(audio_dim, out_dim)

    def forward(self, x, audio):
        
        #x: (BT, 256, H, W)
        #audio: (BT, 256)
        
        x = self.vis(x)  

        B, C, H, W = x.size()
        
        x = x.reshape(1, B * C, H, W)
        
        audio = self.audio_linear(audio) 

        weight, bias = audio[:, :-1], audio[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, H, W -> 1, BT, H, W
        out = F.conv2d(x,
                    weight,
                    padding=0,
                    groups=B,
                    bias=bias)
            
        out = out.transpose(0,1)#  (B 1 H W)
        
        return out


class CrossAttn(nn.Module):
    def __init__(self,
                 q_dim,
                 kv_dim,
                 hidden_dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = q_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        self.q_proj = nn.Linear(q_dim, hidden_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(kv_dim, hidden_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(kv_dim, hidden_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value=None, mask=None):
        B, N, C = query.shape
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, N, S]
        
        if mask is not None:
            mask = mask[:,None,:,None].expand(-1, self.num_heads, -1, -1) # b nh S 1
            k = k * mask
            v = v * mask
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn + (1e4*mask.transpose(-2,-1)-1e4) # b nh 1 S
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        assert attn.shape == (B, self.num_heads, N, S)
        # [B, nh, N, C//nh] -> [B, N, C]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class LoadToken(nn.Module):
    def __init__(self, token_dim, bias, drop) -> None:
        super().__init__()
        self.cross_attn = CrossAttn(
            q_dim=token_dim,
            kv_dim=256,
            hidden_dim=token_dim,
            num_heads=1,
            out_dim=token_dim,
            qkv_bias=bias,
            attn_drop=drop,
            proj_drop=drop,
        )
        self.normq = nn.LayerNorm(token_dim)
        self.normk = nn.LayerNorm(256)
        self.norm = nn.LayerNorm(token_dim)
        self.mlp = Mlp(token_dim, token_dim*2, token_dim)

    def forward(self, tokens, audio, audio_mask):
        ltoken, ttoken = torch.split(tokens, [tokens.shape[1]-1,1], dim=1)
        ttoken = ttoken + self.cross_attn(query=self.normq(ttoken), key=self.normk(audio.permute(0,2,1)), mask=audio_mask[...,0])
        tokens = torch.cat((ltoken, ttoken), dim=1)
        return tokens

class LoadLayer(nn.Module):
    def __init__(self, token_dim, drop, bias=False, pe_shape=None) -> None:
        super().__init__()
        if pe_shape >=14:
            self.loadtoken = LoadToken(
                token_dim=token_dim,
                bias=bias,
                drop=drop
            )
            self.norm = nn.LayerNorm(token_dim)
            self.mlp = Mlp(token_dim, token_dim*2, token_dim)
        self.positional_embedding = nn.Parameter(torch.randn(pe_shape**2, token_dim) / token_dim ** 0.5)
        self.pe_shape = pe_shape

    def forward(self, tokens, audio, audio_mask):
        if self.pe_shape >=14:
            tokens = self.loadtoken(tokens, audio, audio_mask)
            tokens = self.mlp(self.norm(tokens))
        return tokens, self.positional_embedding


class CGAttention(nn.Module):
    def __init__(self, token_dim, vis_dim, hidden_dim, drop=0., bias=True) -> None:
        super().__init__()
        self.norm_v = nn.LayerNorm(vis_dim)
        self.norm_t = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(vis_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(vis_dim, hidden_dim, bias=bias)
        self.proj = nn.Linear(hidden_dim, token_dim)
        self.proj_drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(token_dim)
        self.mlp = Mlp(token_dim, token_dim*2, token_dim, drop=drop)
        self.tau = nn.Parameter(torch.ones(1), requires_grad=True)

    def with_pe(self, vis, pe):
        return vis + pe

    def forward(self, tokens, vis, pe=None):
        b, c, h , w = vis.shape
        vis = rearrange(vis, 'b c h w -> b (h w) c')
        if pe is not None:
            vis = self.with_pe(vis, pe)
        vis = self.norm_v(vis)
        q = self.q_proj(self.norm_t(tokens))
        k = self.k_proj(vis)
        v = self.v_proj(vis)

        q = l2norm(q, dim=-1)
        k = l2norm(k, dim=-1)
        raw_attn = (q @ k.transpose(-2, -1))
        tau = torch.clamp(self.tau, max=0).exp()
        attn = gumbel_softmax(raw_attn, dim=-2, tau=tau)
        hit_map = attn
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1)
        new_tokens = attn @ v
        new_tokens = self.proj_drop(self.proj(new_tokens))
        new_tokens = self.mlp(self.norm(new_tokens+tokens))
        return new_tokens, hit_map.reshape(b, -1, h, w)

class Decoder(nn.Module):
    def __init__(self, token_dim,num_token,backbone='resnet'):
        super().__init__()

        token_dim = token_dim
        self.tokens = nn.Embedding(num_token, token_dim)
        trunc_normal_(self.tokens.weight, std=0.02)
        
        dims = [256,256,256,256]
        # dims=[2048,1024,512,256]
        # pe_shapes = [20, 40, 80]
        if backbone.lower() == 'pvt':
            pe_shapes=[28,56,112]
        else:
            pe_shapes=[14,28,56]

        self.layers = []
        for pe_shape in pe_shapes:
            self.layers.append(LoadLayer(token_dim, drop=.1, bias=False, pe_shape=pe_shape))
        self.cgattention1 = CGAttention(token_dim=token_dim,
                            vis_dim=token_dim,
                            hidden_dim=token_dim,
                            drop=0.1,
                            bias=True)
        self.cgattention2 = CGAttention(token_dim=token_dim,
                    vis_dim=token_dim,
                    hidden_dim=token_dim,
                    drop=0.1,
                    bias=True)
        self.cgattention3= CGAttention(token_dim=token_dim,
                                       vis_dim=token_dim
                                       ,hidden_dim=token_dim,
                                       drop=0.,
                                       bias=True)
        self.layers = nn.ModuleList(self.layers)
        self.fuses = []
        self.fuses.append(Fusion(dims[0],dims[1],token_dim,bias=True))
        self.fuses.append(Fusion(dims[2],token_dim,token_dim,bias=True))
        self.fuses.append(Fusion(dims[3],token_dim,token_dim,bias=True))
        self.fuses = nn.ModuleList(self.fuses)
    
        
        self.proj = DProjector(audio_dim=token_dim, in_dim=token_dim,backbone=backbone)

    def forward(self, vis, audio, audio_mask):
        x_c4, x_c3, x_c2, x_c1 = vis
        tokens = self.tokens.weight[None,...].expand(x_c1.shape[0], -1, -1)
        maps = []
        v = x_c4
        for load, layer,fuse,v_   in zip(self.layers,[self.cgattention1,self.cgattention2,self.cgattention3],self.fuses,[x_c3,x_c2,x_c1]):
            v = fuse(v, v_)
            tokens, pe = load(tokens, audio, audio_mask)
            tokens, hitmap = layer(tokens, v, pe=pe)
            maps.append(hitmap)
        
        out = self.proj(v, tokens[:,-1])

        return out, maps