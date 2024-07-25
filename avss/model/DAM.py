import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange,repeat
from torchvision import models
import pdb
import math
from model.backbone import SwinTransformerBlock,window_partition
# from mmcv_c import checkpoint

class MMBasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 num_heads_fusion=1,
                 fusion_drop=0.0
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.dim = dim

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # fuse before downsampling
        self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                           dim,  # v_in
                           256,  # l_in
                           dim,  # key
                           dim,  # value
                           num_heads=num_heads_fusion,
                           dropout=fusion_drop)

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        self.W_l = nn.Sequential(
            nn.Conv1d(256, 256, 1, 1),  # the init function sets bias to 0 if bias is True
            nn.GELU()
        )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        # initialize the gate to 0
        nn.init.zeros_(self.res_gate[0].weight)
        nn.init.zeros_(self.res_gate[2].weight)

    def forward(self, x, H, W, l, l_mask):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                # x = checkpoint.checkpoint(blk, x, attn_mask)
                x=0
            else:
                x = blk(x, attn_mask)  # output of a Block has shape (B, H*W, dim)

        # PWAM fusion
        x_residual, l_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual)
        l = l + self.W_l(l_residual)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_residual, l, H, W, x_down, Wh, Ww
        else:
            return x,l


class All_Fusion_Block(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.0):
        super(All_Fusion_Block, self).__init__()
        # input x shape: (B, H*W, dim)
        self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                           dim,  # v_in
                           256,  # l_in
                           dim,  # key
                           dim,  # value
                           num_heads=num_heads,
                           dropout=dropout)
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        self.W_l = nn.Sequential(
            nn.Conv1d(256, 256, 1, 1),  # the init function sets bias to 0 if bias is True
            nn.GELU()
        )

    def forward(self, x, l, l_mask):

        H, W = x.shape[2], x.shape[3]
        x = x.view(x.shape[0], x.shape[1], H*W)
        x = x.permute(0, 2, 1).contiguous()
        x_residual, l_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual)
        l = l + self.W_l(l_residual)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.shape[0], x.shape[1], H, W)

        return x, l



class ALL_Fusion_Block_A2V(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.0):
        super(ALL_Fusion_Block_A2V, self).__init__()
        # input x shape: (B, H*W, dim)
        self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                           dim,  # v_in
                           256,  # l_in
                           dim,  # key
                           dim,  # value
                           num_heads=num_heads,
                           dropout=dropout)
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        self.W_l = nn.Sequential(
            nn.Conv1d(256, 256, 1, 1),  # the init function sets bias to 0 if bias is True
            nn.GELU()
        )

    def forward(self, x, l, l_mask):

        H, W = x.shape[2], x.shape[3]
        x = x.view(x.shape[0], x.shape[1], H*W)
        x = x.permute(0, 2, 1).contiguous()
        x_residual, _ = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        x = x + (self.res_gate(x_residual) * x_residual)
        # l = l + self.W_l(l_residual)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.shape[0], x.shape[1], H, W)

        return x, l
    


class ALL_Fusion_Block_V2A(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.0):
        super(ALL_Fusion_Block_V2A, self).__init__()
        # input x shape: (B, H*W, dim)
        self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
                           dim,  # v_in
                           256,  # l_in
                           dim,  # key
                           dim,  # value
                           num_heads=num_heads,
                           dropout=dropout)
        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        self.W_l = nn.Sequential(
            nn.Conv1d(256, 256, 1, 1),  # the init function sets bias to 0 if bias is True
            nn.GELU()
        )

    def forward(self, x, l, l_mask):

        H, W = x.shape[2], x.shape[3]
        x = x.view(x.shape[0], x.shape[1], H*W)
        x = x.permute(0, 2, 1).contiguous()
        _, l_residual = self.fusion(x, l, l_mask)
        # apply a gate on the residual
        # x = x + (self.res_gate(x_residual) * x_residual)
        l = l + self.W_l(l_residual)
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.shape[0], x.shape[1], H, W)

        return x, l
    







class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                         )

        self.image_lang_att = SpatialImageInteraction(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, H*W, dim)

        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)

        lang, lang1 = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim) (B, l_dim, N_l)

        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm, lang1


class SpatialImageInteraction(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageInteraction, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.value_channels_l = l_in_channels
        if out_channels is None:
            self.out_channels = self.value_channels
        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )
        # Values: visual features: (B, H*W, v_in_channels)
        self.f_value_v = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.value_channels_l, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.value_channels_l),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )
        self.W2 = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.l_in_channels, kernel_size=1, stride=1),
        )
        self.num_heads = num_heads
        self.refineimg11 = RefineVisualSim(self.v_in_channels, self.l_in_channels, self.key_channels, kernel=1, num_heads=1)
        # self.refineimg33 = RefineVisualSim(self.v_in_channels, self.l_in_channels, self.key_channels, kernel=3, num_heads=1)
        # self.refineimg55 = RefineVisualSim(self.v_in_channels, self.l_in_channels, self.key_channels, kernel=5, num_heads=1)
        self.refinelan11 = RefineLanSim(self.v_in_channels, self.l_in_channels, self.key_channels, kernel=(1,1), num_heads=1)
        # self.refinelan21 = RefineLanSim(self.v_in_channels, self.l_in_channels, self.key_channels, kernel=(2,1), num_heads=1)
        # self.refinelan31 = RefineLanSim(self.v_in_channels, self.l_in_channels, self.key_channels, kernel=(3,1), num_heads=1)

        self.vis_weight = nn.Parameter(torch.ones(1))
        self.lan_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        n_l = l.size(2)
        l_mask1 = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        value = self.f_value(l)  # (B, self.value_channels, N_l)
        value = value * l_mask1  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        value = value.reshape(B, self.num_heads, self.value_channels // self.num_heads, n_l)
        # (b, num_heads, self.value_channels//self.num_heads, n_l)

        sim_mapv11 = self.refineimg11(x, l, l_mask)
        # sim_mapv33 = self.refineimg33(x, l, l_mask)
        # sim_mapv55 = self.refineimg55(x, l, l_mask)

        vis_weight1 = F.softmax(self.vis_weight, dim=0)
        # sim_mapv = vis_weight1[0] * sim_mapv11 + vis_weight1[1] * sim_mapv33 + vis_weight1[2] * sim_mapv55
        sim_mapv = vis_weight1 * sim_mapv11

        out_v = torch.matmul(sim_mapv, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out_v = out_v.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out_v = out_v.permute(0, 2, 1)  # (B, value_channels, HW)
        out_v = self.W(out_v)  # (B, value_channels, HW)
        out_v = out_v.permute(0, 2, 1)  # (B, HW, value_channels)

        x_v = x.permute(0, 2, 1)  # (B, v_in_channels, H*W)
        x_v = self.f_value_v(x_v)  # (B, value_channels_l, H*W)
        value_v = x_v.reshape(B, self.num_heads, self.l_in_channels // self.num_heads, HW)
        # (b, num_heads, self.value_channels_l//self.num_heads, H*W)

        sim_mapl11 = self.refinelan11(x, l, l_mask)
        # sim_mapl21 = self.refinelan21(x, l, l_mask)
        # sim_mapl31 = self.refinelan31(x, l, l_mask)

        lan_weight1 = F.softmax(self.lan_weight, dim=0)
        # sim_mapl = lan_weight1 * sim_mapl11 + lan_weight1[1] * sim_mapl21 + lan_weight1[2] * sim_mapl31
        sim_mapl = lan_weight1*sim_mapl11

        out_l = torch.matmul(sim_mapl, value_v.permute(0, 1, 3, 2))  # (B, num_heads, N_l, self.l_in_channels//num_heads)
        out_l = out_l.permute(0, 2, 1, 3).contiguous().reshape(B, n_l, self.l_in_channels)  # (B, N_l, l_in_channels)
        out_l = out_l.permute(0, 2, 1)  # (B, l_in_channels, N_l)
        out_l = self.W2(out_l)  # (B, l_in_channels, N_l)

        return out_v, out_l

class RefineVisualSim(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, kernel, num_heads=1):
        super(RefineVisualSim, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        if kernel == 1:
            self.int_channels = key_channels
        elif kernel == 3:
            self.int_channels = key_channels // 2
        elif kernel == 5:
            self.int_channels = key_channels // 4
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.kernel = kernel


        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
            # nn.LayerNorm(self.key_channels)
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.int_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.int_channels),
        )
        self.f_query2 = nn.Sequential(
            nn.Conv1d(self.int_channels * (self.kernel ** 2), self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )


    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        n_l = l.size(2)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        x = x.permute(0, 2, 1)  # (B, v_in_channels, H*W)
        x1 = self.f_query(x)
        x1 = rearrange(x1, 'b c (h w) -> b c h w', h=int(math.sqrt(x.shape[2])))
        x2 = F.unfold(x1, kernel_size=self.kernel, stride=1, padding=self.kernel//2)

        query = self.f_query2(x2)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        query = query.reshape(B, HW, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels // self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4 * l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)

        return sim_map

class RefineLanSim(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, kernel, num_heads=1):
        super(RefineLanSim, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.kernel = kernel
        if self.kernel[0] == 1:
            self.int_channels = key_channels
        elif self.kernel[0] == 2:
            self.int_channels = key_channels // 2
        elif self.kernel[0] == 3:
            self.int_channels = key_channels // 3



        # Keys: visual features: (B, v_in_channels, #words)
        self.f_key = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.l_in_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.l_in_channels),
        )

        # Queries: language features: (B, H*W, l_in_channels)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_query = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.int_channels, kernel_size=1, stride=1),
            # nn.LayerNorm(self.int_channels),
        )
        self.f_query2 = nn.Sequential(
            nn.Conv1d(self.int_channels * self.kernel[0], self.l_in_channels, kernel_size=1, stride=1),
            # nn.LayerNorm(self.l_in_channels),
        )


    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)

        # pdb.set_trace()
        B, HW = x.size(0), x.size(1)
        n_l = l.size(2)

        l1 = self.f_query(l) #(B, int_channels, N_l)
        l1 = l1.unsqueeze(3) #(B, int_channels, N_l, 1)
        l1 = F.pad(l1, (0, 0, self.kernel[0]//2, (self.kernel[0]-1)//2), mode='replicate') #(B, int_channels, N_l+, 1)
        l2 = F.unfold(l1, kernel_size=(self.kernel[0], 1), stride=1) #(B, int_channels*self.kernel[0], N_l)
        query = self.f_query2(l2) #(B, l_in_channels, N_l)
        query = query.permute(0, 2, 1) #(B, N_l, l_in_channels)


        x = x.permute(0, 2, 1)  # (B, v_in_channels, H*W)
        key = self.f_key(x) # (B, l_in_channels, H*W)
        query = query * l_mask  # (B, l_in_channels, N_l)
        query = query.reshape(B, n_l, self.num_heads, self.l_in_channels // self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, N_l, self.l_in_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.l_in_channels // self.num_heads, HW)
        # (b, num_heads, self.l_in_channels//self.num_heads, HW)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, n_l, 1)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, N_l, H*W)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4 * l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, self.num_heads, N_l, H*W)

        return sim_map



class No_Fusion_Block(nn.Module):
    def __init__(self, num_heads=1, dropout=0.0):
        super(No_Fusion_Block, self).__init__()
        # input x shape: (B, H*W, dim)

    def forward(self, x, l, l_mask):

        return x, l
    




class ALL_Fusion_Block_Add(nn.Module):
    def __init__(self, dim,num_heads=1, dropout=0.0):
        super(ALL_Fusion_Block_Add, self).__init__()
        # input x shape: (B, H*W, dim)
        # self.fusion = PWAM(dim,  # both the visual input and for combining, num of channels
        #                    dim,  # v_in
        #                    256,  # l_in
        #                    dim,  # key
        #                    dim,  # value
        #                    num_heads=num_heads,
        #                    dropout=dropout)
        # self.res_gate = nn.Sequential(
        #     nn.Linear(dim, dim, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(dim, dim, bias=False),
        #     nn.Tanh()
        # )
        # self.W_l = nn.Sequential(
        #     nn.Conv1d(256, 256, 1, 1),  # the init function sets bias to 0 if bias is True
        #     nn.GELU()
        # )
        
        self.align_linear = nn.Linear(256,dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.out_linear=nn.Linear(dim,256)

    def forward(self, x, l, l_mask):
        # L BT C 1
        BT,C,H, W = x.shape # BT C H W  
        x_ = x
        l = self.align_linear(l.squeeze(-1)) #[BT C H W]
        l_ = l 
        l_visual = repeat(l_,'bt c -> bt c h w',h=H,w=W)
        x+=l_visual
        x_audio = self.avgpool(x_).squeeze(-1).squeeze(-1)
        l+=x_audio
        l = self.out_linear(l).unsqueeze(2)
        
        
        return x, l
    



class ALL_Fusion_Block_Concat(nn.Module):
    def __init__(self, dim,num_heads=1, dropout=0.0):
        super(ALL_Fusion_Block_Concat, self).__init__()
        self.align_linear = nn.Linear(256,dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.out_conv = nn.Conv2d(2*dim,dim,kernel_size=1)
        self.out_linear=nn.Linear(2*dim,256)

    def forward(self, x, l, l_mask):
        # L BT C 1
        BT,C,H, W = x.shape # BT C H W  
        x_ = x
        l = self.align_linear(l.squeeze(-1)) #[BT C H W]
        l_ = l 
        l_visual = repeat(l_,'bt c -> bt c h w',h=H,w=W)
        # x+=l_visual
        
        x = torch.concat([x,l_visual],dim=1)
        x = self.out_conv(x)
        
        x_audio = self.avgpool(x_).squeeze(-1).squeeze(-1)
        
        # l+=x_audio
        l = torch.concat([l,x_audio],dim=1)
        l = self.out_linear(l).unsqueeze(2)
        
        
        return x, l

