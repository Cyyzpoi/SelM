import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class DAM_Fusion_Block(nn.Module):
    def __init__(self, dim, num_heads=1, dropout=0.0):
        super(DAM_Fusion_Block, self).__init__()
        self.fusion = Audio_Visual_Dual_Fusion(dim,  
                                               dim,  # vis_dim
                                               256,  # audio_dim
                                               dim,  # key_dim
                                               dim,  # value_dim
                                               num_heads=num_heads,
                                               dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )
        self.audio_out_conv = nn.Sequential(
            # the init function sets bias to 0 if bias is True
            nn.Conv1d(256, 256, 1, 1),
            nn.GELU()
        )

    def forward(self, x, audio, audio_mask):
        # x (BT C H W)
        H, W = x.shape[2], x.shape[3]
        x = x.view(x.shape[0], x.shape[1], H*W)

        x = x.permute(0, 2, 1).contiguous()
        # x (BT H*W C)
        x_residual, audio_residual = self.fusion(x, audio, audio_mask)
        # With gate and residual
        x = x + (self.gate(x_residual) * x_residual)
        audio = audio + self.audio_out_conv(audio_residual)

        x = x.permute(0, 2, 1).contiguous()
        x = x.view(x.shape[0], x.shape[1], H, W)  # (BT C H W)

        return x, audio


class Audio_Visual_Dual_Fusion(nn.Module):
    def __init__(self, dim, vis_dim, audio_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(Audio_Visual_Dual_Fusion, self).__init__()

        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                         )

        self.audio_visual_att = AV_Interaction(vis_dim,  
                                               audio_channels,  
                                               key_channels,  
                                               value_channels,  
                                               out_channels=value_channels,  
                                               num_heads=num_heads)

        self.out_conv = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                      nn.GELU(),
                                      nn.Dropout(dropout)
                                      )

    def forward(self, x, audio, audio_mask):
        # input x shape: (B, H*W, dim)

        vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)

        vis_weight, audio = self.audio_visual_att(
            x, audio, audio_mask)  # (B, H*W, dim) (B, l_dim, N_l)

        vis_weight = vis_weight.permute(0, 2, 1)  # (B, dim, H*W)

        vis = torch.mul(vis, vis_weight)
        vis = self.out_conv(vis)  # (B, dim, H*W)

        vis = vis.permute(0, 2, 1)  # (B, H*W, dim)

        return vis, audio


class AV_Interaction(nn.Module):
    def __init__(self, vis_dim, audio_dim, key_channels, value_channels, out_channels=None, num_heads=1):
        super(AV_Interaction, self).__init__()

        self.vis_dim = vis_dim
        self.audio_dim = audio_dim
        self.out_dim = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.value_channels_audio = audio_dim
        if out_channels is None:
            self.out_dim = self.value_channels

        # Values: Audio features: (B, audio_dim, audio_length)

        self.audio_value_conv = nn.Sequential(
            nn.Conv1d(self.audio_dim, self.value_channels,
                      kernel_size=1, stride=1),
        )

        # Values: Visual features: (B, H*W, vis_dim)

        self.vis_value_conv = nn.Sequential(
            nn.Conv1d(self.vis_dim, self.value_channels_audio,
                      kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.value_channels_audio),
        )

        # Out projection
        self.vis_out = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_dim,
                      kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_dim),
        )
        self.audio_out = nn.Sequential(
            nn.Conv1d(self.audio_dim, self.audio_dim, kernel_size=1, stride=1),
        )
        self.num_heads = num_heads
        self.align_vis = Visual_Sim(
            self.vis_dim, self.audio_dim, self.key_channels, kernel=1, num_heads=1)

        self.align_audio = Audio_Sim(
            self.vis_dim, self.audio_dim, self.key_channels, kernel=(1, 1), num_heads=1)


        self.vis_weight = nn.Parameter(torch.ones(1))
        self.audio_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, audio, audio_mask):

        # x shape: (BT, H*W, vis_dim)
        # audio shape: (BT, audio_channels, audio_length)
        # l_mask shape: (BT, audio_length, 1)

        BT, HW = x.size(0), x.size(1)
        audio_length = audio.size(2)
        # (BT, audio_length, 1) -> (BT, 1, audio_length)
        audio_mask_ = audio_mask.permute(0, 2, 1)

        # (B, self.value_channels, audio_length)
        audio_value = self.audio_value_conv(audio)
        audio_value = audio_value * audio_mask_  # (B, self.value_channels, audio_length)
        audio_length = audio_value.size(-1)
        audio_value = audio_value.reshape(
            BT, self.num_heads, self.value_channels // self.num_heads, audio_length)
        # (BT, num_heads, self.value_channels//self.num_heads, audio_length)

        vis_sim_map = self.align_vis(x, audio, audio_mask)


        vis_weight1 = F.softmax(self.vis_weight, dim=0)
        vis_sim_map = vis_weight1*vis_sim_map

        # (BT, num_heads, H*W, self.value_channels//num_heads)
        vis_out = torch.matmul(vis_sim_map, audio_value.permute(0, 1, 3, 2))
        vis_out = vis_out.permute(0, 2, 1, 3).contiguous().reshape(
            BT, HW, self.value_channels)  # (BT, H*W, value_channels)
        vis_out = vis_out.permute(0, 2, 1)  # (BT, value_channels, HW)
        vis_out = self.vis_out(vis_out)  # (BT, value_channels, HW)
        vis_out = vis_out.permute(0, 2, 1)  # (BT, HW, value_channels)

        vis_value = x.permute(0, 2, 1)  # (BT, vis_dim, H*W)
        vis_value = self.vis_value_conv(vis_value)  # (BT, value_channels_audio, H*W)
        value_v = vis_value.reshape(
            BT, self.num_heads, self.audio_dim // self.num_heads, HW)
        # (BT, num_heads, self.value_channels_audio//self.num_heads, H*W)

        audio_sim_map = self.align_audio(x, audio, audio_mask)


        audio_weight = F.softmax(self.audio_weight, dim=0)
        audio_sim_map = audio_sim_map*audio_weight

        # (B, num_heads, audio_length, self.audio_dim//num_heads)
        audio_out = torch.matmul(audio_sim_map, value_v.permute(0, 1, 3, 2))
        audio_out = audio_out.permute(0, 2, 1, 3).contiguous().reshape(
            BT, audio_length, self.audio_dim)  # (BT, audio_length, audio_dim)
        audio_out = audio_out.permute(0, 2, 1)  # (BT, audio_dim, audio_length)
        audio_out = self.audio_out(audio_out)  # (BT, audio_dim, audio_length)

        return vis_out, audio_out


class Visual_Sim(nn.Module):
    def __init__(self, vis_dim, audio_dim, key_channels, kernel, num_heads=1):
        super(Visual_Sim, self).__init__()


        self.vis_dim = vis_dim
        self.audio_dim = audio_dim
        if kernel == 1:
            self.int_channels = key_channels
        elif kernel == 3:
            self.int_channels = key_channels // 2
        elif kernel == 5:
            self.int_channels = key_channels // 4
            
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.kernel = kernel

        # Keys: Audio features: (B, audio_dim, audio_length)
        
        self.key_conv = nn.Sequential(
            nn.Conv1d(self.audio_dim, self.key_channels,
                      kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, vis_dim)
        self.query_conv = nn.Sequential(
            nn.Conv1d(self.vis_dim, self.int_channels,
                      kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.int_channels),
        )
        self.query_conv2 = nn.Sequential(
            nn.Conv1d(self.int_channels * (self.kernel ** 2),
                      self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

    def forward(self, x, audio, audio_mask):
        
        # x shape: (BT, H*W, vis_dim)
        # audio shape: (BT, audio_channels, audio_length)
        # l_mask shape: (BT, audio_length, 1)
        
        BT, HW = x.size(0), x.size(1)
        audio_length = audio.size(2)
        audio_mask = audio_mask.permute(0, 2, 1)  # (BT, audio_length, 1) -> (B, 1, audio_length)

        x = x.permute(0, 2, 1)  # (B, vis_dim, H*W)
        x1 = self.query_conv(x)
        x1 = rearrange(x1, 'b c (h w) -> b c h w',
                       h=int(math.sqrt(x.shape[2])))
        x2 = F.unfold(x1, kernel_size=self.kernel,
                      stride=1, padding=self.kernel//2)

        query = self.query_conv2(x2)  # (BT, key_channels, H*W) 
        query = query.permute(0, 2, 1)  # (BT, H*W, key_channels)
        key = self.key_conv(audio)  # (BT, key_channels, audio_length)
        key = key * audio_mask  # (B, key_channels, audio_length)
        query = query.reshape(
            BT, HW, self.num_heads, self.key_channels // self.num_heads).permute(0, 2, 1, 3)
        # (BT, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(BT, self.num_heads,
                          self.key_channels // self.num_heads, audio_length)
        # (BT, num_heads, self.key_channels//self.num_heads, audio_length)
        audio_mask = audio_mask.unsqueeze(1)  # (BT, 1, 1, audio_length)

        sim_map = torch.matmul(query, key)  # (BT, self.num_heads, H*W, audio_length)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        # assign a very small number to padding positions
        sim_map = sim_map + (1e4 * audio_mask - 1e4)
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, H*W, audio_length)

        return sim_map


class Audio_Sim(nn.Module):
    def __init__(self, vis_dim, audio_dim, key_channels, kernel, num_heads=1):
        super(Audio_Sim, self).__init__()


        self.vis_dim = vis_dim
        self.audio_dim = audio_dim
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.kernel = kernel
        if self.kernel[0] == 1:
            self.int_channels = key_channels
        elif self.kernel[0] == 2:
            self.int_channels = key_channels // 2
        elif self.kernel[0] == 3:
            self.int_channels = key_channels // 3

        # Keys: Visual features: (B, vis_dim, H*W)
        self.f_key = nn.Sequential(
            nn.Conv1d(self.vis_dim, self.audio_dim,
                      kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.audio_dim),
        )

        # Queries: Audio features: (B, audio_dim, audio_length)

        self.f_query = nn.Sequential(
            nn.Conv1d(self.audio_dim, self.int_channels,
                      kernel_size=1, stride=1),
        )
        self.f_query2 = nn.Sequential(
            nn.Conv1d(self.int_channels *
                      self.kernel[0], self.audio_dim, kernel_size=1, stride=1),
        )

    def forward(self, x, audio, audio_mask):
        # x shape: (BT, H*W, vis_dim)
        # audio shape: (BT, audio_channels, audio_length)
        # l_mask shape: (BT, audio_length, 1)

        BT, HW = x.size(0), x.size(1)
        audio_length = audio.size(2)

        audio_1 = self.f_query(audio)  # (BT, int_channels, audio_length)
        audio_1 = audio_1.unsqueeze(3)  # (BT, int_channels, audio_length, 1)
        # (BT, int_channels, audio_length, 1)
        audio_1 = F.pad(
            audio_1, (0, 0, self.kernel[0]//2, (self.kernel[0]-1)//2), mode='replicate')
        # (BT, int_channels*self.kernel[0], audio_length)
        audio_2 = F.unfold(audio_1, kernel_size=(self.kernel[0], 1), stride=1)
        query = self.f_query2(audio_2)  # (BT, audio_dim, audio_length)
        query = query.permute(0, 2, 1)  # (BT, audio_length, audio_dim)

        x = x.permute(0, 2, 1)  # (BT, vis_dim, H*W)
        key = self.f_key(x)  # (BT, audio_dim, H*W)
        query = query * audio_mask  # (BT, audio_dim, audio_length)
        query = query.reshape(
            BT, audio_length, self.num_heads, self.audio_dim // self.num_heads).permute(0, 2, 1, 3)
        # (BT, num_heads, audio_length, self.audio_dim//self.num_heads)
        key = key.reshape(BT, self.num_heads,
                          self.audio_dim // self.num_heads, HW)
        # (BT, num_heads, self.audio_dim//self.num_heads, HW)
        audio_mask = audio_mask.unsqueeze(1)  # (BT, 1, audio_length, 1)

        sim_map = torch.matmul(query, key)  # (BT, self.num_heads, audio_length, H*W)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        # assign a very small number to padding positions
        sim_map = sim_map + (1e4 * audio_mask - 1e4)
        sim_map = F.softmax(sim_map, dim=-1)  # (BT, self.num_heads, audio_length, H*W)

        return sim_map
