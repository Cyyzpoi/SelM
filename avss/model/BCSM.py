import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from mamba_ssm import Mamba


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Mamba_channel_temporal_block(nn.Module):
    def __init__(self, dim):
        super(Mamba_channel_temporal_block, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.mamba_block = Mamba(d_model=dim, device='cuda')

    def forward(self, feature):
        # feature [B T C]
        feature = self.layernorm(feature)
        feature = feature+self.mamba_block(feature)
        return feature


class Encoder(nn.Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        r"""Pass the input through the endocder layers in turn.

        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output


class BCSM(nn.Module):
    def __init__(self, gamma=1, dim=256):
        super(BCSM, self).__init__()
        self.gamma = gamma

        self.d_model = dim

        self.vis_linear = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_model) for i in range(4)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.video_encoder = nn.ModuleList(
            [Encoder(Mamba_channel_temporal_block(dim=256), num_layers=3) for i in range(4)])
        self.audio_encoder = nn.ModuleList(
            [Encoder(Mamba_channel_temporal_block(dim=256), num_layers=3) for i in range(4)])

        self.audio_gated = nn.ModuleList([nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(inplace=True)
        ) for i in range(4)])
        self.video_gated = nn.ModuleList([nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(inplace=True)
        ) for i in range(4)])

    def forward(self, visual_feature_list, audio_feature):

        # Shape for Features
        # BT x 256 x 56 x 56
        # BT x 256 x 28 x 28
        # BT x 256 x 14 x 14
        # BT x 256 x  7 x  7

        x1, x2, x3, x4 = visual_feature_list
        bt = x1.shape[0]
        x1_ = self.avgpool(x1)
        x1_ = x1_.squeeze()
        x2_ = self.avgpool(x2)
        x2_ = x2_.squeeze()
        x3_ = self.avgpool(x3)
        x3_ = x3_.squeeze()
        x4_ = self.avgpool(x4)
        x4_ = x4_.squeeze()

        x1_ = x1_.view(bt//10, 10, -1)  # [B 5 256]
        x2_ = x2_.view(bt//10, 10, -1)  # [B 5 256]
        x3_ = x3_.view(bt//10, 10, -1)  # [B 5 256]
        x4_ = x4_.view(bt//10, 10, -1)  # [B 5 256]

        audio_input = audio_feature  # [B T 256]
        # [BT 256]
        audio_feature = audio_feature.view(-1, audio_feature.size(-1))
        x1_, x2_, x3_, x4_ = [self.vis_linear[i](x) for i, x in enumerate(
            [x1_, x2_, x3_, x4_])]  # [B T 256]

        audio_out1 = self.audio_encoder[0](audio_input)
        audio_out2 = self.audio_encoder[1](audio_input)
        audio_out3 = self.audio_encoder[2](audio_input)
        audio_out4 = self.audio_encoder[3](audio_input)

        video_out1 = self.video_encoder[0](x1_)
        video_out2 = self.video_encoder[1](x2_)
        video_out3 = self.video_encoder[2](x3_)
        video_out4 = self.video_encoder[3](x4_)

        audio_gate1 = self.audio_gated[0](
            audio_out1)  # [B, T, 256]
        audio_gate2 = self.audio_gated[1](audio_out2)
        audio_gate3 = self.audio_gated[2](audio_out3)
        audio_gate4 = self.audio_gated[3](audio_out4)

        video_gate1 = self.video_gated[0](
            video_out1)  # [B, T, 256]
        video_gate2 = self.video_gated[1](video_out2)
        video_gate3 = self.video_gated[2](video_out3)
        video_gate4 = self.video_gated[3](video_out4)

        audio_gate1 = audio_gate1.reshape(bt, self.d_model, 1, 1)
        audio_gate2 = audio_gate2.transpose(1, 0)
        audio_gate2 = audio_gate2.reshape(bt, self.d_model, 1, 1)
        audio_gate3 = audio_gate3.transpose(1, 0)
        audio_gate3 = audio_gate3.reshape(bt, self.d_model, 1, 1)
        audio_gate4 = audio_gate4.transpose(1, 0)
        audio_gate4 = audio_gate4.reshape(bt, self.d_model, 1, 1)

        video_gate1 = video_gate1.transpose(1, 0)
        video_gate1 = video_gate1.reshape(bt, self.d_model)
        video_gate2 = video_gate2.transpose(1, 0)
        video_gate2 = video_gate2.reshape(bt, self.d_model)
        video_gate3 = video_gate3.transpose(1, 0)
        video_gate3 = video_gate3.reshape(bt, self.d_model)
        video_gate4 = video_gate4.transpose(1, 0)
        video_gate4 = video_gate4.reshape(bt, self.d_model)

        x1 = x1 + audio_gate1 * x1 * self.gamma
        x2 = x2 + audio_gate2 * x2 * self.gamma
        x3 = x3 + audio_gate3 * x3 * self.gamma
        x4 = x4 + audio_gate4 * x4 * self.gamma

        video_gate = (video_gate1 + video_gate2 +
                      video_gate3 + video_gate4) / 4
        audio_feature = audio_feature + video_gate * audio_feature * self.gamma

        return [x4, x3, x2, x1], audio_feature.unsqueeze(2)
