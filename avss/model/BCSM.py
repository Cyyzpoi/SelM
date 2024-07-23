import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from mamba_ssm import Mamba


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Mamba_channel_temporal_block(nn.Module):
	def __init__(self,dim):
		super(Mamba_channel_temporal_block,self).__init__()
		self.layernorm = nn.LayerNorm(dim)
		self.mamba_block = Mamba(d_model=dim,device='cuda')
	def forward(self,feature):
		#feature [B T C]
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

		self.v_fc = nn.ModuleList([nn.Linear(256,self.d_model) for i in range(4)])
        
		
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		

  
		self.video_encoder = nn.ModuleList([Encoder(Mamba_channel_temporal_block(dim=256),num_layers=3) for i in range(4)])
		self.audio_encoder = nn.ModuleList([Encoder(Mamba_channel_temporal_block(dim=256),num_layers=3) for i in range(4)])
  
  
  
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
        
		x1_ = x1_.view(bt//5, 5, -1)
		x2_ = x2_.view(bt//5, 5, -1)
		x3_ = x3_.view(bt//5, 5, -1)
		x4_ = x4_.view(bt//5, 5, -1)

		# optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
		audio_rnn_input = audio_feature
		audio_feature = audio_feature.view(-1, audio_feature.size(-1))
		x1_, x2_, x3_, x4_ = [self.v_fc[i](x) for i, x in enumerate([x1_, x2_, x3_, x4_])]
  
		audio_key_value_feature1 = self.audio_encoder[0](audio_rnn_input)
		audio_key_value_feature2 = self.audio_encoder[1](audio_rnn_input)
		audio_key_value_feature3 = self.audio_encoder[2](audio_rnn_input)
		audio_key_value_feature4 = self.audio_encoder[3](audio_rnn_input)
  
		video_key_value_feature1 = self.video_encoder[0](x1_)
		video_key_value_feature2 = self.video_encoder[1](x2_)
		video_key_value_feature3 = self.video_encoder[2](x3_)
		video_key_value_feature4 = self.video_encoder[3](x4_)

		audio_gate1 = self.audio_gated[0](audio_key_value_feature1) # [B, T, C]
		audio_gate2 = self.audio_gated[1](audio_key_value_feature2)
		audio_gate3 = self.audio_gated[2](audio_key_value_feature3)
		audio_gate4 = self.audio_gated[3](audio_key_value_feature4)
        
		video_gate1 = self.video_gated[0](video_key_value_feature1) # [5, B, 1]
		video_gate2 = self.video_gated[1](video_key_value_feature2)
		video_gate3 = self.video_gated[2](video_key_value_feature3)
		video_gate4 = self.video_gated[3](video_key_value_feature4)

		# audio_gate1 = audio_gate1.transpose(1, 0)
		audio_gate1 = audio_gate1.reshape(bt, 256, 1, 1)
		audio_gate2 = audio_gate2.transpose(1, 0)
		audio_gate2 = audio_gate2.reshape(bt, 256, 1, 1)
		audio_gate3 = audio_gate3.transpose(1, 0)
		audio_gate3 = audio_gate3.reshape(bt, 256, 1, 1)
		audio_gate4 = audio_gate4.transpose(1, 0)
		audio_gate4 = audio_gate4.reshape(bt, 256, 1, 1)

		video_gate1 = video_gate1.transpose(1, 0)
		video_gate1 = video_gate1.reshape(bt, 256)
		video_gate2 = video_gate2.transpose(1, 0)
		video_gate2 = video_gate2.reshape(bt, 256)
		video_gate3 = video_gate3.transpose(1, 0)
		video_gate3 = video_gate3.reshape(bt, 256)
		video_gate4 = video_gate4.transpose(1, 0)
		video_gate4 = video_gate4.reshape(bt, 256)
        # gamma = F.softmax(gamma)
		x1 = x1 + audio_gate1 * x1 * self.gamma
		x2 = x2 + audio_gate2 * x2 * self.gamma
		x3 = x3 + audio_gate3 * x3 * self.gamma
		x4 = x4 + audio_gate4 * x4 * self.gamma
  
        
		video_gate = (video_gate1 + video_gate2 + video_gate3 + video_gate4) / 4
		audio_feature = audio_feature + video_gate * audio_feature * self.gamma

		return [x4, x3, x2, x1], audio_feature.unsqueeze(2)




class TemporalAttention_A2V(nn.Module):
	def __init__(self):
		super(TemporalAttention_A2V, self).__init__()
		self.gamma = 1
		self.video_input_dim = 256
		self.audio_input_dim = 256

		self.video_fc_dim = 256
		self.audio_fc_dim = 768
		self.d_model = 256

		self.v_fc = nn.ModuleList([nn.Linear(256,self.video_fc_dim) for i in range(4)])
        
		# self.relu = nn.ReLU()
		# self.dropout = nn.Dropout(0.2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		# self.video_encoder = nn.ModuleList([InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
		# self.video_encoder = nn.ModuleList([Mamba(d_model=256,device='cuda') for i in range(4)])
		# self.video_decoder = nn.ModuleList([CrossModalRelationAttModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024) for i in range(4)]) 
		# self.audio_encoder = nn.ModuleList([InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
		# self.audio_encoder = nn.ModuleList([Mamba(d_model=256,device='cuda') for i in range(4)])
		# self.audio_decoder = nn.ModuleList([CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
		# self.audio_visual_rnn_layer = nn.ModuleList([RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1) for i in range(4)])

  
		# self.video_encoder = nn.ModuleList([Encoder(Mamba_channel_temporal_block(dim=256),num_layers=3) for i in range(4)])
		self.audio_encoder = nn.ModuleList([Encoder(Mamba_channel_temporal_block(dim=256),num_layers=3) for i in range(4)])
  
  
		# self.v_mamba = nn.ModuleList([Encoder(Mamba(d_model=256,device='cuda'),2) for i in range(4)])
		# self.a_mamba = nn.ModuleList([Encoder(Mamba(d_model=256,device='cuda'),2) for i in range(4)])
  
		# self.v_mamba = nn.ModuleList([Mamba(d_model=256,device='cuda') for i in range(4)])
		# self.a_mamba = nn.ModuleList([Mamba(d_model=256,device='cuda') for i in range(4)])
  
		self.audio_gated = nn.ModuleList([nn.Sequential(
						nn.Linear(self.d_model, self.d_model),
						nn.SiLU(inplace=True)
					) for i in range(4)])
		# self.video_gated = nn.ModuleList([nn.Sequential(
		# 				nn.Linear(self.d_model, self.d_model),
		# 				nn.SiLU(inplace=True)
		# 			) for i in range(4)])
        
	def forward(self, visual_feature_list, audio_feature):
		# shape for pvt-v2-b5
		# BF x 256 x 56 x 56
		# BF x 256 x 28 x 28
		# BF x 256 x 14 x 14
		# BF x 256 x  7 x  7

		
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
        
		x1_ = x1_.view(bt//5, 5, -1)
		x2_ = x2_.view(bt//5, 5, -1)
		x3_ = x3_.view(bt//5, 5, -1)
		x4_ = x4_.view(bt//5, 5, -1)

		# optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
		audio_rnn_input = audio_feature
		audio_feature = audio_feature.view(-1, audio_feature.size(-1))
		x1_, x2_, x3_, x4_ = [self.v_fc[i](x) for i, x in enumerate([x1_, x2_, x3_, x4_])]
		# x1_, x2_, x3_, x4_ = [self.dropout(self.relu(x)) for x in [x1_, x2_, x3_, x4_]]
  
		# x1_,
        
		# visual_rnn_input = [x1_, x2_, x3_, x4_]

		# audio_rnn_output1 = self.a_mamba[0](audio_rnn_input)
		# audio_rnn_output2 = self.a_mamba[1](audio_rnn_input)
		# audio_rnn_output3 = self.a_mamba[2](audio_rnn_input)
		# audio_rnn_output4 = self.a_mamba[3](audio_rnn_input)
  
		# visual_rnn_output1 = self.v_mamba[0](visual_rnn_input[0])
		# visual_rnn_output2 = self.v_mamba[1](visual_rnn_input[1])
		# visual_rnn_output3 = self.v_mamba[2](visual_rnn_input[2])
		# visual_rnn_output4 = self.v_mamba[3](visual_rnn_input[3])
  
		# # audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer[0](audio_rnn_input, visual_rnn_input[0])
		# # audio_rnn_output2, visual_rnn_output2 = self.audio_visual_rnn_layer[1](audio_rnn_input, visual_rnn_input[1])
		# # audio_rnn_output3, visual_rnn_output3 = self.audio_visual_rnn_layer[2](audio_rnn_input, visual_rnn_input[2])
		# # audio_rnn_output4, visual_rnn_output4 = self.audio_visual_rnn_layer[3](audio_rnn_input, visual_rnn_input[3])
        
		# audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [5, B, 256]
		# audio_encoder_input2 = audio_rnn_output2.transpose(1, 0).contiguous()  # [5, B, 256]
		# audio_encoder_input3 = audio_rnn_output3.transpose(1, 0).contiguous()  # [5, B, 256]
		# audio_encoder_input4 = audio_rnn_output4.transpose(1, 0).contiguous()  # [5, B, 256]
        
		# visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [5, B, 512]
		# visual_encoder_input2 = visual_rnn_output2.transpose(1, 0).contiguous()  # [5, B, 512]
		# visual_encoder_input3 = visual_rnn_output3.transpose(1, 0).contiguous()  # [5, B, 512]
		# visual_encoder_input4 = visual_rnn_output4.transpose(1, 0).contiguous()  # [5, B, 512]

		# # audio query
		# video_key_value_feature1 = self.video_encoder[0](visual_encoder_input1)
		# video_key_value_feature2 = self.video_encoder[1](visual_encoder_input2)
		# video_key_value_feature3 = self.video_encoder[2](visual_encoder_input3)
		# video_key_value_feature4 = self.video_encoder[3](visual_encoder_input4)
        
		# # audio_query_output1 = self.audio_decoder[0](audio_encoder_input1, video_key_value_feature1)
		# # audio_query_output2 = self.audio_decoder[1](audio_encoder_input2, video_key_value_feature2)
		# # audio_query_output3 = self.audio_decoder[2](audio_encoder_input3, video_key_value_feature3)
		# # audio_query_output4 = self.audio_decoder[3](audio_encoder_input4, video_key_value_feature4)
        
		# # video query
		# audio_key_value_feature1 = self.audio_encoder[0](audio_encoder_input1)
		# audio_key_value_feature2 = self.audio_encoder[1](audio_encoder_input2)
		# audio_key_value_feature3 = self.audio_encoder[2](audio_encoder_input3)
		# audio_key_value_feature4 = self.audio_encoder[3](audio_encoder_input4)
        
		# video_query_output1 = self.video_decoder[0](visual_encoder_input1, audio_key_value_feature1)
		# video_query_output2 = self.video_decoder[1](visual_encoder_input2, audio_key_value_feature2)
		# video_query_output3 = self.video_decoder[2](visual_encoder_input3, audio_key_value_feature3)
		# video_query_output4 = self.video_decoder[3](visual_encoder_input4, audio_key_value_feature4)
  
		audio_key_value_feature1 = self.audio_encoder[0](audio_rnn_input)
		audio_key_value_feature2 = self.audio_encoder[1](audio_rnn_input)
		audio_key_value_feature3 = self.audio_encoder[2](audio_rnn_input)
		audio_key_value_feature4 = self.audio_encoder[3](audio_rnn_input)
  
		# video_key_value_feature1 = self.video_encoder[0](x1_)
		# video_key_value_feature2 = self.video_encoder[1](x2_)
		# video_key_value_feature3 = self.video_encoder[2](x3_)
		# video_key_value_feature4 = self.video_encoder[3](x4_)

		audio_gate1 = self.audio_gated[0](audio_key_value_feature1) # [B, T, C]
		audio_gate2 = self.audio_gated[1](audio_key_value_feature2)
		audio_gate3 = self.audio_gated[2](audio_key_value_feature3)
		audio_gate4 = self.audio_gated[3](audio_key_value_feature4)
        
		# video_gate1 = self.video_gated[0](video_key_value_feature1) # [5, B, 1]
		# video_gate2 = self.video_gated[1](video_key_value_feature2)
		# video_gate3 = self.video_gated[2](video_key_value_feature3)
		# video_gate4 = self.video_gated[3](video_key_value_feature4)

		# audio_gate1 = audio_gate1.transpose(1, 0)
		audio_gate1 = audio_gate1.reshape(bt, 256, 1, 1)
		audio_gate2 = audio_gate2.transpose(1, 0)
		audio_gate2 = audio_gate2.reshape(bt, 256, 1, 1)
		audio_gate3 = audio_gate3.transpose(1, 0)
		audio_gate3 = audio_gate3.reshape(bt, 256, 1, 1)
		audio_gate4 = audio_gate4.transpose(1, 0)
		audio_gate4 = audio_gate4.reshape(bt, 256, 1, 1)

		# video_gate1 = video_gate1.transpose(1, 0)
		# video_gate1 = video_gate1.reshape(bt, 256)
		# video_gate2 = video_gate2.transpose(1, 0)
		# video_gate2 = video_gate2.reshape(bt, 256)
		# video_gate3 = video_gate3.transpose(1, 0)
		# video_gate3 = video_gate3.reshape(bt, 256)
		# video_gate4 = video_gate4.transpose(1, 0)
		# video_gate4 = video_gate4.reshape(bt, 256)
        # gamma = F.softmax(gamma)
		x1 = x1 + audio_gate1 * x1 * self.gamma
		x2 = x2 + audio_gate2 * x2 * self.gamma
		x3 = x3 + audio_gate3 * x3 * self.gamma
		x4 = x4 + audio_gate4 * x4 * self.gamma
  
        
		# video_gate = (video_gate1 + video_gate2 + video_gate3 + video_gate4) / 4
		# audio_feature = audio_feature + video_gate * audio_feature * self.gamma

		return [x4, x3, x2, x1], audio_feature.unsqueeze(2)







class TemporalAttention_V2A(nn.Module):
	def __init__(self):
		super(TemporalAttention_V2A, self).__init__()
		self.gamma = 1
		self.video_input_dim = 256
		self.audio_input_dim = 256

		self.video_fc_dim = 256
		self.audio_fc_dim = 768
		self.d_model = 256

		self.v_fc = nn.ModuleList([nn.Linear(256,self.video_fc_dim) for i in range(4)])
        
		# self.relu = nn.ReLU()
		# self.dropout = nn.Dropout(0.2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		# self.video_encoder = nn.ModuleList([InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
		# self.video_encoder = nn.ModuleList([Mamba(d_model=256,device='cuda') for i in range(4)])
		# self.video_decoder = nn.ModuleList([CrossModalRelationAttModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024) for i in range(4)]) 
		# self.audio_encoder = nn.ModuleList([InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
		# self.audio_encoder = nn.ModuleList([Mamba(d_model=256,device='cuda') for i in range(4)])
		# self.audio_decoder = nn.ModuleList([CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
		# self.audio_visual_rnn_layer = nn.ModuleList([RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1) for i in range(4)])

  
		self.video_encoder = nn.ModuleList([Encoder(Mamba_channel_temporal_block(dim=256),num_layers=3) for i in range(4)])
		# self.audio_encoder = nn.ModuleList([Encoder(Mamba_channel_temporal_block(dim=256),num_layers=3) for i in range(4)])
  
  
		# self.v_mamba = nn.ModuleList([Encoder(Mamba(d_model=256,device='cuda'),2) for i in range(4)])
		# self.a_mamba = nn.ModuleList([Encoder(Mamba(d_model=256,device='cuda'),2) for i in range(4)])
  
		# self.v_mamba = nn.ModuleList([Mamba(d_model=256,device='cuda') for i in range(4)])
		# self.a_mamba = nn.ModuleList([Mamba(d_model=256,device='cuda') for i in range(4)])
  
		# self.audio_gated = nn.ModuleList([nn.Sequential(
		# 				nn.Linear(self.d_model, self.d_model),
		# 				nn.SiLU(inplace=True)
		# 			) for i in range(4)])
		self.video_gated = nn.ModuleList([nn.Sequential(
						nn.Linear(self.d_model, self.d_model),
						nn.SiLU(inplace=True)
					) for i in range(4)])
        
	def forward(self, visual_feature_list, audio_feature):
		# shape for pvt-v2-b5
		# BF x 256 x 56 x 56
		# BF x 256 x 28 x 28
		# BF x 256 x 14 x 14
		# BF x 256 x  7 x  7

		
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
        
		x1_ = x1_.view(bt//5, 5, -1)
		x2_ = x2_.view(bt//5, 5, -1)
		x3_ = x3_.view(bt//5, 5, -1)
		x4_ = x4_.view(bt//5, 5, -1)

		# optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
		audio_rnn_input = audio_feature
		audio_feature = audio_feature.view(-1, audio_feature.size(-1))
		x1_, x2_, x3_, x4_ = [self.v_fc[i](x) for i, x in enumerate([x1_, x2_, x3_, x4_])]
		# x1_, x2_, x3_, x4_ = [self.dropout(self.relu(x)) for x in [x1_, x2_, x3_, x4_]]
  
		# x1_,
        
		# visual_rnn_input = [x1_, x2_, x3_, x4_]

		# audio_rnn_output1 = self.a_mamba[0](audio_rnn_input)
		# audio_rnn_output2 = self.a_mamba[1](audio_rnn_input)
		# audio_rnn_output3 = self.a_mamba[2](audio_rnn_input)
		# audio_rnn_output4 = self.a_mamba[3](audio_rnn_input)
  
		# visual_rnn_output1 = self.v_mamba[0](visual_rnn_input[0])
		# visual_rnn_output2 = self.v_mamba[1](visual_rnn_input[1])
		# visual_rnn_output3 = self.v_mamba[2](visual_rnn_input[2])
		# visual_rnn_output4 = self.v_mamba[3](visual_rnn_input[3])
  
		# # audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer[0](audio_rnn_input, visual_rnn_input[0])
		# # audio_rnn_output2, visual_rnn_output2 = self.audio_visual_rnn_layer[1](audio_rnn_input, visual_rnn_input[1])
		# # audio_rnn_output3, visual_rnn_output3 = self.audio_visual_rnn_layer[2](audio_rnn_input, visual_rnn_input[2])
		# # audio_rnn_output4, visual_rnn_output4 = self.audio_visual_rnn_layer[3](audio_rnn_input, visual_rnn_input[3])
        
		# audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [5, B, 256]
		# audio_encoder_input2 = audio_rnn_output2.transpose(1, 0).contiguous()  # [5, B, 256]
		# audio_encoder_input3 = audio_rnn_output3.transpose(1, 0).contiguous()  # [5, B, 256]
		# audio_encoder_input4 = audio_rnn_output4.transpose(1, 0).contiguous()  # [5, B, 256]
        
		# visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [5, B, 512]
		# visual_encoder_input2 = visual_rnn_output2.transpose(1, 0).contiguous()  # [5, B, 512]
		# visual_encoder_input3 = visual_rnn_output3.transpose(1, 0).contiguous()  # [5, B, 512]
		# visual_encoder_input4 = visual_rnn_output4.transpose(1, 0).contiguous()  # [5, B, 512]

		# # audio query
		# video_key_value_feature1 = self.video_encoder[0](visual_encoder_input1)
		# video_key_value_feature2 = self.video_encoder[1](visual_encoder_input2)
		# video_key_value_feature3 = self.video_encoder[2](visual_encoder_input3)
		# video_key_value_feature4 = self.video_encoder[3](visual_encoder_input4)
        
		# # audio_query_output1 = self.audio_decoder[0](audio_encoder_input1, video_key_value_feature1)
		# # audio_query_output2 = self.audio_decoder[1](audio_encoder_input2, video_key_value_feature2)
		# # audio_query_output3 = self.audio_decoder[2](audio_encoder_input3, video_key_value_feature3)
		# # audio_query_output4 = self.audio_decoder[3](audio_encoder_input4, video_key_value_feature4)
        
		# # video query
		# audio_key_value_feature1 = self.audio_encoder[0](audio_encoder_input1)
		# audio_key_value_feature2 = self.audio_encoder[1](audio_encoder_input2)
		# audio_key_value_feature3 = self.audio_encoder[2](audio_encoder_input3)
		# audio_key_value_feature4 = self.audio_encoder[3](audio_encoder_input4)
        
		# video_query_output1 = self.video_decoder[0](visual_encoder_input1, audio_key_value_feature1)
		# video_query_output2 = self.video_decoder[1](visual_encoder_input2, audio_key_value_feature2)
		# video_query_output3 = self.video_decoder[2](visual_encoder_input3, audio_key_value_feature3)
		# video_query_output4 = self.video_decoder[3](visual_encoder_input4, audio_key_value_feature4)
  
		# audio_key_value_feature1 = self.audio_encoder[0](audio_rnn_input)
		# audio_key_value_feature2 = self.audio_encoder[1](audio_rnn_input)
		# audio_key_value_feature3 = self.audio_encoder[2](audio_rnn_input)
		# audio_key_value_feature4 = self.audio_encoder[3](audio_rnn_input)
  
		video_key_value_feature1 = self.video_encoder[0](x1_)
		video_key_value_feature2 = self.video_encoder[1](x2_)
		video_key_value_feature3 = self.video_encoder[2](x3_)
		video_key_value_feature4 = self.video_encoder[3](x4_)

		# audio_gate1 = self.audio_gated[0](audio_key_value_feature1) # [B, T, C]
		# audio_gate2 = self.audio_gated[1](audio_key_value_feature2)
		# audio_gate3 = self.audio_gated[2](audio_key_value_feature3)
		# audio_gate4 = self.audio_gated[3](audio_key_value_feature4)
        
		video_gate1 = self.video_gated[0](video_key_value_feature1) # [5, B, 1]
		video_gate2 = self.video_gated[1](video_key_value_feature2)
		video_gate3 = self.video_gated[2](video_key_value_feature3)
		video_gate4 = self.video_gated[3](video_key_value_feature4)

		# audio_gate1 = audio_gate1.transpose(1, 0)
		# audio_gate1 = audio_gate1.reshape(bt, 256, 1, 1)
		# audio_gate2 = audio_gate2.transpose(1, 0)
		# audio_gate2 = audio_gate2.reshape(bt, 256, 1, 1)
		# audio_gate3 = audio_gate3.transpose(1, 0)
		# audio_gate3 = audio_gate3.reshape(bt, 256, 1, 1)
		# audio_gate4 = audio_gate4.transpose(1, 0)
		# audio_gate4 = audio_gate4.reshape(bt, 256, 1, 1)

		video_gate1 = video_gate1.transpose(1, 0)
		video_gate1 = video_gate1.reshape(bt, 256)
		video_gate2 = video_gate2.transpose(1, 0)
		video_gate2 = video_gate2.reshape(bt, 256)
		video_gate3 = video_gate3.transpose(1, 0)
		video_gate3 = video_gate3.reshape(bt, 256)
		video_gate4 = video_gate4.transpose(1, 0)
		video_gate4 = video_gate4.reshape(bt, 256)
        # gamma = F.softmax(gamma)
		# x1 = x1 + audio_gate1 * x1 * self.gamma
		# x2 = x2 + audio_gate2 * x2 * self.gamma
		# x3 = x3 + audio_gate3 * x3 * self.gamma
		# x4 = x4 + audio_gate4 * x4 * self.gamma
  
        
		video_gate = (video_gate1 + video_gate2 + video_gate3 + video_gate4) / 4
		audio_feature = audio_feature + video_gate * audio_feature * self.gamma

		return [x4, x3, x2, x1], audio_feature.unsqueeze(2)