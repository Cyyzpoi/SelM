import torch
import torch.nn as nn
import torchvision.models as models
import os
import torch.nn.functional as F
from model.pvt import pvt_v2_b5
from model.layers import Decoder
# from model.backbone import SwinBasicLayer
from avss.model.BCSM import BCSM
# from TPAVI import TPAVIModule
# from model.dual_decoder import Decoder_




class SelM_PVT(nn.Module):
    def __init__(self, 
                 config=None,
                 training=True):
        super(SelM_PVT,self).__init__()
        self.cfg=config
        
        
        self.encoder_backbone = pvt_v2_b5()
        self.tem_att=BCSM()
        
    
        
        
        self.in_proj4 = nn.Sequential(nn.Conv2d(512,256,kernel_size=1),
                                     nn.GroupNorm(32,256))
        self.in_proj3 = nn.Sequential(nn.Conv2d(320,256,kernel_size=1),
                                     nn.GroupNorm(32,256))
        self.in_proj2 = nn.Sequential(nn.Conv2d(128,256,kernel_size=1),
                                     nn.GroupNorm(32,256))
        self.in_proj1 = nn.Sequential(nn.Conv2d(64,256,kernel_size=1),
                                     nn.GroupNorm(32,256))
        
        
        
        
        
        
        
        self.decoder=Decoder(token_dim=256,num_token=2)
        self.audio_proj=nn.Linear(128,256)
        self.audio_norm=nn.LayerNorm(256)
        
        self.training=training
        if self.training:
            self.initialize_pvt_weights()
        
    def freeze(self,flag):
        if flag==True:
            for p in self.encoder_backbone.parameters():
                p.requires_grad=False
        if flag==False:
            for p in self.encoder_backbone.parameters():
                p.requires_grad=True
        
        
    def forward(self,x_in,audio_embed,vid_temporal_mask_flag):
        
        audio_temporal_mask_flag = vid_temporal_mask_flag.unsqueeze(-1)
        
        BT,C,h,w=x_in.shape
        vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1).contiguous()
        input_shape=x_in.shape[-2:]
        audio_embed=self.audio_proj(audio_embed)
        audio_embed=self.audio_norm(audio_embed)
        audio_embed=audio_embed * audio_temporal_mask_flag
        audio_embed=audio_embed.unsqueeze(2)
        audio_mask=audio_temporal_mask_flag.unsqueeze(-1)
        # audio_embed=audio_embed.unsqueeze(1).permute(0,2,1)
        # audio_mask=torch.ones([BT,1,1]).cuda()
        
        x1, x2, x3, x4 = self.encoder_backbone(x_in*vid_temporal_mask_flag,audio_embed,audio_mask)
        
        x1,_ = x1
        x2,_ = x2 
        x3,_ = x3
        x4, audio_embed = x4
        
        # x1=self.conv1(x1) #[B*T 256,56,56]
        # x2=self.conv2(x2)
        # x3=self.conv3(x3)
        # x4=self.conv4(x4)
        
        x1 = x1 * vid_temporal_mask_flag
        x2 = x2 * vid_temporal_mask_flag
        x3 = x3 * vid_temporal_mask_flag
        x4 = x4 *vid_temporal_mask_flag
        
        
        x1 = self.in_proj1(x1)
        x2 = self.in_proj2(x2)
        x3 = self.in_proj3(x3)
        x4 = self.in_proj4(x4)
        
        
        audio_embed = audio_embed.view(1,BT,256)
        feature_map_list,audio_embed = self.tem_att([x1,x2,x3,x4],audio_embed)
        
        a_fea_list=[None]*4
        

        
        fuse_mask,maps=self.decoder(feature_map_list,audio_embed,audio_mask)
        fuse_mask=F.interpolate(fuse_mask,input_shape,mode='bilinear',align_corners=True)
        
        
        fuse_mask = fuse_mask*vid_temporal_mask_flag
        
        return fuse_mask,feature_map_list,a_fea_list,maps
    
        
               
    
    def initialize_pvt_weights(self,):
        pvt_model_dict = self.encoder_backbone.state_dict()
        pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_PVTV2_PATH)
        # for k, v in pretrained_state_dicts['model'].items():
        #     if k in pvt_model_dict.keys():
        #         print(k, v.requires_grad)
        state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(state_dict)
        self.encoder_backbone.load_state_dict(pvt_model_dict)
        print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
    
    
if __name__=="__main__":
    model=SelM_PVT(pretrained='avs_scripts\\avs_s4\AudioCLIP\\assets\AudioCLIP-Full-Training.pt').cuda()
    img=torch.randn(10,3,224,224).cuda()
    audio=torch.randn(10,1,45000).cuda()
    output=model(img,audio)
    print(output.shape)
        
    
        
        
        
        
        
            
        