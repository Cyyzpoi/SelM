import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model.resnet import B2_ResNet
from model.pvt import pvt_v2_b5
from model.decoder import Decoder
from model.BCSM import BCSM
from model.DAM import DAM_Fusion_Block


class SelM_R50(nn.Module):
    def __init__(self, 
                 config=None,
                 training=True):
        super(SelM_R50,self).__init__()
        self.cfg=config
        
        self.resnet=B2_ResNet()
        # for p in self.resnet.parameters():
        #     p.requires_grad=False
        
        self.freeze(False)
        self.BCSM=BCSM()
        
        self.in_proj4 = nn.Sequential(nn.Conv2d(2048,256,kernel_size=1),
                                     nn.GroupNorm(32,256))
        self.in_proj3 = nn.Sequential(nn.Conv2d(1024,256,kernel_size=1),
                                     nn.GroupNorm(32,256))
        self.in_proj2 = nn.Sequential(nn.Conv2d(512,256,kernel_size=1),
                                     nn.GroupNorm(32,256))
        self.in_proj1 = nn.Sequential(nn.Conv2d(256,256,kernel_size=1),
                                     nn.GroupNorm(32,256))
        
    
        
        self.DAM_Fusion1=DAM_Fusion_Block(dim=256)
        self.DAM_Fusion2=DAM_Fusion_Block(dim=512)
        self.DAM_Fusion3=DAM_Fusion_Block(dim=1024)
        self.DAM_Fusion4=DAM_Fusion_Block(dim=2048)
        
        
        self.decoder=Decoder(num_token=2,token_dim=256)
        self.audio_proj=nn.Linear(128,256)
        self.audio_norm=nn.LayerNorm(256)
        self.training=training
        if self.training:
            self.initialize_weights()
            
    
    def freeze(self,flag):
        if flag==True:
            for p in self.resnet.parameters():
                p.requires_grad=False
        if flag==False:
            for p in self.resnet.parameters():
                p.requires_grad=True
        
    
    def forward(self,x_in,audio_embed,vid_temporal_mask_flag):

        
        audio_temporal_mask_flag = vid_temporal_mask_flag.unsqueeze(-1)
        
        # T=torch.sum(vid_temporal_mask_flag)
        
        vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1).contiguous()
        
        
        
        x = self.resnet.conv1(x_in*vid_temporal_mask_flag)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        
        BT,C,h,w=x.shape
        
        audio_feature=self.audio_proj(audio_embed)
        audio_feature=self.audio_norm(audio_feature)
        input_shape=x_in.shape[-2:]
        audio_feature=audio_feature * audio_temporal_mask_flag
        audio_feature=audio_feature.unsqueeze(2)
        audio_mask=audio_temporal_mask_flag.unsqueeze(-1)
        
        x1 = self.resnet.layer1(x)     # BF x 256  x 56 x 56
        x1,audio_feature=self.DAM_Fusion1(x1,audio_feature,audio_mask)
        x2 = self.resnet.layer2(x1)    # BF x 512  x 28 x 28
        x2,audio_feature=self.DAM_Fusion2(x2,audio_feature,audio_mask)
        x3 = self.resnet.layer3_1(x2)  # BF x 1024 x 14 x 14
        x3,audio_feature=self.DAM_Fusion3(x3,audio_feature,audio_mask)
        x4 = self.resnet.layer4_1(x3)  # BF x 2048 x  7 x  7
        x4,audio_feature=self.DAM_Fusion4(x4,audio_feature,audio_mask)
        
        
        x1 = self.in_proj1(x1)
        x2 = self.in_proj2(x2)
        x3 = self.in_proj3(x3)
        x4 = self.in_proj4(x4)
        
        
        
        x1 = x1 * vid_temporal_mask_flag
        x2 = x2 * vid_temporal_mask_flag
        x3 = x3 * vid_temporal_mask_flag
        x4 = x4 *vid_temporal_mask_flag
        
        
        audio_feature=audio_feature.squeeze(2).view(1,BT,256).contiguous()
        
        feature_map_list, audio_feature=self.BCSM([x1,x2,x3,x4],audio_feature)
            
        fuse_mask,maps=self.decoder(feature_map_list,audio_feature,audio_mask)

        
        fuse_mask=F.interpolate(fuse_mask,input_shape,mode='bilinear',align_corners=True)
        
        
        fuse_mask = fuse_mask *vid_temporal_mask_flag
       
        
        return fuse_mask,feature_map_list,maps
    
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=False)
        resnet50_dict = torch.load(self.cfg.TRAIN.PRETRAINED_RESNET50_PATH)
        res50.load_state_dict(resnet50_dict)
        pretrained_dict = res50.state_dict()
        # print(pretrained_dict.keys())
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
        print(f'==> Load pretrained ResNet50 parameters from {self.cfg.TRAIN.PRETRAINED_RESNET50_PATH}')
 
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

        
        x1, x2, x3, x4 = self.encoder_backbone(x_in*vid_temporal_mask_flag,audio_embed,audio_mask)
        
        x1,_ = x1
        x2,_ = x2 
        x3,_ = x3
        x4, audio_embed = x4
        
        
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
        
        
        fuse_mask,maps=self.decoder(feature_map_list,audio_embed,audio_mask)
        fuse_mask=F.interpolate(fuse_mask,input_shape,mode='bilinear',align_corners=True)
        
        
        fuse_mask = fuse_mask*vid_temporal_mask_flag
        
        return fuse_mask,feature_map_list,maps
    
    def initialize_pvt_weights(self,):
        pvt_model_dict = self.encoder_backbone.state_dict()
        pretrained_state_dicts = torch.load(
            self.cfg.TRAIN.PRETRAINED_PVTV2_PATH)
        # for k, v in pretrained_state_dicts['model'].items():
        #     if k in pvt_model_dict.keys():
        #         print(k, v.requires_grad)
        state_dict = {k: v for k, v in pretrained_state_dicts.items()
                      if k in pvt_model_dict.keys()}
        pvt_model_dict.update(state_dict)
        self.encoder_backbone.load_state_dict(pvt_model_dict)
        print(
            f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
 
    
    
if __name__=="__main__":
    model=SelM_R50().cuda()
        
    
        
        
        
        
        
            
        