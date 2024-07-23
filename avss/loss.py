import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def F10_sigmoid_Focal_loss(pred_masks,gt_mask,gt_temporal_flag,alpha=-1,gamma=0):
    assert len(pred_masks.shape) == 4
    pred_masks_sig=pred_masks.sigmoid()
    ce_loss=F.binary_cross_entropy_with_logits(pred_masks,gt_mask,reduction='none')
    p_t=pred_masks_sig*gt_mask+(1-pred_masks_sig)*(1-gt_mask)
    loss=ce_loss*((1-p_t)**gamma)
    
    if alpha>=0:
        alpha_t=alpha*gt_mask+(1-alpha)*(1-gt_mask)
        loss = alpha_t*loss
        
    loss=loss*gt_temporal_flag
    
    loss = torch.sum(loss)/torch.sum(gt_temporal_flag)
    
    return loss



def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total five frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1) # [bs*10, 224, 224]
    # loss = nn.CrossEntropyLoss()(pred_mask, ten_gt_masks)
    #! notice:
    loss = nn.CrossEntropyLoss(reduction='none')(pred_mask, ten_gt_masks) # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1) # [bs*10]
    loss = loss * gt_temporal_mask_flag # [bs*10]
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss


def F10_Dice_Loss(pred_masks,gt_masks,gt_temporal_mask_flag):
    assert len(pred_masks.shape)==4
    if gt_masks.shape[1] == 1:
        gt_masks = gt_masks.squeeze(1)
        
    
    pred_mask = torch.sigmoid(pred_masks)

    pred_mask = pred_mask.flatten(1)
    gt_mask = gt_masks.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    loss = loss * gt_temporal_mask_flag
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
    return loss
    



def A_MaskedV_SimmLoss(pred_masks, a_fea_list, v_map_list, \
                        gt_temporal_mask_flag, \
                        count_stages=[], \
                        mask_pooling_type='avg', norm_fea=True, threshold=False,\
                        euclidean_flag=False, kl_flag=False):
    """
    [audio] - [masked visual feature map] matching loss, Loss_AVM_AV reported in the paper

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    a_fea_list: audio feature list, lenth = nl_stages, each of shape: [bs, T, C], C is equal to [256]
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*10, C, H, W], C is equal to [256]
    count_stages: loss is computed in these stages
    """
    assert len(pred_masks.shape) == 4
    bg_idx = (pred_masks.shape[1] - 1)
    pred_masks = torch.softmax(pred_masks, dim=1) # [B*10, NUM_CLASSES, 224, 224]
    pred_masks = torch.argmax(pred_masks, dim=1).unsqueeze(1) # [B*10, 1, 224, 224]
    pred_masks = (pred_masks != bg_idx).float() # [B*10, 1, 224, 224]
    total_loss = 0

    for stage in count_stages:
        a_fea, v_map = a_fea_list[stage], v_map_list[stage] # v_map: [BT, C, H, W]
        a_fea = a_fea.view(-1, a_fea.shape[-1]) # [B*10, C]

        C, H, W = v_map.shape[1], v_map.shape[-2], v_map.shape[-1]
        assert C == a_fea.shape[-1], 'Error: dimensions of audio and visual features are not equal'

        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred_masks) # [bs*10, 1, H, W]
        elif mask_pooling_type == 'max':
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(pred_masks) # [bs*10, 1, H, W]
        # downsample_pred_masks = torch.sigmoid(downsample_pred_masks) # [B*5, 1, H, W]

        if threshold:
            downsample_pred_masks = (downsample_pred_masks > 0.5).float() # [bs*10, 1, H, W]
            obj_pixel_num = downsample_pred_masks.sum(-1).sum(-1) # [bs*10, 1]
            masked_v_map = torch.mul(v_map, downsample_pred_masks)  # [bs*10, C, H, W]
            masked_v_fea = masked_v_map.sum(-1).sum(-1) / (obj_pixel_num + 1e-6)# [bs*10, C]
        else:
            masked_v_map = torch.mul(v_map, downsample_pred_masks)
            masked_v_fea = masked_v_map.mean(-1).mean(-1) # [bs*10, C]

        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        if euclidean_flag:
            euclidean_distance = F.pairwise_distance(a_fea, masked_v_fea, p=2) # [bs*10]
            # loss = euclidean_distance.mean()
            #! notice:
            loss = euclidean_distance * gt_temporal_mask_flag # [bs*10]
            loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        elif kl_flag:
            # loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), a_fea.softmax(dim=-1), reduction='sum')
            #! notice:
            loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), a_fea.softmax(dim=-1), reduction='none') #[bs*10, C]
            loss = loss.sum(-1) # [bs*10]
            loss = loss * gt_temporal_mask_flag
            loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        
        total_loss += loss

    total_loss /= len(count_stages)

    return total_loss


def closer_loss(pred_masks, a_fea_list, v_map_list, \
                        gt_temporal_mask_flag, \
                        count_stages=[], \
                        mask_pooling_type='avg', norm_fea=True, \
                        euclidean_flag=False, kl_flag=False):
    """
    [audio] - [masked visual feature map] matching loss, Loss_AVM_VV reported in the paper

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    a_fea_list: audio feature list, lenth = nl_stages, each of shape: [bs, T, C], C is equal to [256]
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*10, C, H, W], C is equal to [256]
    count_stages: loss is computed in these stages
    """
    assert len(pred_masks.shape) == 4
    bg_idx = (pred_masks.shape[1] - 1)
    pred_masks = torch.softmax(pred_masks, dim=1) # [B*5, NUM_CLASSES, 224, 224]
    pred_masks = torch.argmax(pred_masks, dim=1).unsqueeze(1) # [B*5, 1, 224, 224]
    pred_masks = (pred_masks != bg_idx).float() # [B*5, 1, 224, 224]
    total_loss = 0
    for stage in count_stages:
        a_fea, v_map = a_fea_list[stage], v_map_list[stage] # v_map: [BT, C, H, W]
        a_fea = a_fea.view(-1, a_fea.shape[-1]) # [B*5, C]

        C, H, W = v_map.shape[1], v_map.shape[-2], v_map.shape[-1]
        assert C == a_fea.shape[-1], 'Error: dimensions of audio and visual features are not equal'

        if mask_pooling_type == "avg":
            downsample_pred_masks = nn.AdaptiveAvgPool2d((H, W))(pred_masks) # [bs*10, 1, H, W]
        elif mask_pooling_type == 'max':
            downsample_pred_masks = nn.AdaptiveMaxPool2d((H, W))(pred_masks) # [bs*10, 1, H, W]
        # downsample_pred_masks = torch.sigmoid(downsample_pred_masks) # [B*5, 1, H, W]

        ###############################################################################
        # pick the closest pair
        if norm_fea:
            a_fea = F.normalize(a_fea, dim=-1)

        a_fea_simi = torch.cdist(a_fea,a_fea,p=2) # [BT, BT]
        a_fea_simi = a_fea_simi + 10*torch.eye(a_fea_simi.shape[0]).cuda() #
        idxs = a_fea_simi.argmin(dim=0) # [BT]

        masked_v_map = torch.mul(v_map, downsample_pred_masks)
        masked_v_fea = masked_v_map.mean(-1).mean(-1) # [bs*10, C]
        if norm_fea:
            masked_v_fea = F.normalize(masked_v_fea, dim=-1)

        target_fea = masked_v_fea[idxs]
        ###############################################################################
        if euclidean_flag:
            euclidean_distance = F.pairwise_distance(target_fea, masked_v_fea, p=2)
            # loss = euclidean_distance.mean()
            #! notice:
            loss = euclidean_distance * gt_temporal_mask_flag # [bs*10]
            loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        elif kl_flag:
            # loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), target_fea.softmax(dim=-1), reduction='sum')
            #! notice:
            loss = F.kl_div(masked_v_fea.softmax(dim=-1).log(), a_fea.softmax(dim=-1), reduction='none') #[bs*10, C]
            loss = loss.sum(-1) # [bs*10]
            loss = loss * gt_temporal_mask_flag
            loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
        
        total_loss += loss

    total_loss /= len(count_stages)

    return total_loss


def Mix_Dice_loss(pred_mask, norm_gt_mask, gt_temporal_mask_flag):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = norm_gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    loss = loss * gt_temporal_mask_flag
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
    return loss



def IouSemanticAwareLoss(pred_masks, gt_mask, \
                        a_fea_list, v_map_list, \
                        gt_temporal_mask_flag, \
                        sa_loss_flag=False, count_stages=[], lambda_1=0, \
                        mask_pooling_type='avg', norm_fea=True, \
                        threshold=False, closer_flag=False, euclidean_flag=False, kl_flag=False,loss_type='dice',hitmaps=None):
    """
    loss for multiple sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    gt_mask: ground truth mask of the first frame (one-shot) or five frames, shape: [bs*10, 224, 224]
    a_fea_list: feature list of audio features
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*10, C, H, W]
    count_stages: additional constraint loss on which stages' visual-audio features
    """
    
    if loss_type=='dice':
        loss_func=F10_Dice_Loss
    else:
        loss_func=F10_IoU_BCELoss

    
    total_loss = 0
    iou_loss = loss_func(pred_masks, gt_mask, gt_temporal_mask_flag)
    total_loss += iou_loss

    # Focal_loss = F10_sigmoid_Focal_loss(pred_masks,gt_mask,gt_temporal_flag=gt_temporal_mask_flag)
    Focal_loss=torch.zeros(1)
    
    # total_loss+=Focal_loss
    
    if sa_loss_flag:
        
        total_loss += lambda_1 * masked_av_loss
    else:
        masked_av_loss = torch.zeros(1)

    muti_scale_hitmap_loss=0.0
    
    if hitmaps is not None:
        for maps,lambda_ in zip(hitmaps,[0.001,0.01,0.1]):
            maps = maps[:,1].unsqueeze(1)
            # print(maps.shape)
            # print(first_gt_mask.shape)
            # breakpoint()
            if maps.shape[-2:] != gt_mask.shape[-2:]:
                mask_ = F.interpolate(maps, gt_mask.shape[-2:], mode='nearest')
            one_mask = torch.ones_like(gt_mask)
            norm_gt_mask = torch.where(gt_mask > 0, one_mask, gt_mask)
            muti_scale_hitmap_loss+=Mix_Dice_loss(mask_,norm_gt_mask,gt_temporal_mask_flag)*lambda_
    else:
        muti_scale_hitmap_loss=0.0
    
    total_loss+=muti_scale_hitmap_loss
    
    loss_dict = {}
    loss_dict['iou_loss'] = iou_loss.item()
    loss_dict['sa_loss'] = muti_scale_hitmap_loss
    loss_dict['hitmap_loss']=muti_scale_hitmap_loss
    loss_dict['lambda_1'] = lambda_1

    return total_loss, loss_dict








if __name__ == "__main__":

    pdb.set_trace()
