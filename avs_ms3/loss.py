import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def F5_Dice_Loss(pred_masks,gt_mask):
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)

    pred_mask = pred_masks.flatten(1)
    gt_mask = gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()


def F5_sigmoid_Focal_loss(pred_masks,gt_mask,alpha=-1,gamma=0):
    assert len(pred_masks.shape) == 4
    pred_masks_sig=pred_masks.sigmoid()
    ce_loss=F.binary_cross_entropy_with_logits(pred_masks,gt_mask,reduction='none')
    p_t=pred_masks_sig*gt_mask+(1-pred_masks_sig)*(1-gt_mask)
    loss=ce_loss*((1-p_t)**gamma)
    
    if alpha>=0:
        alpha_t=alpha*gt_mask+(1-alpha)*(1-gt_mask)
        loss = alpha_t*loss
    return loss.mean()


def F5_IoU_BCELoss(pred_mask, five_gt_masks):
    """
    binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask) # [bs*5, 1, 224, 224]
    # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]
    loss = nn.BCELoss()(pred_mask, five_gt_masks)

    return loss





def Seg_Loss(pred_masks, gt_mask, \
                        loss_type='dice',hitmaps=None,mask_feature=None):
    """
    loss for multiple sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    gt_mask: ground truth mask of the first frame (one-shot) or five frames, shape: [bs, 1, 1, 224, 224]
    a_fea_list: feature list of audio features
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W]
    count_stages: additional constraint loss on which stages' visual-audio features
    """
    if loss_type=='dice':
        loss_func=F5_Dice_Loss
    else:
        loss_func=F5_IoU_BCELoss
    
    total_loss = 0
    iou_loss = loss_func(pred_masks, gt_mask)
    total_loss += iou_loss
    bce_loss = F5_sigmoid_Focal_loss(pred_masks,gt_mask)
    total_loss += bce_loss

    # total_loss+=masked_av_loss
    
    muti_scale_hitmap_loss=0.0
    
    if hitmaps is not None:
        for maps,lambda_ in zip(hitmaps,[0.001,0.01,0.1]):
            maps = maps[:,1].unsqueeze(1)
            # print(maps.shape)
            # print(first_gt_mask.shape)
            # breakpoint()
            if maps.shape[-2:] != gt_mask.shape[-2:]:
                mask_ = F.interpolate(maps, gt_mask.shape[-2:], mode='nearest')
            muti_scale_hitmap_loss+=loss_func(mask_,gt_mask)*lambda_
        total_loss+=muti_scale_hitmap_loss
    else:
        muti_scale_hitmap_loss=torch.zeros(1)
    
    if mask_feature is not None:
        mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
        mask_feature = F.interpolate(
        mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
        mix_loss=0.1*loss_func(mask_feature,gt_mask)
    else:
        mix_loss=0.0
        
    total_loss+=mix_loss
    
   
    
    loss_dict = {}
    loss_dict['iou_loss'] = iou_loss.item()
    loss_dict['bce_loss'] = bce_loss.item()
    loss_dict['hitmap_loss']=muti_scale_hitmap_loss.item()
    
    return total_loss, loss_dict








if __name__ == "__main__":

    pdb.set_trace()
