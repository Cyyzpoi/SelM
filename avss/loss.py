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



def Seg_Loss(pred_masks, gt_mask,
                        gt_temporal_mask_flag,loss_type='dice',hitmaps=None):
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
    loss_dict['hitmap_loss']=muti_scale_hitmap_loss

    return total_loss, loss_dict








if __name__ == "__main__":

    pdb.set_trace()
