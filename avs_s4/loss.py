import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def F1_Dice_loss(pred_masks, first_gt_mask):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs*5, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, 1, h, w)
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

    pred_mask = first_pred.flatten(1)
    gt_mask = first_gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()





def sigmoid_focal_loss(pred_masks, first_gt_mask, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    assert len(pred_masks.shape) == 4
    # pred_masks = torch.sigmoid(pred_masks)

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]
    first_pred_sig=first_pred.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(first_pred, first_gt_mask, reduction="none")
    p_t = first_pred_sig * first_gt_mask + (1 - first_pred_sig) * (1 - first_gt_mask)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * first_gt_mask + (1 - alpha) * (1 - first_gt_mask)
        loss = alpha_t * loss
    return loss.mean()


def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks) # [bs*5, 1, 224, 224]
    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()

    first_pred = torch.index_select(pred_masks, dim=0, index=indices) # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1) # [bs, 1, 224, 224]
    first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

    return first_bce_loss






def Seg_Loss(pred_masks, first_gt_mask,  \
        loss_type='dice',hitmaps=None):
    """
    loss for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    a_fea_list: feature list of audio features
    v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W]
    count_stages: additional constraint loss on which stages' visual-audio features
    """
    if loss_type=='dice':
        loss_func=F1_Dice_loss
    else:
        loss_func=F1_IoU_BCELoss
        
    
    total_loss = 0
    f1_iou_loss = loss_func(pred_masks, first_gt_mask)
    total_loss += f1_iou_loss


    bce_loss=sigmoid_focal_loss(pred_masks,first_gt_mask,alpha=-1,gamma=0)

    total_loss+=bce_loss
    
    muti_scale_hitmap_loss=0.0
    
    if hitmaps is not None:
        for maps,lambda_ in zip(hitmaps,[0.001,0.01,0.1]):
            maps = maps[:,1].unsqueeze(1)
            # print(maps.shape)
            # print(first_gt_mask.shape)
            # breakpoint()
            if maps.shape[-2:] != first_gt_mask.shape[-2:]:
                mask_ = F.interpolate(maps, first_gt_mask.shape[-2:], mode='nearest')
            muti_scale_hitmap_loss+=loss_func(mask_,first_gt_mask)*lambda_
    total_loss+=muti_scale_hitmap_loss
    
    
    loss_dict = {}
    loss_dict['iou_loss'] = f1_iou_loss.item()
    loss_dict['bce_loss'] = bce_loss.item()
    loss_dict['hitmap_loss']=muti_scale_hitmap_loss.item()
    return total_loss, loss_dict


if __name__ == "__main__":

    pdb.set_trace()
