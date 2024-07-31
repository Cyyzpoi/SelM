setting='S4'
visual_backbone="resnet" # "resnet" or "pvt"

python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights "../pretrained model/S4_R50.pth" \
    --test_batch_size 2 \
    --save_pred_mask  \