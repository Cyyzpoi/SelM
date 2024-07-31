setting='MS3'
visual_backbone="pvt" # "resnet" or "pvt"

python test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --weights "../pretrained model/MS3_PVT.pth" \
    --test_batch_size 2 \
    --save_pred_mask  \