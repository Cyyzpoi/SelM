setting='S4'
visual_backbone="resnet" # "resnet" or "pvt"

python train.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --max_epoches 40 \
        --train_batch_size 2 \
        --val_batch_size 2 \
        --wt_dec 0.05 \
        --lr 2e-5 \
 

