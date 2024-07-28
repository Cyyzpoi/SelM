setting='AVSS'
visual_backbone="resnet" # "resnet" or "pvt"
python -m torch.distributed.launch --nproc_per_node 8 train.py --session_name ${setting}_${visual_backbone} --visual_backbone ${visual_backbone} --max_epoches 30 --train_batch_size 8 --val_batch_size 2 --lr 2e-5 --wt_dec 0.01 --start_eval_epoch 0 --eval_interval 1 