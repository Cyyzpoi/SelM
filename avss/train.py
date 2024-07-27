import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
import torch.cuda.amp as amp
import utils.misc as utils
from model.SelM import SelM_R50,SelM_PVT
from config import cfg
from color_dataloader import V2Dataset
from torchvggish import vggish
from loss import Seg_Loss

from utils import pyutils
from utils.utility import logger
from utils.compute_color_metrics import calc_color_miou_fscore
from utils.system import setup_logging



class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="AVSS", type=str, help="the AVSS setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--val_batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=30, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument("--start_eval_epoch", default=0, type=int)
    parser.add_argument("--eval_interval", default=1, type=int)


    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    
    parser.add_argument('--local-rank',type=int,default=-1,help='local rank for DDP')

    parser.add_argument('--device',default='cuda',help='device for DDP')
    
    
    args = parser.parse_args()


    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)
    # Fix seed
    FixSeed = 123
    seed = FixSeed + utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    if os.path.exists(log_dir):
        log_dir = os.path.join(args.log_dir, '{}_{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S'), np.random.randint(1, 10)))
   
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = [ 'train.py', 'test.py', 'config.py', 'color_dataloader.py', './model/SelM.py', './model/BCSM.py', 'loss.py','./model/DAM.py','./model/decoder.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    
    # model=SelM_R50(config=cfg)
    if (args.visual_backbone).lower() == "resnet":
        model = SelM_R50(config=cfg)
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        model = SelM_PVT(config=cfg)
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2") 
    
    # model = torch.nn.DataParallel(model).cuda()
    model.to(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.train()
    logger.info("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
    
    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()


    # Data
    train_dataset = V2Dataset('train') 
    # train_dataset = V2Dataset('train', debug_flag=True) 
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size//8,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        sampler=train_sampler)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = V2Dataset('val')
    val_sampler=torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size//8,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        sampler=val_sampler)

    N_CLASSES = train_dataset.num_classes

    scaler=amp.GradScaler()
    
    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.AdamW(model_params, args.lr)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_hitmap_loss = pyutils.AverageMeter('hitmap_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    miou_noBg_list = []
    fscore_list, fscore_noBg_list = [], []
    max_fs, max_fs_noBg = 0, 0
    for epoch in range(args.max_epoches):
        train_sampler.set_epoch(epoch)
        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, label, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data # [bs, 5, 3, 224, 224], ->[bs, 5, 1, 96, 64], [bs, 10, 1, 224, 224]
            #! notice:
            vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
            gt_temporal_mask_flag  = gt_temporal_mask_flag.cuda()
            
            imgs = imgs.cuda()
            label = label.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask_num = 10
            label = label.view(B*mask_num, H, W)
            #! notice
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
            gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]

            with torch.no_grad():
                audio_feature = audio_backbone(audio) # [B*T, 128]
                #! notice:
                audio_feature = audio_feature * vid_temporal_mask_flag.unsqueeze(-1)
            # pdb.set_trace()
            with amp.autocast():
                output, v_map_list,hitmaps= model(imgs, audio_feature, vid_temporal_mask_flag) # [bs*5, 24, 224, 224]
                loss, loss_dict = Seg_Loss(output, label,gt_temporal_mask_flag,loss_type='bce',hitmaps=hitmaps)

            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_hitmap_loss.add({'hitmap_loss': loss_dict['hitmap_loss']})

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # scheduler.step()
            # loss.backward()
            # optimizer.step()

            global_step += 1
            if (global_step-1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, hitmap_loss:%.4f, lr: %.6f'%(
                            global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_hitmap_loss.pop('hitmap_loss'), optimizer.param_groups[0]['lr'])
                # train_log = ['Iter:%5d/%5d' % (global_step - 1, max_step),
                #         'Total_Loss:%.4f' % (avg_meter_total_loss.pop('total_loss')),
                #         'iou_loss:%.4f' % (avg_meter_L1.pop('iou_loss')),
                #         'hitmap_loss:%.4f' % (avg_meter_L4.pop('hitmap_loss')),
                #         'lambda_1:%.4f' % (args.lambda_1),
                #         'lr: %.4f' % (optimizer.param_groups[0]['lr'])]
                # print(train_log, flush=True)
                logger.info(train_log)

        # Validation:
        if epoch >= args.start_eval_epoch and epoch % args.eval_interval == 0:
        # if epoch >= args.start_eval_epoch:
            model.eval()
            
            miou_pc = torch.zeros((N_CLASSES)) # miou value per class (total sum)
            Fs_pc = torch.zeros((N_CLASSES)) # f-score per class (total sum)
            cls_pc = torch.zeros((N_CLASSES)) # count per class
            with torch.no_grad():
                # out_list=[]
                # mask_list=[]
                for n_iter, batch_data in enumerate(val_dataloader):
                    imgs, audio, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                    vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
                    gt_temporal_mask_flag  = gt_temporal_mask_flag.cuda()

                    imgs = imgs.cuda()
                    mask = mask.cuda()
                    B, frame, C, H, W = imgs.shape
                    imgs = imgs.view(B*frame, C, H, W)
                    mask = mask.view(B*frame, H, W)
                    #! notice
                    vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
                    gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]

                    audio_feature = audio_backbone(audio) # # [B*T, 128]
                    #! notice:
                    audio_feature = audio_feature * vid_temporal_mask_flag.unsqueeze(-1)

                    output, _, _= model(imgs, audio_feature, vid_temporal_mask_flag) # [bs*5, 21, 224, 224]
                    
                    _miou_pc, _fscore_pc, _cls_pc ,_= calc_color_miou_fscore(output, mask)
                    # compute miou, J-measure
                    miou_pc += _miou_pc
                    cls_pc += _cls_pc
                    # compute f-score, F-measure
                    Fs_pc += _fscore_pc

                    
                
                # pdb.set_trace()
                miou_pc = miou_pc / cls_pc
                print(f"[miou] {torch.sum(torch.isnan(miou_pc)).item()} classes are not predicted in this batch")
                miou_pc[torch.isnan(miou_pc)] = 0
                miou = torch.mean(miou_pc).item()
                miou_noBg = torch.mean(miou_pc[:-1]).item()
                f_score_pc = Fs_pc / cls_pc
                print(f"[fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes are not predicted in this batch")
                f_score_pc[torch.isnan(f_score_pc)] = 0
                f_score = torch.mean(f_score_pc).item()
                f_score_noBg = torch.mean(f_score_pc[:-1]).item()
                # pdb.set_trace()
                
                # if miou > max_miou:
                #     model_save_path = os.path.join(checkpoint_dir, '%s_miou_best.pth'%(args.session_name))
                #     torch.save(model.state_dict(), model_save_path)
                #     best_epoch = epoch
                #     logger.info('save miou best model to %s'%model_save_path)
                # if (miou + f_score) > (max_miou + max_fs):
                #     model_save_path = os.path.join(checkpoint_dir, '%s_miou_and_fscore_best.pth'%(args.session_name))
                #     torch.save(model.state_dict(), model_save_path)
                #     best_epoch = epoch
                #     logger.info('save miou and fscore best model to %s'%model_save_path)     

                miou_list.append(miou)
                miou_noBg_list.append(miou_noBg)
                max_miou = max(miou_list)
                max_miou_noBg = max(miou_noBg_list)
                fscore_list.append(f_score)
                fscore_noBg_list.append(f_score_noBg)
                max_fs = max(fscore_list)
                max_fs_noBg = max(fscore_noBg_list)

                val_log = 'Epoch: {}, Miou: {}, maxMiou: {}, Miou(no bg): {}, maxMiou (no bg): {} '.format(epoch, miou, max_miou, miou_noBg, max_miou_noBg)
                val_log += ' Fscore: {}, maxFs: {}, Fscore(no bg): {}, max Fscore (no bg): {}'.format(f_score, max_fs, f_score_noBg, max_fs_noBg)
                # print(val_log)
                logger.info(val_log)
                
                model_save_path = os.path.join(checkpoint_dir, 'epoch_%d.pth'%(epoch+1))
                utils.save_on_master(model.module.state_dict(),model_save_path)
                logger.info('save miou best model to %s'%model_save_path)

            model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
