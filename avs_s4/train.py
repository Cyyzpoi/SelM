import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
import sys
sys.path.append('../avs_s4')
from config import cfg
from dataloader import S4Dataset
from loss import Seg_Loss
from torchvggish import vggish
from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import pdb
from model.SelM import SelM_PVT,SelM_R50
# from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--val_batch_size", default=2, type=int)
    parser.add_argument("--max_epoches", default=40, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=0.05, type=float)

    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)

    args = parser.parse_args()

    # writer=SummaryWriter('logs')
    
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = [ 'train.py', 'test.py', 'config.py', 'dataloader.py', './model/SelM.py', './model/BCSM.py', './model/decoder.py','./model/DAM.py','loss.py']
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
    
    
    if (args.visual_backbone).lower() == "resnet":
        model = SelM_R50(config=cfg)
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        model = SelM_PVT(config=cfg)
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")
    
    
    
    # model = SelM_R50(config=cfg)
    # model.load_state_dict(torch.load(args.weights))
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False,find_unused_parameters=True).cuda()
    model.train()
    
    audio_backbone=vggish.VGGish(cfg=cfg,device='cuda')
    audio_backbone.eval()
    
    # for k, v in model.named_parameters():
    #         print(k, v.requires_grad)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dataset = S4Dataset('train',backbone=args.visual_backbone)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = S4Dataset('val',backbone=args.visual_backbone)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    # Optimizer
    model_params = model.parameters()
    
    optimizer = torch.optim.AdamW(model_params, args.lr,weight_decay=args.wt_dec)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[15,30,45],gamma=0.5,verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                                  lambda x: (1 - x / (len(train_dataloader) * args.max_epoches)) ** 0.9,verbose=False)
    
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_bce_loss = pyutils.AverageMeter('bce_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_hitmaploss=pyutils.AverageMeter('hitmap_loss')
    avg_meter_Fs = pyutils.AverageMeter('Fscore')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(args.max_epoches):
        if epoch==0:
            model.freeze(False)
            print('the model has been unfreezed')
        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 45000], [bs, 1, 1, 224, 224]
            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            # writer.add_images('origin_images',img_tensor=imgs,global_step=global_step)
            mask = mask.view(B, 224, 224)
            audio = audio.view(-1, audio.shape[2], audio.shape[3],audio.shape[4]) # [B*T, 1, 96, 64]

            with torch.no_grad():
                audio=audio_backbone(audio)
            
            output,v_feature,hitmaps= model(imgs, audio) # [bs*5, 1, 224, 224]
            # writer.add_images('mask',img_tensor=output,global_step=global_step)
            loss, loss_dict = Seg_Loss(output, mask.unsqueeze(1).unsqueeze(1),hitmaps=hitmaps)

            
            
            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_bce_loss.add({'bce_loss': loss_dict['bce_loss']})
            avg_meter_hitmaploss.add({'hitmap_loss':loss_dict['hitmap_loss']})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            # writer.add_scalar("iou_loss",loss.item(),global_step)
            # writer.add_scalar("learning_rate",optimizer.param_groups[0]['lr'],global_step)
            # writer.add_scalar("l1_loss",l1loss.item(),global_step)
            global_step += 1

            if (global_step-1) % 50 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, bce_loss:%.4f, hitmap_loss:%.4f,lr: %.6f'%(
                            global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_bce_loss.pop('bce_loss'), avg_meter_hitmaploss.pop('hitmap_loss'),optimizer.param_groups[0]['lr'])
                # train_log = ['Iter:%5d/%5d' % (global_step - 1, max_step),
                #         'Total_Loss:%.4f' % (avg_meter_loss.pop('total_loss')),
                #         'iou_loss:%.4f' % (avg_meter_iou_loss.pop('iou_loss')),
                #         'sa_loss:%.4f' % (avg_meter_sa_loss.pop('sa_loss')),
                #         'lambda_1:%.4f' % (args.lambda_1),
                #         'lr: %.4f' % (optimizer.param_groups[0]['lr'])]
                # print(train_log, flush=True)
                logger.info(train_log)
        # lr_scheduler.step()

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                imgs, audio, mask, _, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B*frame, C, H, W)
                mask = mask.view(B*frame, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3],audio.shape[4])

                audio=audio_backbone(audio)
                
                output,_,_= model(imgs, audio) # [bs*5, 1, 224, 224]
    

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            # writer.add_scalar("miou",miou,epoch+1)
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
                torch.save(model.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s'%model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            # print(val_log)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))











