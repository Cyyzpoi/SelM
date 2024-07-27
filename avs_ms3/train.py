import sys
sys.path.append(
    '../avs_ms3')
from utils.utility import logger, mask_iou
from model.SelM import SelM_R50,SelM_PVT
import pdb
from utils.system import setup_logging
from utils import pyutils
from torchvggish import vggish
from loss import Seg_Loss
from dataloader import MS3Dataset
from config import cfg
import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

import warnings

# from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="MS3",
                        type=str, help="the MS3 setting")
    parser.add_argument("--visual_backbone", default="resnet",
                        type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--val_batch_size", default=2, type=int)
    parser.add_argument("--max_epoches", default=100, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)


    parser.add_argument("--load_s4_params", action='store_true',
                        default=False, help='use S4 parameters for initilization')
    parser.add_argument("--trained_s4_model_path", type=str,
                        default='', help='pretrained S4 model')


    parser.add_argument("--weights", type=str, default='',
                        help='path of trained model')
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
    log_dir = os.path.join(args.log_dir, '{}'.format(
        time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = [ 'train.py', 'test.py', 'config.py',
                       'dataloader.py', './model/SelM.py', './model/decoder.py','./model/BCSM.py','loss.py']
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
    # model = torch.nn.DataParallel(model).cuda()
    model.train()
    
    model_save_path = os.path.join(
                    checkpoint_dir, '%s_best.pth' % (args.session_name))
    torch.save(model.state_dict(), model_save_path)
    breakpoint()
    

    audio_backbone = vggish.VGGish(cfg=cfg, device='cuda')
    audio_backbone.eval()

    logger.info("==> Total params: %.2fM" % (sum(p.numel()
                for p in model.parameters()) / 1e6))

    # load pretrained S4 model
    if args.load_s4_params:  # fine-tune single sound source segmentation model
        model_dict = model.state_dict()
        s4_state_dicts = torch.load(args.trained_s4_model_path)
        state_dict = {'module.' + k: v for k,
                      v in s4_state_dicts.items() if 'module.' + k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        logger.info("==> Reload pretrained S4 model from %s" %
                    (args.trained_s4_model_path))

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data
    train_dataset = MS3Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = MS3Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.val_batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.AdamW(model_params, args.lr, weight_decay=0.05)
    # lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[20],gamma=0.5,verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                                  lambda x: (1 - x / (len(train_dataloader) * args.max_epoches)) ** 0.9,verbose=False)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_bce_loss = pyutils.AverageMeter('bce_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_hitmaploss = pyutils.AverageMeter('hitmap_loss')
    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(args.max_epoches):
        if epoch == 15:
            model.freeze(False)
            print('the model has been unfreezed')
        for n_iter, batch_data in enumerate(train_dataloader):
            # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5 or 1, 1, 224, 224]
            imgs, audio, mask, _ = batch_data

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            # writer.add_images('origin_images',img_tensor=imgs,global_step=global_step)
            mask_num = 5
            mask = mask.view(B*mask_num, 1, 224, 224)
            # [B*T, 1, 96, 64]
            audio = audio.view(-1, audio.shape[2],
                               audio.shape[3], audio.shape[4])

            with torch.no_grad():
                audio = audio_backbone(audio)

            output, v_map_list, maps = model(
                imgs, audio)  # [bs*5, 1, 224, 224]
            # writer.add_images('mask',img_tensor=output,global_step=global_step)
            loss, loss_dict = Seg_Loss(output, mask, hitmaps=maps, mask_feature=None)

            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_bce_loss.add({'bce_loss': loss_dict['bce_loss']})
            avg_meter_hitmaploss.add({'hitmap_loss': loss_dict['hitmap_loss']})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            global_step += 1
            if (global_step-1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, bce_loss:%.4f, hitmap_loss:%.4f,lr: %.7f' % (
                            global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_bce_loss.pop('bce_loss'), avg_meter_hitmaploss.pop('hitmap_loss'), optimizer.param_groups[0]['lr'])
                logger.info(train_log)
        # lr_scheduler.step()
        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                imgs, audio, mask, _ = batch_data

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B*frame, C, H, W)
                mask = mask.view(B*frame, 224, 224)
                audio = audio.view(-1, audio.shape[2],
                                   audio.shape[3], audio.shape[4])

                audio = audio_backbone(audio)

                output, _, _ = model(imgs, audio)  # [bs*5, 1, 224, 224]

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})
            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(
                    checkpoint_dir, '%s_best.pth' % (args.session_name))
                torch.save(model.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s' % model_save_path)
            miou_list.append(miou)
            max_miou = max(miou_list)
            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(
                epoch, miou, max_miou)
            # print(val_log)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
