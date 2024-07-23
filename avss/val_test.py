import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
import torch.cuda.amp as amp
import model.misc as utils
from avss.model.SelM import SelM_R50

from config import cfg
from color_dataloader import V2Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger
from utils.compute_color_metrics import calc_color_miou_fscore
from utils.system import setup_logging
import pdb


# import os





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

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=30, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument("--start_eval_epoch", default=0, type=int)
    parser.add_argument("--eval_interval", default=2, type=int)

    parser.add_argument('--masked_av_flag', action='store_true', default=False, help='additional sa/masked_va loss for five frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing loss')
    parser.add_argument("--masked_av_stages", default=[], nargs='+', type=int, help='compute sa/masked_va loss in which stages: [0, 1, 2, 3]')
    parser.add_argument('--threshold_flag', action='store_true', default=False, help='whether thresholding the generated masks')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')
    parser.add_argument('--norm_fea_flag', action='store_true', default=False, help='normalize audio-visual features')
    parser.add_argument('--closer_flag', action='store_true', default=False, help='use closer loss for masked_va loss')
    parser.add_argument('--euclidean_flag', action='store_true', default=False, help='use euclidean distance for masked_va loss')
    parser.add_argument('--kl_flag', action='store_true', default=False, help='use kl loss for masked_va loss')

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3]')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')

    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    
    # parser.add_argument('--rank',type=)

    args = parser.parse_args()

    # if (args.visual_backbone).lower() == "resnet":
    #     from model import ResNet_AVSModel as AVSModel
    #     print('==> Use ResNet50 as the visual backbone...')
    # elif (args.visual_backbone).lower() == "pvt":
    #     from model import PVT_AVSModel as AVSModel
    #     print('==> Use pvt-v2 as the visual backbone...')
    # else:
    #     raise NotImplementedError("only support the resnet50 and pvt-v2")

    utils.init_distributed_mode(args)
    
    
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
    if os.path.exists(log_dir):
        log_dir = os.path.join(args.log_dir, '{}_{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S'), np.random.randint(1, 10)))
   
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'color_dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
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
    # model = AVSModel.Pred_endecoder(channel=256, \
    #                                     config=cfg, \
    #                                     tpavi_stages=args.tpavi_stages, \
    #                                     tpavi_vv_flag=args.tpavi_vv_flag, \
    #                                     tpavi_va_flag=args.tpavi_va_flag)
    
    model=SelM_R50(config=cfg)
    
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    logger.info("==> Total params: %.2fM" % ( sum(p.numel() for p in model.parameters()) / 1e6))


    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()


    # Data
    train_dataset = V2Dataset('train') 
    # train_dataset = V2Dataset('train', debug_flag=True) 
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = V2Dataset('test')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    N_CLASSES = train_dataset.num_classes

    scaler=amp.GradScaler()
    
    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, args.lr)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
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
        # Validation:
        if epoch >= args.start_eval_epoch and epoch % args.eval_interval == 0:
        # if epoch >= args.start_eval_epoch:
            model.eval()
            
            miou_pc = torch.zeros((N_CLASSES)) # miou value per class (total sum)
            Fs_pc = torch.zeros((N_CLASSES)) # f-score per class (total sum)
            cls_pc = torch.zeros((N_CLASSES)) # count per class
            with torch.no_grad():
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

                    with torch.no_grad():
                        audio_feature = audio_backbone(audio) # # [B*T, 128]
                        #! notice:
                        audio_feature = audio_feature * vid_temporal_mask_flag.unsqueeze(-1)

                    output, _, _,_= model(imgs, audio_feature, vid_temporal_mask_flag) # [bs*5, 21, 224, 224]

                    _miou_pc, _fscore_pc, _cls_pc ,_= calc_color_miou_fscore(output, mask,)
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
                
                if miou > max_miou:
                    model_save_path = os.path.join(checkpoint_dir, '%s_miou_best.pth'%(args.session_name))
                    torch.save(model.state_dict(), model_save_path)
                    best_epoch = epoch
                    logger.info('save miou best model to %s'%model_save_path)
                if (miou + f_score) > (max_miou + max_fs):
                    model_save_path = os.path.join(checkpoint_dir, '%s_miou_and_fscore_best.pth'%(args.session_name))
                    torch.save(model.state_dict(), model_save_path)
                    best_epoch = epoch
                    logger.info('save miou and fscore best model to %s'%model_save_path)     

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

            model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
