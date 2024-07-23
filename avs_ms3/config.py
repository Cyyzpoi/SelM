from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.BATCH_SIZE = 4 # default 16
cfg.LAMBDA_1 = 5 # default: 5
cfg.MASK_NUM = 1 # 5 for fully supervised, 1 for weakly supervised

###############################
# TRAIN
cfg.TRAIN = edict()

cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avs_scripts/avs_ms3/model/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avs_scripts/avs_ms3/model/vggish-10086976.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avs_scripts/avs_ms3/model/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avs_scripts/avs_ms3/model/pvt_v2_b5.pth"

cfg.TRAIN.FINE_TUNE_SSSS = False
cfg.TRAIN.PRETRAINED_S4_aAVS_WO_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"
cfg.TRAIN.PRETRAINED_S4_AVS_WITH_TPAVI_PATH = "../avs_s4/train_logs/checkpoints/checkpoint_xxx.pth.tar"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = "/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avsbench_data/ms3_meta_data.csv"
cfg.DATA.DIR_IMG = "/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avsbench_data/ms3_data/ms3_data/visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = "/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avsbench_data/ms3_data/ms3_data/audio_log_mel"
cfg.DATA.DIR_MASK = "/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avsbench_data/ms3_data/ms3_data/gt_masks"
cfg.DATA.DIR_AUDIO_WAV="/home/cyyzpoi/workplace/AVS/AudioClip-AVS/avsbench_data/ms3_data/ms3_data/audio_wav"
cfg.DATA.IMG_SIZE = (224, 224)
###############################



if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()