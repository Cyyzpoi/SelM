from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
cfg = edict()
cfg.MASK_NUM = 10 # 10 for fully supervised
cfg.NUM_CLASSES = 71 # 70 + 1 background

###############################
# TRAIN
cfg.TRAIN = edict()

cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "../pretrained backbone/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = True #! notice
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "./torchvggish/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "../pretrained backbone/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "../pretrained backbone/pvt_v2_b5.pth"

cfg.TRAIN.FINE_TUNE_SSSS = False
cfg.TRAIN.PRETRAINED_S4_AVS_WO_TPAVI_PATH = "../single_source_scripts/logs/ssss_20220118-111301/checkpoints/checkpoint_29.pth.tar"
cfg.TRAIN.PRETRAINED_S4_AVS_WITH_TPAVI_PATH = "../single_source_scripts/logs/ssss_20220118-112809/checkpoints/checkpoint_68.pth.tar"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.CROP_IMG_AND_MASK = True
cfg.DATA.CROP_SIZE = 224 # short edge

cfg.DATA.META_CSV_PATH = "../data/AVSS/metadata.csv" #! notice: you need to change the path
cfg.DATA.LABEL_IDX_PATH = "../data/AVSS/label2idx.json" #! notice: you need to change the path

cfg.DATA.DIR_BASE = "../data/AVSS" #! notice: you need to change the path
cfg.DATA.IMG_SIZE = (224, 224)
###############################
cfg.DATA.RESIZE_PRED_MASK = True
cfg.DATA.SAVE_PRED_MASK_IMG_SIZE = (360, 240) # (width, height)




if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()