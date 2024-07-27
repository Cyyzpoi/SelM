from easydict import EasyDict as edict
import pdb

"""
default config
"""
cfg = edict()

##############################
# TRAIN
cfg.TRAIN = edict()
# TRAIN.SCHEDULER
cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "../pretrained backbone/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = False
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "../../pretrained_backbones/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = False
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "../pretrained backbone/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "../pretrained backbone/pvt_v2_b5.pth"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.ANNO_CSV = "../data/avsbench_data/s4_meta_data.csv"
cfg.DATA.DIR_IMG = "../data/avsbench_data/s4_data/s4_data/visual_frames"
cfg.DATA.DIR_AUDIO_LOG_MEL = "../data/avsbench_data/s4_data/s4_data/audio_log_mel"
cfg.DATA.DIR_MASK = "../data/avsbench_data/s4_data/s4_data/gt_masks"
cfg.DATA.IMG_SIZE = (224, 224)
###############################





if __name__ == "__main__":
    print(cfg)
    pdb.set_trace()
