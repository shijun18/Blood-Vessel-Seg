import os
import json
import glob
import pandas as pd

from utils import get_path_with_annotation,get_path_with_annotation_ratio
from utils import get_weight_path,get_weight_list


__cnn_net__ = ['unet','unet++','FPN','deeplabv3+','att_unet', \
                'res_unet','sfnet','vnet','vnet_lite','sanet','icnet']

__trans_net__ = ['UTNet','UTNet_encoder','TransUNet','ResNet_UTNet','SwinUNet']
__encoder_name__ = ['simplenet','resnet18','resnet34','resnet50','se_resnet50', \
                   'resnext50_32x4d','timm-resnest50d','mobilenetv3_large_075','xception', \
                    'efficientnet-b0', 'efficientnet-b5','timm_efficientnet_b0', 'timm_efficientnet_b5']
__new_encoder_name__ = ['swin_transformer']
__mode__ = ['cls','seg','mtl']

PATH = {
    'BloodVessel':{
        '2d':'./dataset/BloodVessel/npy_data/2d_data',
        '3d':'./dataset/BloodVessel/npy_data/train'
    },
}
    
DISEASE = 'BloodVessel' 
MODE = 'seg'
NET_NAME = 'sfnet'
ENCODER_NAME = 'resnet50'
VERSION = 'v7.3.4-roi'


DEVICE = '1'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = True
# True if use external pre-trained model 
EX_PRE_TRAINED = True if 'pretrain' in VERSION else False
# True if use resume model
RESUME = False
# [1-N]
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = 1   # or [1-N]
NUM_CLASSES = 2  # 2 for binary, more for multiple classes
ROI_NAME = 'BV'
#---------------------------------

#--------------------------------- mode and data path setting

if 'vnet' in NET_NAME:
    PATH_LIST = glob.glob(os.path.join(PATH[DISEASE]['3d'],'*.hdf5'))
    INPUT_SHAPE = (128,256,256)
    BATCH_SIZE = 4
else:
    PATH_LIST = glob.glob(os.path.join(PATH[DISEASE]['2d'],'*.hdf5'))
    INPUT_SHAPE = (448,448)
    BATCH_SIZE = 64


# CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,'cls',VERSION.replace('-cls','').replace('-freeze',''),ROI_NAME,str(CURRENT_FOLD))
CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))

WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

if PRE_TRAINED:
    # WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/{}/{}'.format(DISEASE,'cls',VERSION.replace('-cls','').replace('-freeze',''),ROI_NAME))
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME))
else:
     WEIGHT_PATH_LIST = None


INIT_TRAINER = {
    'net_name':NET_NAME,
    'encoder_name':ENCODER_NAME,
    'lr':1e-3, 
    'n_epoch':250,
    'channels':1,
    'num_classes':NUM_CLASSES, 
    'roi_number':ROI_NUMBER,
    'scale':[[0,500]],
    'input_shape':INPUT_SHAPE,
    'crop':0,
    'batch_size':BATCH_SIZE,
    'num_workers':2*GPU_NUM,
    'device':DEVICE,
    'pre_trained':PRE_TRAINED,
    'ex_pre_trained':EX_PRE_TRAINED,
    'resume':RESUME,
    'weight_path':WEIGHT_PATH,
    'use_ssl':None if 'ssl' not in VERSION else 'ssl',
    'weight_decay': 0.0001,
    'momentum': 0.99,
    'gamma': 0.1,
    'milestones': [30,60,90],
    'T_max':5,
    'mode':MODE,
    'topk':20,
    'use_fp16':True, #False if the machine you used without tensor core
    'aux_deepvision':False if 'sup' not in VERSION else True
 }
#---------------------------------

__seg_loss__ = ['TopkCEPlusDice','TopKLoss','DiceLoss','CEPlusDice','CELabelSmoothingPlusDice','OHEM','Cross_Entropy']
__cls_loss__ = ['BCEWithLogitsLoss']
__mtl_loss__ = ['BCEPlusDice','BCEPlusTopk']
# Arguments when perform the trainer 
loss_index = 0 if len(VERSION.split('.')) == 2 else eval(VERSION.split('.')[-1].split('-')[0])
if MODE == 'cls':
    LOSS_FUN = __cls_loss__[loss_index]
elif MODE == 'seg' :
    LOSS_FUN = __seg_loss__[loss_index]
else:
    LOSS_FUN = __mtl_loss__[loss_index]

print('>>>>> loss fun:%s'%LOSS_FUN)

SETUP_TRAINER = {
    'output_dir':'./ckpt/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
    'log_dir':'./log/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME), 
    'optimizer':'AdamW',
    'loss_fun':LOSS_FUN,
    'class_weight':None, #[1,4]
    'lr_scheduler':'CosineAnnealingWarmRestarts',#'CosineAnnealingWarmRestarts','MultiStepLR',
    'freeze_encoder': False if 'freeze' not in VERSION else True,
    'get_roi': False if 'roi' not in VERSION else True,
    'monitor': 'val_acc' if MODE == 'cls' else 'train_run_dice'
  }
#---------------------------------

TEST_PATH = None