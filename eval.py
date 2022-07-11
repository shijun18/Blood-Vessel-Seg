import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast as autocast
from utils import get_weight_path,ensemble,mhd_reader,ensemble_v2

import SimpleITK as sitk
from torch.utils.data import Dataset
import glob

from skimage.transform import resize
import cv2
from skimage.exposure.exposure import rescale_intensity
from skimage.draw import polygon

import warnings
warnings.filterwarnings('ignore')

class Trunc_and_Normalize(object):
  '''
  truncate gray scale and normalize to [0,1]
  '''
  def __init__(self, scale):
    self.scale = scale
    assert len(self.scale) == 2, 'scale error'

  def __call__(self, sample):
 
        # gray truncation
        image = sample['image']
        image = image - self.scale[0]
        gray_range = self.scale[1] - self.scale[0]
        image[image < 0] = 0
        image[image > gray_range] = gray_range
        
        image = image / gray_range

        sample['image'] = image

        return sample


class CropResize(object):
    '''
    Data preprocessing.
    Adjust the size of input data to fixed size by cropping and resize
    Args:
    - dim: tuple of integer, fixed size
    - crop: single integer, factor of cropping, H/W ->[crop:-crop,crop:-crop]
    '''
    def __init__(self, dim=None,crop=0):
        self.dim = dim
        self.crop = crop

    def __call__(self, sample):

        # image: numpy array
        # crop
        image = sample['image']
        if self.crop != 0:
            if len(image.shape) > 2:
                image = image[:,self.crop:-self.crop, self.crop:-self.crop]
            else:
                image = image[self.crop:-self.crop, self.crop:-self.crop]
        # resize
        if self.dim is not None and image.shape != self.dim:
            image = resize(image, self.dim, anti_aliasing=True,mode='constant')

        sample['image'] = image

        return sample



class To_Tensor(object):
    '''
    Convert the data to torch Tensor.
    '''

    def __call__(self,sample):
        # expand dims
        image = sample['image']
        image = np.expand_dims(image,axis=0)
        sample['image'] = torch.from_numpy(image)
        return sample


class Get_ROI(object):
    def __init__(self, keep_size=12,pad_flag=False):
        self.keep_size = keep_size
        self.pad_flag = pad_flag
    
    def __call__(self, sample):
        '''
        sample['image'] must be scaled to (0~1)
        '''
        image = sample['image']
        h,w = image.shape
        roi = self.get_body(image)

        if np.sum(roi) != 0:
            roi_nz = np.nonzero(roi)
            roi_bbox = [
                np.maximum((np.amin(roi_nz[0]) - self.keep_size), 0), # left_top x
                np.maximum((np.amin(roi_nz[1]) - self.keep_size), 0), # left_top y
                np.minimum((np.amax(roi_nz[0]) + self.keep_size), h), # right_bottom x
                np.minimum((np.amax(roi_nz[1]) + self.keep_size), w)  # right_bottom y
            ]
        else:
            roi_bbox = [0,0,h,w]

        image = image[roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]]
        # pad
        if self.pad_flag:
            nh, nw = roi_bbox[2] - roi_bbox[0], roi_bbox[3] - roi_bbox[1]
            if abs(nh - nw) > 1:
                if nh > nw:
                    pad = ((0,0),(int(nh-nw)//2,int(nh-nw)//2))
                else:
                    pad = ((int(nw-nh)//2,int(nw-nh)//2),(0,0))
                image = np.pad(image,pad,'constant')

        sample['image'] = image
        sample['bbox'] = roi_bbox
        return sample

    def get_body(self,image):
        body_array = np.zeros_like(image, dtype=np.uint8)
        img = rescale_intensity(image, out_range=(0, 255))
        img = img.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        body = cv2.erode(img, kernel, iterations=1)
        blur = cv2.GaussianBlur(body, (5, 5), 0)
        _, body = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        area = [[c, cv2.contourArea(contours[c])] for c in range(len(contours))]
        area.sort(key=lambda x: x[1], reverse=True)
        body = np.zeros_like(body, dtype=np.uint8)
        for j in range(min(len(area),3)):
            if area[j][1] > area[0][1] / 20:
                contour = contours[area[j][0]]
                r = contour[:, 0, 1]
                c = contour[:, 0, 0]
                rr, cc = polygon(r, c)
                body[rr, cc] = 1
        body_array = cv2.medianBlur(body, 5)

        return body_array


class RandomFlip(object):

    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):
        # image: numpy array, (D,H,W)
        # label: integer, 0,1,..
        image = sample['image']

        if 'v' in self.mode:
            if len(image.shape) == 3:
                image = image[:, ::-1, ...]
            else:
                image = image[::-1, ...]

        elif 'h' in self.mode:
            image = image[..., ::-1]

        sample['image'] = image
        return sample




class InfDataset(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - numpy_array: input data with 3d shape -> (d,h,w)
    - transform: the data augmentation methods
    '''
    def __init__(self,numpy_array, transform=None):
        self.data = numpy_array
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,index):
        image = self.data[index]
        sample = {'image': image}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def resize_and_pad(pred,num_classes,target_shape,bboxs):
    from skimage.transform import resize
    final_pred = []

    for bbox, pred_item in zip(bboxs,pred):
        h,w = bbox[2]-bbox[0], bbox[3]-bbox[1]
        new_pred = np.zeros(target_shape,dtype=np.float32)

        for z in range(1,num_classes):
            roi_pred = resize((pred_item == z).astype(np.float32),(h,w),mode='constant')
            new_pred[bbox[0]:bbox[2],bbox[1]:bbox[3]][roi_pred>=0.5] = z
           
        final_pred.append(new_pred)
    final_pred = np.stack(final_pred,axis=0)
    return final_pred

def get_net(net_name,encoder_name,channels=1,num_classes=2,input_shape=(512,512),**kwargs):

    if net_name == 'unet':
        if encoder_name in ['simplenet','swin_transformer','swinplusr18']:
            from model.unet import unet
            net = unet(net_name,
                encoder_name=encoder_name,
                in_channels=channels,
                classes=num_classes,
                aux_classifier=True)
        else:
            import segmentation_models_pytorch as smp
            net = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    elif net_name == 'unet++':
        if encoder_name is None:
            raise ValueError(
                "encoder name must not be 'None'!"
            )
        else:
            import segmentation_models_pytorch as smp
            net = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )

    elif net_name == 'FPN':
        if encoder_name is None:
            raise ValueError(
                "encoder name must not be 'None'!"
            )
        else:
            import segmentation_models_pytorch as smp
            net = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    
    elif net_name == 'deeplabv3+':
        if encoder_name in ['swinplusr18']:
            from model.deeplabv3plus import deeplabv3plus
            net = deeplabv3plus(net_name,
                encoder_name=encoder_name,
                in_channels=channels,
                classes=num_classes)
        else:
            import segmentation_models_pytorch as smp
            net = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
    elif net_name == 'res_unet':
        from model.res_unet import res_unet
        net = res_unet(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes,
            **kwargs)

    elif net_name == 'sfnet':
            from model.sfnet import sfnet
            net = sfnet(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes,
            **kwargs)
    
    elif net_name == 'sanet':
            from model.sanet import sanet
            net = sanet(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes,
            **kwargs)
    
    elif net_name == 'att_unet':
        from model.att_unet import att_unet
        net = att_unet(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes)
    
    elif net_name == 'bisenetv1':
        from model.bisenetv1 import bisenetv1
        net = bisenetv1(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes)
    
    elif net_name.startswith('vnet'):
        import model.vnet as vnet
        net = vnet.__dict__[net_name](
            init_depth=input_shape[0],
            in_channels=channels,
            classes=num_classes,
        )

    ## external transformer + U-like net
    elif net_name == 'UTNet':
        from model.trans_model.utnet import UTNet
        net = UTNet(channels, base_chan=32,num_classes=num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    elif net_name == 'UTNet_encoder':
        from model.trans_model.utnet import UTNet_Encoderonly
        # Apply transformer blocks only in the encoder
        net = UTNet_Encoderonly(channels, base_chan=32, num_classes=num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    elif net_name =='TransUNet':
        from model.trans_model.transunet import VisionTransformer as ViT_seg
        from model.trans_model.transunet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = num_classes 
        config_vit.n_skip = 3 
        config_vit.patches.grid = (int(input_shape[0]/16), int(input_shape[1]/16))
        net = ViT_seg(config_vit, img_size=input_shape[0], num_classes=num_classes)
        

    elif net_name == 'ResNet_UTNet':
        from model.trans_model.resnet_utnet import ResNet_UTNet
        net = ResNet_UTNet(channels, num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True)
    
    elif net_name == 'SwinUNet':
        from model.trans_model.swin_unet import SwinUnet, SwinUnet_config
        config = SwinUnet_config()
        config.num_classes = num_classes
        config.in_chans = channels
        net = SwinUnet(config, img_size=input_shape[0], num_classes=num_classes)
    
    return net


def eval_process(numpy_array,config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device
    device = torch.device("cuda:0")
    test_transformer = transforms.Compose([
                Trunc_and_Normalize(config.scale),
                Get_ROI(pad_flag=False) if config.get_roi else transforms.Lambda(lambda x:x),
                RandomFlip(config.flip) if config.flip else transforms.Lambda(lambda x:x),
                CropResize(dim=config.input_shape,crop=config.crop),
                To_Tensor()
            ])

    if len(config.input_shape) == 3:
        numpy_array = np.expand_dims(numpy_array,axis=0)

    test_dataset = InfDataset(numpy_array,transform=test_transformer)

    test_loader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    
    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    print(weight_path)

    s_time = time.time()
    # get net
    net = get_net(config.net_name,
            config.encoder_name,
            config.channels,
            config.num_classes,
            config.input_shape,
            aux_deepvision=config.aux_deepvision,
            aux_classifier=config.aux_classifier
    )
    checkpoint = torch.load(weight_path,map_location='cpu')
    msg=net.load_state_dict(checkpoint['state_dict'],strict=False)
    
    print(msg)
    get_net_time = time.time() - s_time
    print('define net and load weight need time:%.3f'%(get_net_time))

    pred = []
    s_time = time.time()

    net = net.to(device)
    net.eval()
    move_time = time.time() - s_time
    print('move net to GPU need time:%.3f'%(move_time))

    with torch.no_grad():
        for _, sample in enumerate(test_loader):
            data = sample['image']
            ####
            # data = data.cuda()
            data = data.to(device)
            with autocast(True):
                output = net(data)
                
            if isinstance(output,tuple) or isinstance(output,list):
                seg_output = output[0]
            else:
                seg_output = output
            # seg_output = torch.argmax(seg_output,1).detach().cpu().numpy()
            seg_output = (seg_output[:,1] > 0.4).detach().cpu().numpy()                         

            if config.get_roi:
                bboxs = torch.stack(sample['bbox'],dim=0).cpu().numpy().T
                seg_output = resize_and_pad(seg_output,config.num_classes,config.input_shape,bboxs)

            pred.append(seg_output)
    pred = np.concatenate(pred,axis=0).squeeze().astype(np.uint8)

    if config.flip == 'h':
        pred = pred[...,::-1]
    elif config.flip == 'v':
        pred = pred[:,::-1,]

    return pred,move_time+get_net_time


class Config:

    num_classes_dict = {
        'BloodVessel':2,
    }
    scale_dict = {
        'BloodVessel':[0,500]
    }

    roi_dict = {
        'BloodVessel':'BV'
    }
    
    input_shape = (448,448) #(256,256)(512,512)(448,448) (96,256,256)
    channels = 1
    crop = 0
    roi_number = 1
    batch_size = 32
    
    disease = 'BloodVessel'
    mode = 'seg'
    num_classes = num_classes_dict[disease]
    scale = scale_dict[disease]

    net_name = 'sanet'
    encoder_name = 'resnet18'
    version = 'v10.1-roi'
    flip = None
    
    fold = 1
    device = "0"
    roi_name = roi_dict[disease]
    
    get_roi = False if 'roi' not in version else True
    aux_deepvision = False if 'sup' not in version else True
    aux_classifier = mode != 'seg'
    ckpt_path = f'./ckpt/{disease}/{mode}/{version}/{roi_name}'

    post_fix = f'_{flip}' if flip else ''


if __name__ == '__main__':

    # test data
    data_path_dict = {
        'BloodVessel':'./dataset/BloodVessel/raw_data/test'
    }
    
    start = time.time()
    config = Config()
    data_path = data_path_dict[config.disease]
    sample_list = glob.glob(os.path.join(data_path,'*.mhd'))
    sample_list.sort()

    ensemble_result = {}
    meta_data = {}
    for fold in [1,2,3,4,5]:
        print('>>>>>>>>>>>> Fold%d >>>>>>>>>>>>'%fold)

        config.fold = fold
        config.ckpt_path = f'./ckpt/{config.disease}/{config.mode}/{config.version}/{config.roi_name}/fold{str(fold)}'
        save_dir = f'./result/{config.disease}/{config.mode}/{config.version}/{config.roi_name}{config.post_fix}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for sample in sample_list:
            print('>>>>>>>>>>>> %s is being processed'%sample)
            
            item_metadata,input_image = mhd_reader(sample)

            print(f'data size: {input_image.shape}')

            sample_start = time.time()
            pred,extra_time = eval_process(input_image,config)
            
            total_time = time.time() - sample_start 
            actual_time = total_time - extra_time
            print('total time:%.3f'%total_time)
            print('actual time:%.3f'%actual_time)

            if sample not in ensemble_result:
                ensemble_result[sample] = []
                meta_data[sample] = {
                    'spacing': item_metadata.GetSpacing(),
                    'origin': item_metadata.GetOrigin(),
                    'direction': item_metadata.GetDirection()}
            ensemble_result[sample].append(pred)

    #### for ensemble
    for sample in sample_list:
        print('>>>> %s in post processing'%sample)
        ensemble_pred = ensemble(np.stack(ensemble_result[sample],axis=0),config.num_classes - 1)
        ensemble_pred_v2 = ensemble_v2(np.stack(ensemble_result[sample],axis=0),config.num_classes - 1)
        print(f'v1:{np.sum(ensemble_pred)},v2:{np.sum(ensemble_pred_v2)}')

        #### save 
        ensemble_pred = ensemble_pred_v2
        sitk_data = sitk.GetImageFromArray(ensemble_pred.astype(np.uint8))
        sitk_data.SetSpacing(meta_data[sample]['spacing'])
        sitk_data.SetOrigin(meta_data[sample]['origin'])
        sitk_data.SetDirection(meta_data[sample]['direction'])
        save_path = os.path.join(save_dir,os.path.basename(sample))
        sitk.WriteImage(sitk_data,save_path,useCompression=False)