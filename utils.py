import os
import pandas as pd
import h5py
import numpy as np
import torch
import random
from skimage.metrics import hausdorff_distance
from skimage import measure
import copy

import SimpleITK as sitk


def mhd_reader(data_path):
    data = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(data).astype(np.float32)
    return data,image


def post_seg(seg_result,post_index=None,keep_max=True,keep_number=3): 
    seg_result = copy.deepcopy(seg_result)
    for i in post_index:
        tmp_seg_result = (seg_result == i).astype(np.float32)
        labels = measure.label(tmp_seg_result)
        area = []
        for j in range(1,np.amax(labels) + 1):
            area.append(np.sum(labels == j))
        if keep_max:
            if len(area) != 0:
                tmp_seg_result[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0
            seg_result[seg_result == i] = 0
            seg_result[tmp_seg_result == 1] = i
        else:
            area_dict = {}
            for i in range(len(area)):
                area_dict[i+1] = area[i]
            area_list = sorted(area_dict.items(), key=lambda x:x[1])
            for i in range(max(0,len(area_list) - keep_number)):
                tmp_seg_result[np.logical_and(labels > 0, labels == area_list[i][0])] = 0
            seg_result[seg_result == i] = 0
            seg_result[tmp_seg_result == 1] = i

    return seg_result



def ensemble(array,num_classes):
    # print(array.shape)
    _C = array.shape[0]
    result = np.zeros(array.shape[1:],dtype=np.uint8)
    for i in range(num_classes):
        roi = np.sum((array == i+1).astype(np.uint8),axis=0)
        # print(roi.shape)
        result[roi > (_C // 2)] = i+1
    return result


def ensemble_v2(array,num_classes):
    # print(array.shape)
    _C = array.shape[0]
    result = np.zeros(array.shape[1:],dtype=np.uint8)
    for i in range(num_classes):
        roi = np.sum((array == i+1).astype(np.uint8),axis=0)
        # print(roi.shape)
        result[roi > 0] = i+1
    return result


def csv_reader_single(csv_file,key_col=None,value_col=None):
    '''
    Extracts the specified single column, return a single level dict.
    The value of specified column as the key of dict.

    Args:
    - csv_file: file path
    - key_col: string, specified column as key, the value of the column must be unique. 
    - value_col: string,  specified column as value
    '''
    file_csv = pd.read_csv(csv_file)
    key_list = file_csv[key_col].values.tolist()
    value_list = file_csv[value_col].values.tolist()
    
    target_dict = {}
    for key_item,value_item in zip(key_list,value_list):
        target_dict[key_item] = value_item

    return target_dict


def binary_dice(y_true, y_pred):
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def multi_dice(y_true,y_pred,num_classes):
    dice_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        dice = binary_dice(true,pred)
        dice_list.append(dice)
    
    dice_list = [round(case, 4) for case in dice_list]
    
    return dice_list, round(np.mean(dice_list),4)


def hd_2d(true,pred):
    hd_list = []
    for i in range(true.shape[0]):
        if np.sum(true[i]) != 0 and np.sum(pred[i]) != 0:
            hd_list.append(hausdorff_distance(true[i],pred[i]))
    
    return np.mean(hd_list)

def multi_hd(y_true,y_pred,num_classes):
    hd_list = []
    for i in range(num_classes):
        true = (y_true == i+1).astype(np.float32)
        pred = (y_pred == i+1).astype(np.float32)
        hd = hd_2d(true,pred)
        hd_list.append(hd)
    
    hd_list = [round(case, 4) for case in hd_list]
    
    return hd_list, round(np.mean(hd_list),4)




def save_as_nii(data, save_path):
    sitk_data = sitk.GetImageFromArray(data)
    sitk.WriteImage(sitk_data, save_path)



def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def get_path_with_annotation(input_path,path_col,tag_col):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    final_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            final_list.append(path)
    
    return final_list


def get_path_with_annotation_ratio(input_path,path_col,tag_col,ratio=0.5):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    with_list = []
    without_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            with_list.append(path)
        else:
            without_list.append(path)
    if int(len(with_list)/ratio) < len(without_list):
        random.shuffle(without_list)
        without_list = without_list[:int(len(with_list)/ratio)]    
    return with_list + without_list


def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/10e9))
    print('%.3f M' % (params/10e6))


def get_weight_list(ckpt_path):
    path_list = []
    for fold in os.scandir(ckpt_path):
        if fold.is_dir():
            weight_path = os.listdir(fold.path)
            weight_path.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            path_list.append(os.path.join(fold.path,weight_path[-1]))
            # print(os.path.join(fold.path,weight_path[-1]))
    path_list.sort(key=lambda x:x.split('/')[-2])
    return path_list


def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None
    
    

def remove_weight_path(ckpt_path,retain=3):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=3):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain)
        else:
            remove_weight_path(ckpt_path,retain)
            break  

def rename_weight_path(ckpt_path):
    if os.path.isdir(ckpt_path):
        for pth in os.scandir(ckpt_path):
            if ':' in pth.name:
                new_pth = pth.path.replace(':','=')
                print(pth.name,' >>> ',os.path.basename(new_pth))
                os.rename(pth.path,new_pth)
            else:
                break


def dfs_rename_weight(ckpt_path):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_rename_weight(sub_path.path)
        else:
            rename_weight_path(ckpt_path)
            break  



if __name__ =='__main__':
    import pickle

    # nii_dir = './result/BloodVessel/seg/nnunet/nii/2d/fold123/labels'
    # mhd_dir = './result/BloodVessel/seg/nnunet/mhd/2d/fold123/labels'

    # if not os.path.exists(mhd_dir):
    #     os.makedirs(mhd_dir)
    
    # item_list = os.listdir(nii_dir)
    # for item in item_list:
    #     ID = item.split('.')[0]
    #     new_item = os.path.join(mhd_dir,f'{ID}.mhd')
    #     item_data = sitk.ReadImage(os.path.join(nii_dir,item))
    #     item_array = sitk.GetArrayFromImage(item_data)

    #     sitk_data = sitk.GetImageFromArray(item_array.astype(np.uint8))
    #     sitk_data.SetSpacing(item_data.GetSpacing())
    #     sitk_data.SetOrigin(item_data.GetOrigin())
    #     sitk_data.SetDirection(item_data.GetDirection())
    #     sitk.WriteImage(sitk_data,new_item,useCompression=False)


    
    # npz_dir = './result/BloodVessel/seg/nnunet/npz/3d_cascade_fullres/fold023'
    # mhd_dir = './result/BloodVessel/seg/nnunet/mhd/3d_cascade_fullres/fold023-prob10/labels'

    # npz_dir = './result/BloodVessel/seg/nnunet/npz/3d_cascade_fullres/fold023'
    # mhd_dir = './result/BloodVessel/seg/nnunet/mhd/3d_cascade_fullres/fold023-prob17/labels'

    # npz_dir = './result/BloodVessel/seg/nnunet/npz/ensemble_3d_and_3d_cascade/fold01234'
    # mhd_dir = './result/BloodVessel/seg/nnunet/mhd/ensemble_3d_and_3d_cascade/fold01234-prob15/labels'

    # npz_dir = './result/BloodVessel/seg/nnunet/npz/ensemble_2d_3d_and_3d_cascade/fold01234'
    # mhd_dir = './result/BloodVessel/seg/nnunet/mhd/ensemble_2d_3d_and_3d_cascade/fold01234-prob15/labels'
    
    
    # npz_dir = './result/BloodVessel/seg/nnunet/npz/ensemble_2d_and_3d_cascade/fold01234'
    # mhd_dir = './result/BloodVessel/seg/nnunet/mhd/ensemble_2d_and_3d_cascade/fold01234-prob15/labels'


    npz_dir = './result/BloodVessel/seg/nnunet/npz/ensemble_3d_and_3d_cascade_fake/fold01234'
    mhd_dir = './result/BloodVessel/seg/nnunet/mhd/ensemble_3d_and_3d_cascade_fake/fold01234-prob00005/labels'

    if not os.path.exists(mhd_dir):
        os.makedirs(mhd_dir)
    
    item_list = os.listdir(npz_dir)
    item_list.sort()
    for item in item_list:
        if '.npz' in item and item in ['test06.npz']:
            final_result = np.zeros((128,448,448),dtype=np.uint8)
            ID = item.split('.')[0]
            new_item = os.path.join(mhd_dir,f'{ID}.mhd')
            item_array = np.array(np.load(os.path.join(npz_dir,item))['softmax'])
            print('0.5:',np.sum(np.argmax(item_array,axis=0)))
            # print(item_array.shape)
            final_result[:,1:,1:] = (item_array[1] > 0.00005)
            print('0.00005:',np.sum(final_result))
            with open(os.path.join(npz_dir,f'{ID}.pkl'),'rb') as f:
                item_data = pickle.load(f)[0]
                # print(item_data[0])
            sitk_data = sitk.GetImageFromArray(final_result.astype(np.uint8))
            sitk_data.SetSpacing(item_data['original_spacing'])
            sitk_data.SetOrigin(item_data['itk_origin'])
            sitk_data.SetDirection(item_data['itk_direction'])
            sitk.WriteImage(sitk_data,new_item,useCompression=False)
