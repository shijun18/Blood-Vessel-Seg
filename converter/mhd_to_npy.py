import sys
sys.path.append('..')
import os
import glob
from tqdm import tqdm
import time
import shutil
import numpy as np


from utils import save_as_hdf5,mhd_reader,save_as_nii



def mhd_to_nii(image_path,label_path,save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    path_list = os.listdir(image_path)

    id_list = set([case.split('.')[0] for case in path_list])
    print(len(id_list))
    start = time.time()
    for ID in tqdm(id_list):
       
        img_path = glob.glob(os.path.join(image_path, ID + '.mhd'))[0]
        lab_path = glob.glob(os.path.join(label_path, ID + '.mhd'))[0]
   
        _,images = mhd_reader(img_path)
        _,labels = mhd_reader(lab_path)

        print(images.shape)
        print(labels.shape)

        print(np.unique(labels))

        nii_img_path = os.path.join(save_path, ID + '_img.nii.gz')
        nii_lab_path = os.path.join(save_path, ID + '_lab.nii.gz')

        save_as_nii(images.astype(np.int16), nii_img_path)
        save_as_nii(labels.astype(np.uint8), nii_lab_path)

    print("run time: %.3f" % (time.time() - start))



def mhd_to_hdf5(image_path,label_path,save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    path_list = os.listdir(image_path)

    id_list = set([case.split('.')[0] for case in path_list])
    print(len(id_list))
    start = time.time()
    for ID in tqdm(id_list):
       
        img_path = glob.glob(os.path.join(image_path, ID + '.mhd'))[0]
        lab_path = glob.glob(os.path.join(label_path, ID + '.mhd'))[0]
   
        _,images = mhd_reader(img_path)
        _,labels = mhd_reader(lab_path)

        print(images.shape)
        print(labels.shape)

        print(np.unique(labels))

        hdf5_path = os.path.join(save_path, ID + '.hdf5')

        save_as_hdf5(images.astype(np.int16), hdf5_path, 'image')
        save_as_hdf5(labels.astype(np.uint8), hdf5_path, 'label')

    print("run time: %.3f" % (time.time() - start))


if __name__ == "__main__":

    # image_path = '../dataset/BloodVessel/raw_data/train/image'
    # label_path = '../dataset/BloodVessel/raw_data/train/label'
    # save_path = '../dataset/BloodVessel/npy_data/train/'

    # mhd_to_hdf5(image_path,label_path,save_path)


    image_path = '../dataset/BloodVessel/raw_data/train/image'
    label_path = '../dataset/BloodVessel/raw_data/train/label'
    save_path = '../dataset/BloodVessel/nii_data/train/'
    mhd_to_nii(image_path,label_path,save_path)