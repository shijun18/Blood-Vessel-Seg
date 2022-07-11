import os,glob
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def metadata_reader(data_path):

    info = []
    data = sitk.ReadImage(data_path)
    # print(data)
    size = list(data.GetSize()[:2])
    z_size = data.GetSize()[-1]
    thick_ness = data.GetSpacing()[-1]
    pixel_spacing = list(data.GetSpacing()[:2])
    info.append(size)
    info.append(z_size)
    info.append(thick_ness)
    info.append(pixel_spacing)
    return info

# All samples are saved in the same folder
def get_metadata(input_path, save_path):

    path_list = os.listdir(input_path)
    id_list = set([case.split('.')[0] for case in path_list])
    print(len(id_list))
    info = []
    for ID in tqdm(id_list):
        info_item = [ID]
        image_path = glob.glob(os.path.join(input_path, ID + '.mhd'))[0]
        info_item.extend(metadata_reader(image_path))
        info.append(info_item)
    col = ['id', 'size', 'num', 'thickness', 'pixel_spacing']
    info_data = pd.DataFrame(columns=col, data=info)
    info_data.to_csv(save_path, index=False)

if __name__ == "__main__":

    data_path = '../dataset/BloodVessel/raw_data/train/image'

    get_metadata(data_path,'./bv_metadata.csv')
    