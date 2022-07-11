import os
import pandas as pd
import numpy as np

def statistic_csv(csv_list,save_path,col):
    data = []
    for csv in csv_list:
        df = pd.read_csv(csv,sep=',')
        del df['0']
        data_item =  np.asarray(df,dtype=np.float32)[:,1:]
        data_item = np.where(np.logical_or(data_item==0.,data_item==1.0),np.nan,data_item)
        data_mean = np.nanmean(data_item,axis=0)
        # print(data_mean)
        data.append(np.round(data_mean,decimals=4))
    data = np.stack(data,axis=0)
    data = np.hstack((data,np.nanmean(data,axis=1)[:,None]))
    mean = np.round(np.mean(data,axis=0)[None,:],decimals=4)
    # print(mean)
    std = np.round(np.std(data,axis=0)[None,:],decimals=4)
    data = np.vstack((data,mean,std))
    df = pd.DataFrame(data=data,columns=col)
    df.to_csv(save_path,index=False)

if __name__ == '__main__':
    disease = 'THOR_GTV' #THOR_GTV
    mode = 'seg'
    # version_list = ['v7.1-roi-zero','v7.1-roi-equal','v7.1-roi-half','v7.1-roi-quar','v7.1-roi-half-ssl']
    # version_list = ['v7.1.1-roi-half']
    version_list = ['v10.1-roi-all-x16']
    col_dict = {
        'HaN_GTV':["GTV","Total"],
        'THOR_GTV':["GTV","Total"]
    }
    roi_name_dict = {
        'HaN_GTV':"GTV_mtl",
        'THOR_GTV':"GTV_equal"
    }

    roi_name = roi_name_dict[disease]

    col = col_dict[disease]

    for version in version_list:
        save_dir = f'./result/{disease}/{mode}/{version}/{roi_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # dice_csv_list = [f'./result/{disease}/{mode}/{version}/{roi_name}/fold{str(i)}_dice.csv' for i in range(1,6)]
        # save_path = f'./result/{disease}/{mode}/{version}/{roi_name}/dice.csv'
        # dice_list = statistic_csv(dice_csv_list,save_path,col)

        # hd_csv_list = [f'./result/{disease}/{mode}/{version}/{roi_name}/fold{str(i)}_hd.csv' for i in range(1,6)]
        # save_path = f'./result/{disease}/{mode}/{version}/{roi_name}/hd.csv'
        # hd_list = statistic_csv(hd_csv_list,save_path,col)

        dice_csv_list = [f'./result/{disease}/{mode}/{version}/{roi_name}/ts_fold{str(i)}_dice.csv' for i in range(1,6)]
        save_path = f'./result/{disease}/{mode}/{version}/{roi_name}/ts_dice.csv'
        dice_list = statistic_csv(dice_csv_list,save_path,col)

        hd_csv_list = [f'./result/{disease}/{mode}/{version}/{roi_name}/ts_fold{str(i)}_hd.csv' for i in range(1,6)]
        save_path = f'./result/{disease}/{mode}/{version}/{roi_name}/ts_hd.csv'
        hd_list = statistic_csv(hd_csv_list,save_path,col)