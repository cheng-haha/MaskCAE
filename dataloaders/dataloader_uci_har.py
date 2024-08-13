import pandas as pd
import numpy as np
import os

from dataloaders.dataloader_base import BASE_DATA
# ============================== UCI_HAR_DATA ======================================
class UCI_HAR_DATA(BASE_DATA):
    """
    The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. 
    Each person performed six activities 
    (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) 
    wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, 
    we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. 
    The experiments have been video-recorded to label the data manually. 
    The obtained dataset has been randomly partitioned into two sets, 
    where 70% of the volunteers was selected for generating the training data and 30% the test data. 

    The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters 
    and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). 
    The sensor acceleration signal, which has gravitational and body motion components, 
    was separated using a Butterworth low-pass filter into body acceleration and gravity. 
    The gravitational force is assumed to have only low frequency components, 
    therefore a filter with 0.3 Hz cutoff frequency was used. 
    From each window, a vector of features was obtained by calculating variables from the time and frequency domain. 
    See 'features_info.txt' for more details. 

        1 WALKING
        2 WALKING_UPSTAIRS
        3 WALKING_DOWNSTAIRS
        4 SITTING
        5 STANDING
        6 LAYING
    """
    def __init__(self, args):

        """
        root_path : Root directory of the data set
        difference (bool) : Whether to calculate the first order derivative of the original data
        datanorm_type (str) : Methods of data normalization: "standardization", "minmax" , "per_sample_std", "per_sample_minmax"
        
        spectrogram (bool): Whether to convert raw data into frequency representations
            scales : Depends on the sampling frequency of the data （ UCI 数据的采样频率？？）
            wavelet : Methods of wavelet transformation

        """
        self.used_cols    = [] #no use , because this data format has save each sensor in one file
        self.col_names   =  ['body_acc_x_', 'body_acc_y_', 'body_acc_z_',
                             'body_gyro_x_', 'body_gyro_y_', 'body_gyro_z_',
                             'total_acc_x_', 'total_acc_y_', 'total_acc_z_']

        self.label_map = [ 
            (1, 'WALKING'),
            (2, 'WALKING_UPSTAIRS'),
            (3, 'WALKING_DOWNSTAIRS'),
            (4, 'SITTING'),
            (5, 'STANDING'),
            (6, 'LAYING'),
        ]

        self.drop_activities = []

        # All ID used for training [ 1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
        # All ID used for Test  [ 2,  4,  9, 10, 12, 13, 18, 20, 24]
        # this is given train test split
        self.train_keys   = [ 1,  3,  5,  7,  8, 14, 15, 16, 17, 21, 22, 23, 26, 27, 28, 29, 6, 11, 19, 25, 30]
        self.vali_keys    = [ ]
        self.test_keys    = [ 2,  4,  9, 10, 12, 13, 18, 20, 24]

        self.exp_mode     = args.exp_mode
        self.down_sample:bool  \
                          = args.down_sample
        self.split_tag    = "sub"

        self.all_keys   = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
        all_keys        = np.array_split(self.all_keys,5)
        self.LOCV_keys  = [list(elem) for elem in all_keys]
        self.sub_ids_of_each_sub = {}
		
        self.file_encoding = {}


        self.labelToId = {int(x[0]): i for i, x in enumerate(self.label_map)}
        self.all_labels = list(range(len(self.label_map)))

        self.drop_activities = [self.labelToId[i] for i in self.drop_activities]
        self.no_drop_activites = [item for item in self.all_labels if item not in self.drop_activities]


        self.leave_one_out = True
        self.full_None_Overlapping = False
        self.Semi_None_Overlapping = True

        super(UCI_HAR_DATA, self).__init__(args)





    def load_all_the_data(self, root_path):

        print(" ----------------------- load all the data -------------------")
        # ====================  Load the sensor values ====================
        train_vali_path = os.path.join(root_path, "train/Inertial Signals/")
        test_path  = os.path.join(root_path, "test/Inertial Signals/")

        train_vali_dict = {}
        test_dict  = {}

        file_list = os.listdir(train_vali_path)
        for file in file_list:

            train_vali = pd.read_csv(train_vali_path + file,header=None, delim_whitespace=True)
            test  = pd.read_csv(test_path+file[:-9]+"test.txt",header=None, delim_whitespace=True)
			
            train_vali_dict[file[:-9]] = train_vali
            test_dict[file[:-9]] = test


        # =================== Define the sub id  and the label for each segments FOR  TRAIN VALI  ================
        train_vali = pd.DataFrame(np.stack([train_vali_dict[col].values.reshape(-1) for col in self.col_names], axis=1), columns = self.col_names)

        train_vali_subjects = pd.read_csv(os.path.join(root_path,"train/subject_train.txt"), header=None)
        train_vali_subjects.columns = ["subjects"]

        train_vali_label = pd.read_csv(os.path.join(root_path,"train/y_train.txt"),header=None)
        train_vali_label.columns = ["labels"]

        index = []
        labels = []
        sub_list = []

        assert train_vali_dict["body_acc_x_"].shape[0] == train_vali_subjects.shape[0]

        # repeat the id and the label for each segs 128 tims
        for i in range(train_vali_dict["body_acc_x_"].shape[0]):
            sub = train_vali_subjects.loc[i,"subjects"]
            sub_id = "{}_{}".format(sub,i)

            ac_id = train_vali_label.loc[i,"labels"]
            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)

            index.extend(128*[sub_id])
            labels.extend(128*[ac_id])
            sub_list.extend(128*[sub])

        train_vali["sub_id"] = index
        train_vali["sub"] = sub_list
        train_vali["activity_id"] = labels

        # =================== Define the sub id  and the label for each segments  FOR TEST ================
        test = pd.DataFrame(np.stack([test_dict[col].values.reshape(-1) for col in self.col_names], axis=1), columns = self.col_names)

        test_subjects = pd.read_csv(os.path.join(root_path,"test/subject_test.txt"), header=None)
        test_subjects.columns = ["subjects"]

        test_label = pd.read_csv(os.path.join(root_path,"test/y_test.txt"),header=None)
        test_label.columns = ["labels"]

        index = []
        labels = []
        sub_list = []
        assert test_dict["body_acc_x_"].shape[0] == test_subjects.shape[0]

        for i in range(test_dict["body_acc_x_"].shape[0]):
            sub = test_subjects.loc[i,"subjects"]
            sub_id = "{}_{}".format(sub,i)

            ac_id = test_label.loc[i,"labels"]

            if sub not in self.sub_ids_of_each_sub.keys():
                self.sub_ids_of_each_sub[sub] = []
            self.sub_ids_of_each_sub[sub].append(sub_id)

            index.extend(128*[sub_id])
            labels.extend(128*[ac_id])
            sub_list.extend(128*[sub])

        test["sub_id"] = index
        test["sub"] = sub_list
        test["activity_id"] = labels



        # The split may be different as the default setting, so we concat all segs together
        df_all = pd.concat([train_vali,test])


        df_dict = {}
        for i in df_all.groupby("sub_id"):
            df_dict[i[0]] = i[1]
        df_all = pd.concat(df_dict)
        
        if self.down_sample:
            # Downsampling! form 50hz to 30hz
            df_all.reset_index(drop=True,inplace=True)
            index_list = list( np.arange(0,df_all.shape[0],5/3).astype(int) )
            df_all = df_all.iloc[index_list]
        
        # ================= Label Transformation ===================
        df_all["activity_id"] = df_all["activity_id"].map(self.labelToId)
        df_all = df_all.set_index('sub_id')
        data_y = df_all.iloc[:,-1]
        data_x = df_all.iloc[:,:-1]


        data_x = data_x.reset_index()
        # drop partial sensors for special task.
        # data_x = data_x.drop(columns=['total_acc_x_', 'total_acc_y_', 'total_acc_z_'])

        return data_x, data_y