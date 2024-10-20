import pandas as pd
import numpy as np
import os
import random
import pywt
import pickle
from tqdm import tqdm
import torch
from random import sample
from dataloaders.utils import Normalizer,components_selection_one_signal,mag_3_signals,PrepareWavelets,FiltersExtention
from sklearn.utils import class_weight
from skimage.transform import resize

freq1 = 0.3
freq2 = 20


# ========================================       Data loader Base class               =============================
class BASE_DATA():

    def __init__(self, args):
        """
        root_path                      : Root directory of the data set
        freq_save_path                 : The path to save genarated Spectrogram. If the file has already been generated, Load it directly
        window_save_path               : The path to save genarated Window index. If the file has already been generated, Load it directly
                                         This could save time by avoiding generate them again.

        data_name (Str)                : the name of data set
                                       --->[TODO]

        freq (int)                     :Sampling Frequency of the correponding dataset
        representation_type (Str)      :  What kind of representation should be load 
                                       --->[time, freq, time_freq]

        difference  (Bool)             : Whether to calculate the first order derivative of the original data
        datanorm_type (Str)            : How to normalize the data 
                                       --->[standardization, minmax, per_sample_std, per_sample_minmax]

        load_all (Bool)                : This is for Freq representation data. Whether load all files in one time. this could save time by training, but it needs a lot RAM
        train_vali_quote (float)       : train vali split quote , default as 0.8

        windowsize                     :  the size of Sliding Window
        -------------------------------------------------------
		if training mode, The sliding step is 50% of the windowsize
        if test mode, The step is 10% of the windowsize. (It should be as one, But it results in to many window samples, it is difficult to generate the spectrogram)
        -------------------------------------------------------        
        drop_transition  (Bool)        : Whether to drop the transition parts between different activities
        wavelet_function (Str)         : Method to generate Spectrogram
                                       ---> []

        """
        self.root_path              = args.root_path
        self.freq_save_path         = args.freq_save_path
        self.window_save_path       = args.window_save_path
        self.data_name              = args.data_name

        window_save_path = os.path.join(self.window_save_path,self.data_name)
        if not os.path.exists(window_save_path):
            os.makedirs(window_save_path)
        self.window_save_path       = window_save_path
        self.representation_type    = args.representation_type
        self.overlap_rate           = args.overlap_rate
        self.valid_rate             = args.valid_rate
        #assert self.data_name in []
        self.freq                   = args.sampling_freq  

        self.difference             = args.difference
        self.filtering              = args.filtering
        self.magnitude              = args.magnitude
        self.datanorm_type          = args.datanorm_type
        self.load_all               = args.load_all
        self.train_vali_quote       = args.train_vali_quote
        self.windowsize             = args.windowsize
        self.drop_transition        = args.drop_transition
        # self.wavelet_function       = args.wavelet_function
        #self.wavelet_filtering      = args.wavelet_filtering
        #self.number_wavelet_filtering = args.number_wavelet_filtering


        # ======================= Load the Data =================================
        self.data_x, self.data_y = self.load_all_the_data(self.root_path)
        # data_x : sub_id, sensor_1, sensor_2,..., sensor_n , sub
        # data_y : activity_id   index:sub_id
        # update col_names
        self.col_names = list(self.data_x.columns)[1:-1]

        # ======================= Differencing the Data =================================
        if self.difference:
            print("Channel Augmentation : Differencing")
            self.data_x = self.differencing(self.data_x.set_index('sub_id').copy())

        # data_x : sub_id, sensor_1, sensor_2,..., sensor_n , sub


        # ======================== Filtering the Data =========================================
        if self.filtering:
            print("Channel Augmentation : Acc Gyro Filtering")
            self.data_x = self.Sensor_data_noise_grav_filtering(self.data_x.set_index('sub_id').copy())

        # ======================== Mag the Data =====================================

        if self.magnitude:
            print("Channel Augmentation : Magnitute Calculating for acc and Gyro")
            self.data_x.columns , columns_groups = self.regroup_and_reindex_all_cols(self.data_x.set_index('sub_id').copy())

            temp_columns = list(self.data_x.columns[1:-1])
            for cols in columns_groups:
                if len(cols)== 3:
                    col1, col2, col3 = cols
                    col = "_".join(col1.split("_")[:-1])+"_mag"
                    temp_columns.append(col)
                    self.data_x[col] = mag_3_signals(np.array(self.data_x[col1]),
                                                     np.array(self.data_x[col2]),
                                                     np.array(self.data_x[col3]))
            self.data_x = self.data_x[["sub_id"]+temp_columns+["sub"] ]
        



        # ======================= Generate the Sliding window  =================================
        self.slidingwindows = self.get_the_sliding_index(self.data_x.copy(), self.data_y.copy())


        # ----------------------- TODO ----------------------------------------
        if self.representation_type in ["freq", "time_freq"]:
            print("------------freq representation is needed -----------------")
            assert self.freq_save_path is not None
            # train/test is different by the sliding step.
            self.genarate_spectrogram(flag = "train")
            self.genarate_spectrogram(flag = "test")

            if self.load_all:
                print("-------load all freq DATA --------")
                self.data_freq = {}
                for file in tqdm(self.train_freq_file_name):
                    with open(os.path.join(self.train_freq_path,"{}.pickle".format(file)), 'rb') as handle:
                        sample_x = pickle.load(handle)
                    self.data_freq[file] = sample_x


        if self.exp_mode == "LOCV":
            self.num_of_cv = len(self.LOCV_keys)
            
            self.index_of_cv = 0
        elif self.exp_mode == "SOLO":
            self.num_of_cv = 1
        else:
            self.num_of_cv = 1

    def update_train_val_test_keys(self):
        """
        It should be called at the begin of each iteration
        it will update:
        1. train_window_index
        2. vali_window_index
        3. test_window_index
        it will also:
        normalize the data , because each iteration uses different training data
        calculate the weights of each class
        """
        if self.exp_mode in [ "Given","LOCV" ]:
            if self.exp_mode == "LOCV":
                print("-----------------------------------------------------------------------------")
                print("Leave one Out Experiment : The {} Part as the test".format(self.index_of_cv+1))

                self.test_keys      = self.LOCV_keys[self.index_of_cv]
                self.train_keys     = [key for key in self.all_keys if key not in self.test_keys]
                self.vali_keys      = random.sample(self.train_keys, int( self.valid_rate* len(self.train_keys) ) if int( self.valid_rate* len(self.train_keys) ) >=1 else 1 ) 
                # update the index_of_cv for the next iteration
                self.index_of_cv    = self.index_of_cv + 1
            elif self.exp_mode == "SOLO":
                self.test_keys      = self.train_keys = self.vali_keys = [1]
            else:
                self.index_of_cv    = 1
                self.vali_keys      = random.sample( self.train_keys, int( self.valid_rate* len(self.train_keys) ) ) if self.vali_keys == [] else self.vali_keys
            self.train_keys         = list(set(self.train_keys) - set(self.vali_keys))
            print(f'Selected train Subjects:{self.train_keys}\nSelected Valid Subjects:{self.vali_keys}\nSelected Test  Subjects:{self.test_keys}')

            #Normalization the data
            if self.datanorm_type is not None:
                train_vali_x = pd.DataFrame()
                train_vali_keys = self.train_keys + self.vali_keys
                for sub in train_vali_keys:
                    temp    = self.data_x[self.data_x[self.split_tag]==sub]
                    train_vali_x = pd.concat([train_vali_x,temp])
                    
                test_x = pd.DataFrame()
                for sub in self.test_keys:
                    temp    = self.data_x[self.data_x[self.split_tag]==sub]
                    test_x  = pd.concat([test_x,temp])

            
                train_vali_x, test_x    = self.normalization(train_vali_x, test_x)

                self.normalized_data_x  = pd.concat([train_vali_x,test_x])
                self.normalized_data_x.sort_index(inplace=True)
            else:
                self.normalized_data_x = self.data_x.copy()
            # self.normalized_data_x = self.data_x.copy()

            # 根据test的keys  筛选出 window的第一个element有哪些
            all_test_keys = []
            all_vali_keys = []
            if self.split_tag == "sub":
                for sub in self.test_keys:
                    all_test_keys.extend(self.sub_ids_of_each_sub[sub])
                for sub in self.vali_keys:
                    all_vali_keys.extend(self.sub_ids_of_each_sub[sub])
            else:
                all_test_keys = self.test_keys.copy()
                all_vali_keys = self.vali_keys.copy()

            # -----------------train_window_index---------------------
            train_file_name = os.path.join(self.window_save_path,
                                           "{}_droptrans_{}_windowsize_{}_{}_train_ID_{}.pickle".format(self.data_name, 
                                                                                                        self.drop_transition,
                                                                                                        self.exp_mode,
                                                                                                        self.windowsize, 
                                                                                                        self.index_of_cv-1))
            # -----------------valid_window_index---------------------
            valid_file_name = os.path.join(self.window_save_path,
                                           "{}_droptrans_{}_windowsize_{}_{}_valid_ID_{}.pickle".format(self.data_name, 
                                                                                                        self.drop_transition,
                                                                                                        self.exp_mode,
                                                                                                        self.windowsize, 
                                                                                                        self.index_of_cv-1))
            # -----------------test_window_index---------------------
            test_file_name = os.path.join(self.window_save_path,
                                          "{}_droptrans_{}_windowsize_{}_{}_test_ID_{}.pickle".format(self.data_name, 
                                                                                                      self.drop_transition,
                                                                                                      self.exp_mode,
                                                                                                      self.windowsize, 
                                                                                                      self.index_of_cv-1))

            if os.path.exists(train_file_name) and os.path.exists(valid_file_name) and os.path.exists(test_file_name):
                
                with open(train_file_name, 'rb') as handle:
                    self.train_window_index = pickle.load(handle)
                with open(valid_file_name, 'rb') as handle:
                    self.valid_window_index = pickle.load(handle)
                with open(test_file_name, 'rb') as handle:
                    self.test_window_index = pickle.load(handle)
            else:
                self.train_window_index = []
                self.valid_window_index = []
                self.test_window_index  = []
                for index, window in enumerate(self.slidingwindows):
                    sub_id = window[0]
                    if sub_id in all_test_keys:
                        self.test_window_index.append(index)
                    elif sub_id in all_vali_keys:
                        self.valid_window_index.append(index)
                    else:
                        self.train_window_index.append(index)

                with open(train_file_name, 'wb') as handle:
                    pickle.dump(self.train_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(valid_file_name, 'wb') as handle:
                    pickle.dump(self.valid_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(test_file_name, 'wb') as handle:
                    pickle.dump(self.test_window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            raise NotImplementedError

        self.act_weights = self.update_classes_weight()


    def update_classes_weight(self):
        class_transform = {x: i for i, x in enumerate(self.no_drop_activites)}

        y_of_all_windows  = []
        # get all labels of all windows
        for index in self.train_window_index:

            start_index = self.slidingwindows[index][1]
            end_index = self.slidingwindows[index][2]

            y_of_all_windows.append(class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]])
        y_of_all_windows = np.array(y_of_all_windows)


        target_count   = np.array([np.sum(y_of_all_windows == label) for label in set(y_of_all_windows)])
        weight_target  = 1.0 / target_count
        weight_samples = np.array([weight_target[t] for t in y_of_all_windows])
        weight_samples = torch.from_numpy(weight_samples)
        weight_samples = weight_samples.double()


        print("[-] Target sampling weights: ", weight_target)


        return weight_samples

    def load_all_the_data(self, root_path):
        raise NotImplementedError


    def Sensor_data_noise_grav_filtering(self, df):
        """
        df : sensor_1, sensor_2, sub
           index=sub_id
        """
        all_columns = list(df.columns)[:-1]
        #rest_columns = list(set(all_columns) - set(self.col_names))

        filtered_data = []
        for sub_id in df.index.unique():
            temp = df.loc[sub_id,all_columns]
            filtered_temp = pd.DataFrame()

            for col in temp.columns:
                t_signal=np.array(temp[col]) # copie the signal values in 1D numpy array

                if 'acc' in col and "diff" not in col: 
                    # the 2nd output DC_component is the gravity_acc
                    # The 3rd one is the body_component which in this case the body_acc
                    grav_acc,body_acc=components_selection_one_signal(t_signal,freq1,freq2,self.freq) # apply components selection

                    filtered_temp[col]=body_acc
                    filtered_temp['grav'+col]= grav_acc
        
                elif 'gyro' in col and "diff" not in col: 
            
                    # The 3rd output of components_selection is the body_component which in this case the body_gyro component
                    _,        body_gyro=components_selection_one_signal(t_signal,freq1,freq2,self.freq)  # apply components selection
                    filtered_temp[col]=body_gyro # t_body_acc storing with the appropriate axis selected 
                else: 
            
                    filtered_temp[col] = t_signal

            #filtered_temp = filtered_temp[sorted(list(filtered_temp.columns))]
            filtered_temp.index = temp.index
            filtered_data.append(filtered_temp)

        filtered_data = pd.concat(filtered_data)
        #filtered_data = pd.concat([df.loc[:,rest_columns], filtered_data], axis=1)
        #filtered_data = filtered_data[sorted(list(filtered_data.columns))]
        filtered_data = pd.concat([filtered_data, df.iloc[:,-1]], axis=1)

        return filtered_data.reset_index()




    def differencing(self, df):
        # columns = [, "acc_x"..."acc_y", "sub"], index is sub_id
        # define the name for differenced columns
        all_columns = list(df.columns)[:-1]
        rest_columns = list(set(all_columns) - set(self.col_names))

        columns = ["diff_"+i for i in self.col_names]

        # The original data has been divided into segments by sub_id: a segment belongs to a same user 
        # There is no continuity between different segments, so diffrecne is only done within each segment

        # ALL data Diff
        diff_data = []
        for id in df.index.unique():
            diff_data.append(df.loc[id,self.col_names].diff())
        diff_data = pd.concat(diff_data)
        diff_data.columns = columns
        diff_data.fillna(method ="backfill",inplace=True)

        data = pd.concat([df.iloc[:,:-1],  diff_data], axis=1)
        #data = data[sorted(list(data.columns))]
        data = pd.concat([data, df.iloc[:,-1]], axis=1)

        return data.reset_index()

    def normalization(self, train_vali, test=None):
        train_vali_sensors  = train_vali.iloc[:,1:-1]
        self.normalizer     = Normalizer(self.datanorm_type)
        self.normalizer.fit(train_vali_sensors)
        train_vali_sensors = self.normalizer.normalize(train_vali_sensors)
        train_vali_sensors = pd.concat([train_vali.iloc[:,0],train_vali_sensors,train_vali.iloc[:,-1]], axis=1)
        if test is None:
            return train_vali_sensors
        else:
            test_sensors  = test.iloc[:,1:-1]
            test_sensors  = self.normalizer.normalize(test_sensors)
            test_sensors  =  pd.concat([test.iloc[:,0],test_sensors,test.iloc[:,-1]], axis=1)
            return train_vali_sensors, test_sensors

    def get_the_sliding_index(self, data_x, data_y , flag = "all"):
        """
        Because of the large amount of data, it is not necessary to store all the contents of the slidingwindow, 
        but only to access the index of the slidingwindow
        Each window consists of three parts: sub_ID , start_index , end_index
        The sub_ID ist used for train test split, if the subject train test split is applied
        """
        if os.path.exists(os.path.join(self.window_save_path,
                                       "{}_{}_drop_trans_{}_windowsize{}.pickle".format(self.data_name, 
                                                                                        flag, 
                                                                                        self.drop_transition,
                                                                                        self.windowsize))):
            print("-----------------------Sliding file are generated -----------------------")
            with open(os.path.join(self.window_save_path,
                                   "{}_{}_drop_trans_{}_windowsize{}.pickle".format(self.data_name, 
                                                                                    flag, 
                                                                                    self.drop_transition,
                                                                                    self.windowsize)), 'rb') as handle:
                window_index = pickle.load(handle)
        else:
            print("----------------------- Get the Sliding Window -------------------")

            data_y = data_y.reset_index()
            data_x["activity_id"] = data_y["activity_id"]

            if self.drop_transition:
                data_x['act_block'] = ((data_x['activity_id'].shift(1) != data_x['activity_id']) | (data_x['sub_id'].shift(1) != data_x['sub_id'])).astype(int).cumsum()
            else:
                data_x['act_block'] = (data_x['sub_id'].shift(1) != data_x['sub_id']).astype(int).cumsum()

            freq         = self.freq   
            windowsize   = self.windowsize

            if self.overlap_rate:
                displacement = int( (1 - self.overlap_rate) * self.windowsize)
            else:
                displacement = int( 0.5 * self.windowsize)
                
            window_index = []
            for index in data_x.act_block.unique():

                temp_df = data_x[data_x["act_block"]==index]
                assert len(temp_df["sub_id"].unique()) == 1
                sub_id = temp_df["sub_id"].unique()[0]
                start = temp_df.index[0]# + drop_for_augmentation 
                end   = start+windowsize

                while end <= temp_df.index[-1]+1:# + drop_for_augmentation :

                    if temp_df.loc[start:end-1,"activity_id"].mode().loc[0] not in self.drop_activities:
                        window_index.append([sub_id, start, end])

                    start = start + displacement
                    end   = start + windowsize

            with open(os.path.join(self.window_save_path,"{}_{}_drop_trans_{}_windowsize{}.pickle".format(self.data_name, flag, self.drop_transition,windowsize)), 'wb') as handle:
                pickle.dump(window_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # sub_id, start, end
        return window_index


    def genarate_spectrogram(self, flag="train"):
        save_path = os.path.join(self.freq_save_path,self.data_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        displacement = int((1-self.overlap_rate) * self.windowsize)
        slidingwindows = self.slidingwindows
        self.train_freq_path = os.path.join(save_path,"diff_{}_window_{}_step_{}_drop_trans_{}".format(self.difference, self.windowsize,displacement, self.drop_transition))
        freq_path = self.train_freq_path





        if os.path.exists(freq_path):
            print("----------------------- file are generated -----------------------")
            if flag == "train":
                with open(os.path.join(freq_path,"freq_file_name.pickle"), 'rb') as handle:
                    self.train_freq_file_name = pickle.load(handle)
            else:
                with open(os.path.join(freq_path,"freq_file_name.pickle"), 'rb') as handle:
                    self.test_freq_file_name = pickle.load(handle)

        else:
            print("----------------------- spetrogram generating for {} -----------------------".format(flag))
            os.mkdir(freq_path)

            scales1 = np.arange(1, self.freq + 1) 

            totalscal = self.freq 
            fc = pywt.central_frequency(self.wavename)#计算小波函数的中心频率
            cparam = 2 * fc * totalscal  #常数c
            scales2 = cparam/np.arange(totalscal,0,-1) #为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）

            if self.windowsize >= 60 and "synthetic" not in self.data_name:
                l_scale = 2
            else:
                l_scale = 1

            if self.freq >=40 and "synthetic" not in self.data_name:
                f_scale = 2
            else:
                f_scale = 1

            if f_scale>1 or l_scale>1:
                resize_flag = True
            else:
                resize_flag = False
            

            freq_file_name = []

            temp_data = self.normalization(self.data_x.copy())
            for window in slidingwindows:
                sub_id = window[0]
                start_index = window[1]
                end_index = window[2]
	
                name = "{}_{}_{}".format(sub_id,start_index,end_index)
                freq_file_name.append(name)

                sample_x = temp_data.iloc[start_index:end_index,1:-1].values
                scalogram = []

                for j in range(sample_x.shape[1]):
                    if self.difference and j>= int(sample_x.shape[1]/2):
                        [cwtmatr, frequencies] = pywt.cwt(sample_x[:,j],   scales2,  self.wavename, sampling_period = 1.0/self.freq)#连续小波变换模块
                    else:
                        [cwtmatr, frequencies] = pywt.cwt(sample_x[:,j],   scales1,  self.wavename, sampling_period = 1.0/self.freq)#连续小波变换模块
                    if resize_flag:
                        cwtmatr = resize(cwtmatr, (int(self.freq/f_scale), int(self.windowsize/l_scale)), mode = 'constant')
                    scalogram.append(cwtmatr)

                scalogram = np.stack(scalogram)

                with open(os.path.join(freq_path,"{}.pickle".format(name)), 'wb') as handle:
                    pickle.dump(scalogram, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(freq_path,"freq_file_name.pickle"), 'wb') as handle:
                pickle.dump(freq_file_name, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if flag == "train":
                self.train_freq_file_name = freq_file_name
            else:
                self.test_freq_file_name = freq_file_name

    def Sensor_filter_acoording_to_pos_and_type(self, select, filter, all_col_names, filtertype):
        """
        select  (list or None): What location should be chosen
        filter  (list or None): whether all sensors can be filtered 
        all_col_names     (list)  : Columns currently available for Filtering
        """ 
        if select is not None:
            if filter is None:
                raise Exception('This dataset cannot be selected by sensor {}!'.format(filtertype))
            else:
                col_names = []
                for col in all_col_names:
                    selected = False
                    for one_select in select:
                        assert one_select in filter
                        if one_select in col:
                            selected = True
                    if selected:
                        col_names.append(col)
                return col_names
        else:
            return None

    def regroup_and_reindex_all_cols(self, df):
        columns = df.columns[:-1]
        # big gourps
        groups = {}
        for col in columns:
            index = col.split("_")[-1]
            if index in groups.keys():
                groups[index].append(col)
            else:
                groups[index] = [col]
        # sub_groups
        index = 1
        columns_mapping = {}
        columns_groups = []
        for key in groups.keys():
            cols = groups[key]
            cols_set = []
            for col in cols:
                cols_set.append(col.split("_")[0])
            cols_set = set(cols_set)

            for col_begin in cols_set:
                sub_groups= []
                for col in cols:
                    if col.split("_")[0]==col_begin:
                        columns_mapping[col] = "_".join(col.split("_")[:-1])+"_"+str(index)
                        sub_groups.append("_".join(col.split("_")[:-1])+"_"+str(index))
                index= index+1
                if col_begin in ["acc","gyro","gravacc"]:
                    columns_groups.append(sub_groups)
        columns = ["sub_id"]+[columns_mapping[col] for col in df.columns[:-1]] + ["sub"]
        return columns,columns_groups
    
    def create_cross_subject_dataset(self,args):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
