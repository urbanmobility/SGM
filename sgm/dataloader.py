
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
pd.set_option('display.max_column', None)



'''
config:
    data_path: cleaned data path, default='../data1/cleaned/'
    data_name: name of dataset, default='kdd99'
    feature_mode: choice to divide feature, default=0
    anomal_rate: anomal data rate in train data, default='default'
    boundary: number of unique values to determine categorical feature, default=10
    inject_noise: whether inject noise, default='False'
    cont_rate: contaminated data rate in normal train data, default=0.01
'''



def LoadTabularData(params):
    '''Load tabular data'''
    
    target = 'labels'
    filepath = f'{params.data_path}{params.data_name}.csv'
    data = pd.read_csv(filepath)
    start = time.time()

    # divide X and y
    labels = data[target].values.copy()     # array
    data.drop(target, axis=1, inplace=True)

    # divide numerical and categorical
    cols = {}  
    dtypes = data.dtypes
    cols['cat'] = dtypes[dtypes == 'object'].index.to_list()
    cols['num'] = list(set(list(data.columns)).difference(set(cols['cat'])))
    # regard part numerical data to category
#     if params.feature_mode == 0:  
#         cols['num'] = list(set(list(data.columns)).difference(set(cols['cat'])))
#     elif params.feature_mode == 1:
#         for i in data.columns:
#             if len(data[i].unique()) <= params.boundary:
#                 cols['cat'].append(i)
#         cols['cat'] = set(cols['cat'])
#         cols['num'] = list(set(list(data.columns)).difference(set(cols['cat'])))

    anomal_rate_o = len(labels[labels==1]) / len(labels)

    info  = f'==== Dataset: {params.data_name}\n'
    info += '@Original Info: (without labels)\n'
    info += f'Original shape: data={data.shape}, num={len(cols["num"])}, cat={len(cols["cat"])} \n'
    info += f'Original anomal rate is: {anomal_rate_o*100:.2f}% \n'
    info += f'Normal data is {len(labels[labels==0])}; Anomaly data is {len(labels[labels==1])} \n'
    
    
    
    # Divide data to train and test 
    train_idx, test_idx, y_train, y_test = \
                            train_test_split(range(data.shape[0]), labels, test_size=0.2, \
                                            random_state=params.seed, stratify = labels)
    # store test data
#     # wait to add
#     if params.save_test_data:
#         data.loc[test_idx,:].to_csv(f'{params.output_path}test_x.csv', index=False)
#         np.savetxt(f'{params.output_path}test_y.txt', labels[test_idx])
    
    # Code
    cat_num = 0
    if len(cols['cat']) > 0:
        x_cat = pd.get_dummies(data[cols['cat']].astype('object')).values
        cat_num = x_cat.shape[1]

    # Scale/standardization
    num_num = 0
    if len(cols['num']) > 0:
        x_num = scale(data[cols['num']])
        num_num = x_num.shape[1]

    # Concate num and cat
    if num_num > 0 and cat_num > 0:
        x = np.hstack((x_num, x_cat))
    elif num_num > 0:
        x = x_num
    elif cat_num > 0:
        x = x_cat
        
    x_train = x[train_idx]
    x_test = x[test_idx]
        
    # Extract validate set from train set
    x_train, x_val, y_train, y_val = \
                                train_test_split(x_train, y_train, test_size=0.25, \
                                        random_state=params.seed, stratify = y_train)

    
    info += '@Preprocess Info: \n'
    info += f'== Feature Process == \n'
    info += f'Time cost is {time.time() - start:.2f} seconds\n'
    info += f'Data shape: data={x.shape}, num={num_num}, cat={cat_num} \n'
    info += f'Train/Validate/test data shape is: {x_train.shape} / {x_val.shape} / {x_test.shape} \n'

    # Inject noise to train data:
    if params.inject_noise:
        start_inject = time.time()
        idx = np.where(y_train == 0)[0]
        dim = x_train.shape[1]
        normal_num = len(y_train[y_train==0])
        noise_num = int(normal_num * params.cont_rate / (1 - params.cont_rate))
        noise = np.empty((noise_num, dim))
        swap_rate = 0.05
        swap_feature_num = int(dim * swap_rate)
        if swap_feature_num < 1:
            swap_feature_num = 1
        for i in np.arange(noise_num):
            swap_idx = np.random.choice(idx, 2, replace=False)
            swap_feature = np.random.choice(dim, swap_feature_num, replace=False)
            noise[i] = x_train[swap_idx[0]].copy()
            noise[i, swap_feature] = x_train[swap_idx[1], swap_feature]

        x_train = np.append(x_train, noise, axis=0)
        y_train = np.append(y_train, np.zeros((noise_num,)))

        info += f'== Inject Noise == \n'
        info += f'Time cost is {time.time() - start_inject:.2f} seconds\n'
        info += f'Noise inject number is {noise_num}, cont_rate={params.cont_rate}\n'
        info += f'Train data: shape={x_train.shape}, anomal rate={len(y_train[y_train==1])/len(y_train)*100:.2f}%\n'
        info += f'Val data: shape={x_val.shape}, anomal rate={len(y_val[y_val==1])/len(y_val)*100:.2f}%\n'
        info += f'Test data: shape={x_test.shape}, anomal_rate={len(y_test[y_test==1])/len(y_test)*100:.2f}%\n'
    
    # Generate data with anomaly rate
    anomal_rate = params.anomal_rate
    if anomal_rate != 'default':
        anomal_rate = eval(anomal_rate)
        start_adjust = time.time()
        if anomal_rate >= anomal_rate_o:
            idx = np.where(y_train == 0)[0]
            anomal_num = len(y_train[y_train==1])
            normal_num = int(anomal_num * (1 - anomal_rate) / anomal_rate)
            delta_num = len(y_train[y_train==0]) - normal_num
            delta_idx = np.random.choice(idx, delta_num, replace=False)
        else:
            idx = np.where(y_train == 1)[0]
            normal_num = len(y_train[y_train==0])
            anomal_num = int(normal_num * anomal_rate / (1 - anomal_rate))
            delta_num = len(y_train[y_train==1]) - anomal_num
            delta_idx = np.random.choice(idx, delta_num, replace=False)

        x_train = np.delete(x_train, delta_idx, axis=0) 
        y_train = np.delete(y_train, delta_idx, axis=0)

        info += f'== Adjust anomaly rate == \n'
        info += f'Time cost is {time.time() - start_adjust:.2f} seconds\n'
        info += f'Train data: shape={x_train.shape}, anomaly rate={len(y_train[y_train==1])/len(y_train)*100:.2f}%\n'
        info += f'Test data: shape={x_test.shape}, anomaly rate={len(y_test[y_test==1])/len(y_test)*100:.2f}%\n'
    else:
        params.anomal_rate = len(y_train[y_train==1]) / len(y_train)


    info += f'== Finished == \n'
    info += f'Total time cost is {time.time()-start:.2f} seconds\n'

    if params.verbose:
        print(info)
    
    return x_train, y_train, x_val, y_val, x_test, y_test






import pickle

class LoadDocumentData(object):
    '''Load Document Data'''
    
    def __init__(self, params):
    
        # data name: reuters, 20news
        filepath = f'{params.data_path}{params.data_name}.data'

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.seed = params.seed
        self.x_origin = data["X"]
        self.y_origin = data["y"]
        if isinstance(params.anomal_rate, str):
            self.anomal_rate = eval(params.anomal_rate)
        elif isinstance(params.anomal_rate, float):
            self.anomal_rate = params.anomal_rate
        self.class_num = len(list(set(self.y_origin)))
        
        if params.verbose:
            print(f'==== Load data {params.data_name} ====')
            print(f'data shape: {self.x_origin.shape}')
    
    def preprocess(self, anomal_class):
        '''
        Choose one class to be anomaly
        '''
        y = (np.array(self.y_origin) != anomal_class).astype(int)
        anomal_idx = np.where(y == 1)[0]
        
        rm_num  = len(anomal_idx) - int(360 * self.anomal_rate / (1 - self.anomal_rate))
        anomal_idx_rm = np.random.choice(anomal_idx, rm_num, replace=False)
        x = np.delete(self.x_origin, anomal_idx_rm, axis=0) 
        y = np.delete(y, anomal_idx_rm, axis=0) 

        print(f'Anomay rate is {len(y[y==1])} / {len(y)} = {len(y[y==1])/len(y):.4f}')
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, \
                                                            random_state=self.seed, stratify = y) 
        
        print(f'Train data shape: {x_train.shape}, Test data shape: {x_test.shape}')
        
        return x_train, x_test, y_train, y_test
        
        
        
def LoadImageData(params):
    '''Lad image data'''
    
    filepath = f'{params.data_path}{params.data_name}_'
    
    x_origin = np.loadtxt(f"{filepath}data.txt", delimiter=",")
    y_origin = np.loadtxt(f"{filepath}y.txt", delimiter=",")
    
    y = (y_origin != 4.0).astype(int)
    params.anomal_rate = len(y[y==1])/len(y)
    print(f'Anomay rate is {len(y[y==1])} / {len(y)} = {params.anomal_rate:.4f}')
    
    
    x_train, x_test, y_train, y_test = train_test_split(x_origin, y, test_size=0.2, \
                                                            random_state=params.seed, stratify = y) 
    
    print(f'Train data shape: {x_train.shape}, Test data shape: {x_test.shape}')
    
    return x_train, x_test, y_train, y_test