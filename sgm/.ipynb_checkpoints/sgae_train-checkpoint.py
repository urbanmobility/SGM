import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
# from sklearn.metrics import precision_recall_fscore_support as prf
# from scipy.stats.mstats import ks_2samp as kstest

import pandas as pd
from tqdm import tqdm


from sgae import SGAE
from dataloader import LoadDocumentData, LoadImageData, LoadTabularData


def train_image(params):
    
    # Load data
    x_train, x_test, y_train, y_test = LoadImageData(params)
    x_train_device = torch.FloatTensor(x_train).to(params.device)
    x_test = torch.FloatTensor(x_test).to(params.device)
    
    # Experiment settings
    nb_batch = int(x_train.shape[0] / params.batch_size) 
    auc = np.zeros(params.run_num)
    ap = np.zeros(params.run_num)
    
    # Start Train
    for run_idx in tqdm(range(params.run_num)):
        start_time = time.time()
        
        model = SGAE(x_train.shape[1], params.hidden_dim).to(params.device)
        optimizer = optim.Adam(model.parameters(), lr=params.lr)

        if params.verbose and run_idx == 0:
            print(model)

        # One run
        for epoch in range(params.epochs): 
            epoch_time_start = time.time() 
            # train
            model.train()
            
            # calculate norm thresh
            _, dec_train, _ = model(x_train_device)
            norm = calculate_norm(x_train_device, dec_train)
            norm_thresh = np.percentile(norm, params.epsilon)

            loss = 0
            recon_error = 0
            dist_error = 0

            for i_batch, data in enumerate(data_batch(x_train, params.batch_size)):
                if i_batch > nb_batch:
                    break                  

                data = data.to(params.device) 
                scores, x_dec, _ = model(data)
                anomal_flag = recog_anomal(data, x_dec, norm_thresh)
                anomal_flag = torch.tensor(anomal_flag).to(params.device)

                loss_batch, recon_error_batch, dist_error_batch = model.loss_function(data, x_dec, scores, \
                                                anomal_flag, params)
                loss += loss_batch.item()
                recon_error += recon_error_batch.item()
                dist_error += dist_error_batch.item()

                optimizer.zero_grad()
                loss_batch.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
                
            epoch_time = time.time() - epoch_time_start 

            # test
            model.eval()
            scores, _, _ = model(x_test)
            scores = scores.detach().cpu().numpy()
            auc[run_idx] = roc_auc_score(y_test, scores)
            ap[run_idx] = average_precision_score(y_test, scores)

            if params.verbose:
                if (epoch + 1) % params.print_step == 0 or epoch == 0:
                    print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                            f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --DE:{dist_error:.3f}, --DE_r:{dist_error*params.lam_dist:.3f},'+\
                            f'--AUC:{auc[run_idx]:.3f}')            
                    
            # Early Stop
            if params.early_stop:
                scores, _, _ = model(x_train_device)
                scores = scores.detach().cpu().numpy()   
                if np.mean(scores) > params.a / 2:
                    print(f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx]:.3f}')
                    break

        
        print(f'This run finished, AUC={auc[run_idx]:.3f}, AP={ap[run_idx]:.3f}')
    
    print(f'Train Finshed, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP={np.mean(ap):.3f}({np.std(ap):.3f})')
    
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}




def train_document(params):
    
    # Load data
    dataloader = LoadDocumentData(params)
    
    # Experiment settings
    auc = np.zeros((params.run_num, dataloader.class_num))
    ap = np.zeros((params.run_num, dataloader.class_num))
    
    # Start Train
    for run_idx in tqdm(range(params.run_num)):
        start_time = time.time()
        
        # Iterate for normal class
        for normal_idx in range(dataloader.class_num):
            x_train, x_test, y_train, y_test = dataloader.preprocess(normal_idx)
            x_train_device = torch.FloatTensor(x_train).to(params.device)
            x_test = torch.FloatTensor(x_test).to(params.device)
            nb_batch = int(x_train.shape[0] / params.batch_size) 
            
            model = SGAE(x_train.shape[1], params.hidden_dim).to(params.device)
            optimizer = optim.Adam(model.parameters(), lr=params.lr)

            if params.verbose and normal_idx == 0 and run_idx == 0:
                print(model)


            # One run
            for epoch in range(params.epochs): 
                epoch_time_start = time.time() 
                # train
                model.train()

                # calculate norm thresh
                _, dec_train, _ = model(x_train_device)
                norm = calculate_norm(x_train_device, dec_train)
                norm_thresh = np.percentile(norm, params.epsilon)

                loss = 0
                recon_error = 0
                dist_error = 0

                for i_batch, data in enumerate(data_batch(x_train, params.batch_size)):
                    if i_batch > nb_batch:
                        break                  

                    data = data.to(params.device) 
                    scores, x_dec, _ = model(data)
                    anomal_flag = recog_anomal(data, x_dec, norm_thresh)
                    anomal_flag = torch.tensor(anomal_flag).to(params.device)

                    loss_batch, recon_error_batch, dist_error_batch = model.loss_function(data, x_dec, scores, \
                                                    anomal_flag, params)
                    loss += loss_batch.item()
                    recon_error += recon_error_batch.item()
                    dist_error += dist_error_batch.item()

                    optimizer.zero_grad()
                    loss_batch.backward()
                    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                    optimizer.step()


                epoch_time = time.time() - epoch_time_start 

                # test
                model.eval()
                scores, _, _ = model(x_test)
                scores = scores.detach().cpu().numpy()
                auc[run_idx][normal_idx] = roc_auc_score(y_test, scores)
                ap[run_idx][normal_idx] = average_precision_score(y_test, scores)

                if params.verbose:
                    if (epoch + 1) % params.print_step == 0 or epoch == 0:
                        print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                                f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --DE:{dist_error:.3f}, --DE_r:{dist_error*params.lam_dist:.3f},'+\
                                f'--AUC:{auc[run_idx][normal_idx]:.3f}')            
                # Early Stop
                if params.early_stop:
                    scores, _, _ = model(x_train_device)
                    scores = scores.detach().cpu().numpy()   
                    if np.mean(scores) > params.a / 2:
                        print(f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx][normal_idx]:.3f}')
                        break
        
        print(f'This run finished, AUC={np.mean(auc[run_idx]):.3f}, AP={np.mean(ap[run_idx]):.3f}')
        print(f'RE/DE = {recon_error/dist_error:.2f}')
        
        # RUN JUMP
        if np.mean(auc[:run_idx+1]) < params.stop_train and run_idx >= 3:
            print(f'Stop train, this parameter aborted. Mean AUC={np.mean(auc[:run_idx+1])}')
            print(f'AUC: {auc}')
            break
    
    print(f'Train Finished, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP={np.mean(ap):.3f}({np.std(ap):.3f})')
    
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}


def train_tabular(params):
    
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = LoadTabularData(params)
    x_train_device = torch.FloatTensor(x_train).to(params.device)
    x_test = torch.FloatTensor(x_test).to(params.device)
    
    # Experiment settings
    nb_batch = int(x_train.shape[0] / params.batch_size) 
    auc = np.zeros(params.run_num)
    ap = np.zeros(params.run_num)
    
    # Start Train
    for run_idx in tqdm(range(params.run_num)):
        start_time = time.time()
        
        model = SGAE(x_train.shape[1], params.hidden_dim).to(params.device)
        optimizer = optim.Adam(model.parameters(), lr=params.lr)

        if params.verbose and run_idx == 0:
            print(model)

        # One run
        for epoch in range(params.epochs): 
            epoch_time_start = time.time() 
            # train
            model.train()
            
            # calculate norm thresh
            _, dec_train, _ = model(x_train_device)
            norm = calculate_norm(x_train_device, dec_train)
            norm_thresh = np.percentile(norm, params.epsilon)

            loss = 0
            recon_error = 0
            dist_error = 0

            for i_batch, data in enumerate(data_batch(x_train, params.batch_size)):
                if i_batch > nb_batch:
                    break                  

                data = data.to(params.device) 
                scores, x_dec, _ = model(data)
                anomal_flag = recog_anomal(data, x_dec, norm_thresh)
                anomal_flag = torch.tensor(anomal_flag).to(params.device)

                loss_batch, recon_error_batch, dist_error_batch = model.loss_function(data, x_dec, scores, \
                                                anomal_flag, params)
                loss += loss_batch.item()
                recon_error += recon_error_batch.item()
                dist_error += dist_error_batch.item()

                optimizer.zero_grad()
                loss_batch.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
                
            epoch_time = time.time() - epoch_time_start 

            # test
            model.eval()
            scores, _, _ = model(x_test)
            scores = scores.detach().cpu().numpy()
            auc[run_idx] = roc_auc_score(y_test, scores)
            ap[run_idx] = average_precision_score(y_test, scores)

            if params.verbose:
                if (epoch + 1) % params.print_step == 0 or epoch == 0:
                    print(f'Epoch num:[{epoch+1}/{params.epochs}], Time:{epoch_time:.3f} ' +\
                            f'--Loss:{loss:.3f}, --RE:{recon_error:.3f}, --DE:{dist_error:.3f}, --DE_r:{dist_error*params.lam_dist:.3f},'+\
                            f'--AUC:{auc[run_idx]:.3f}')            
                    
            # Early Stop
            if params.early_stop:
                scores, _, _ = model(x_train_device)
                scores = scores.detach().cpu().numpy()   
                if np.mean(scores) > params.a / 2:
                    print(f'Early Stop at Epoch={epoch+1}, AUC={auc[run_idx]:.3f}')
                    break

        
        print(f'This run finished, AUC={auc[run_idx]:.3f}, AP={ap[run_idx]:.3f}')
        
        # RUN JUMP
        if run_idx > 5 and np.mean(auc[:run_idx]) < 0.5:
            print('RUN JUMP')
            print(f'Average AUC is : {np.mean(auc[:run_idx]):.3f}')
            print(f'AUC is : {auc}')
            break
    
    print(f'Train Finished, AUC={np.mean(auc):.3f}({np.std(auc):.3f}), AP={np.mean(ap):.3f}({np.std(ap):.3f})')
    
    return {'AUC': f'{np.mean(auc):.3f}({np.std(auc):.3f})', 'AP': f'{np.mean(ap):.3f}({np.std(ap):.3f})'}


def data_batch(x_train, batch_size):
    ''' Generate data batch, return tensor.
    '''
    n = len(x_train)
    while(1):
        idx = np.random.choice(n, batch_size, replace=False)
        data = x_train[idx]
        data = torch.FloatTensor(data)
        yield data
        
def recog_anomal(data, x_dec, thresh):
    ''' Recognize anomaly
    '''
    norm = calculate_norm(data, x_dec)
    anomal_flag = norm.copy()
    anomal_flag[norm < thresh] = 0
    anomal_flag[norm >= thresh] = 1
    return anomal_flag

def calculate_norm(data, x_dec):
    ''' Calculate l2 norm
    '''
    delta = (data - x_dec).detach().cpu().numpy()
    norm = np.linalg.norm(delta, ord=2, axis=1)
    return norm

        
