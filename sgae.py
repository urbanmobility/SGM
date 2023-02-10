
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SGAE(nn.Module):
    def __init__(self, input_shape, hidden_dim_input, layer_list=[20, 40, 80, 256, 1024]):
        '''
        hidden_dim_input: str
        '''
        super(SGAE, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        if hidden_dim_input == 'auto':
            hidden_dim = []
            if input_shape < 20:
                hidden_dim = [10, input_shape]
            else:
                for idx in range(len(layer_list)):
                    if layer_list[idx] < input_shape:
                        hidden_dim.append(layer_list[idx])
                    elif input_shape - layer_list[idx-1] < layer_list[idx-1] * 0.1:
                        hidden_dim[-1] = input_shape    
                    else:
                        hidden_dim.append(input_shape)
                if input_shape > layer_list[-1] * 1.5:
                    hidden_dim.append(input_shape)
                else:
                    hidden_dim[-1] = input_shape
            hidden_dim.reverse()
            print(f'Model hidden dim is {hidden_dim}')
        else:
            hidden_dim_input = eval(hidden_dim_input)
            hidden_dim = hidden_dim_input[:]
            hidden_dim.insert(0, input_shape) 
        # encoder
        for idx in range(len(hidden_dim) - 1 ):
            self.encoder.add_module(f'en_lr{idx+1}', nn.Linear(hidden_dim[idx], hidden_dim[idx+1]))
            self.encoder.add_module(f'en_relu{idx+1}', nn.ReLU())
            #self.encoder.add_module(f'en_tan{idx}', nn.Tanh())
        # decoder
        for idx in range(len(hidden_dim) -1, 1, -1):
            self.decoder.add_module(f'de_lr{idx}', nn.Linear(hidden_dim[idx], hidden_dim[idx-1]))
            self.decoder.add_module(f'de_relu{idx}', nn.ReLU())
            #self.decoder.add_module(f'de_tan{idx}', nn.Tanh())
        self.decoder.add_module(f'de_lr{1}', nn.Linear(hidden_dim[1], hidden_dim[0]))
        
        
        # scoring network
        self.scores = nn.Sequential()
        self.scores.add_module(f'scores1', nn.Linear(hidden_dim[-1], 10))
        self.scores.add_module(f'relu', nn.ReLU())
        self.scores.add_module(f'scores2', nn.Linear(10, 1))
        
        

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        
        # scores
        scores = self.scores(enc)

        return scores, dec, enc

    def loss_function(self, x, x_dec, scores, anomal_flag, params):
        '''Reoconstruction error'''
        recon_error = torch.mean((x - x_dec) ** 2)
        dist_error = self.compute_dist_error(scores, anomal_flag, params)
        loss = recon_error + params.lam_dist * dist_error
        
        return loss, recon_error, dist_error
    
    def compute_dist_error(self, scores, anomal_flag, params):
        
        # inlier loss
        ref = torch.normal(mean=0, std=1.0, size=(10000,))
        dev = scores - torch.mean(ref)
        inlier_loss = torch.abs(dev)
        # outlier loss
        anomal_flag = anomal_flag.unsqueeze(1)
        outlier_loss = torch.abs(torch.max(params.a - scores, torch.zeros(scores.shape).to(params.device)))
        dist_error = torch.mean((1 - anomal_flag) * inlier_loss + params.lam_out * anomal_flag * outlier_loss)

        return dist_error
        
