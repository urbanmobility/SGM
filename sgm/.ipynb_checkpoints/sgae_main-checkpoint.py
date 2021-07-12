import os
import time
import argparse
import torch
import pandas as pd

from sgae_train import train_document, train_image, train_tabular

class parameter(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # train parameter
        parser.add_argument('--out_dir', type=str, default='./results/', 
                            help="Output directory.")
        parser.add_argument('--epochs', type=int, default=100,
                            help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=1e-4,
                            help='Initial learning rate.')
        parser.add_argument('--early_stop', action='store_false', default=True,
                            help='Whether to early stop.')
        parser.add_argument('--batch_size', type=int, default=1024,
                            help='Batch size.')
        parser.add_argument('--run_num', type=int, default=10,
                            help='Number of experiments')
        parser.add_argument('--cuda', type=str, default='0',
                            help='Choose cuda')
        parser.add_argument('--seed', type=int, default=42, help="Random seed.")
        parser.add_argument('--stop_train', type=float, default=0.5,
                            help='AUC to stop train.')
        
        
        # train information parameter
        parser.add_argument('--verbose', action='store_false', default=True,
                            help='Whether to print training details')
        parser.add_argument('--print_step', type=int, default=5,
                            help='Epoch steps to print training details')
        # data parameter
        parser.add_argument('--data_name', type=str, default='market',
                            help='Dataset name')
        parser.add_argument('--data_path', type=str, default=f'./data/',
                            help='Wether to inject noise to train data')
        parser.add_argument('--inject_noise', type=bool, default=True,
                            help='Whether to inject noise to train data')
        parser.add_argument('--cont_rate', type=float, default=0.01,
                            help='Inject noise to contamination rate')
        parser.add_argument('--anomal_rate', type=str, default='default',
                            help='Adjust anomaly rate')

        
        # model parameter
        ## General
        parser.add_argument('--lam_out', type=float, default=20,
                            help='Parameter Lambda_outliers')
        parser.add_argument('--lam_dist', type=float, default=0.01,
                            help='Parameter Lambda_DE')
        parser.add_argument('--a', type=float, default=6,
                            help='Parameter a')
        parser.add_argument('--epsilon', type=float, default=90,
                            help='Parameter epsilon')
        # Specific
        parser.add_argument('--model_name', type=str, default='SG-AE',
                            help='Choose model')
        parser.add_argument('--hidden_dim', type=str, default='auto',
                            help='Hidden dimension of the model')
        
        if __name__ == '__main__':
            args = parser.parse_args()
        else:
            args = parser.parse_args([])
            
        args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
            
        # Specific design        
        
        self.__dict__.update(args.__dict__)
        
    def update(self, update_dict):
        logs = '==== Parameter Update \n'
        origin_dict = self.__dict__
        for key in update_dict.keys():
            if key in origin_dict:
                logs += f'{key} ({origin_dict[key]} -> {update_dict[key]}), '
                origin_dict[key] = update_dict[key]
            else:
                logs += f'{key} ({update_dict[key]}), '
        self.__dict__ = origin_dict
        print(logs)


if __name__ == '__main__':
    
    start_time = time.time()
    time_name = str(time.strftime("%m%d")) + '_' + str(time.time()).split(".")[1][-3:]
    print(f'Time name is {time_name}')
    print(os.getcwd())
    # Total metrics
    metrics = pd.DataFrame()
    
    # Conduct one experiements
    args = parameter()
    print(f'Device is {args.device.type}-{args.cuda}')
    if args.data_name in ['attack', 'bcsc', 'creditcard', 'diabetic', 'donor', 'intrusion', 'market']:
        an_metrics_dict = train_tabular(args)
    elif args.data_name in ['reuters', '20news']:
        an_metrics_dict = train_document(args)
    elif args.data_name in ['mnist']:
        an_metrics_dict = train_image(args)
    metrics = pd.DataFrame(an_metrics_dict, index=[0])
    metrics.to_csv(f'{args.out_dir}{args.model_name}_{args.data_name}_{time_name}.csv')
    
    print(f'Finished!\nTotal time is {time.time()-start_time:.2f}s')
    print(f'Current time is {time.strftime("%m%d_%H%M")}')
    print(f'Results:')
    print(metrics.sort_values('AUC', ascending=False))
    