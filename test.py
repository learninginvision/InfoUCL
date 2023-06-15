import os
import torch
import numpy as np
import argparse
import yaml
import re
from models import get_model
from datasets import get_dataset
from metrics import knn_monitor
from utils.conf import set_deterministic
from metrics import save_results

class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--device', type=str, default='cuda'  if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cl_default', action='store_true')
    parser.add_argument('--log_dir', type=str, default=None, required=True)
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')

    args = parser.parse_args()
    # find .yaml file
    config_file = None
    for root, dirs, files in os.walk(args.log_dir):
        for file in files:
            if file.endswith('.yaml'):
                config_file = os.path.join(root, file)
                break
    print('config file: ', config_file)
    assert config_file is not None, 'can not find .yaml file in log_dir'

    with open(config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    if not hasattr(args.train, 'lambda_info'):
        args.train.lambda_info = 0.0
    
    args.cl_type = 'scl' if args.cl_default else 'ucl'

    # set deterministic
    set_deterministic(args.seed)

    # define augmentation kwargs, dataset kwargs and dataloader kwargs
    vars(args)['aug_kwargs'] = {
        'name':args.model.name,
        'image_size': args.dataset.image_size,
        'cl_default': args.cl_default
    }
    
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'download':args.download,
    }

    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args

def main(device, args):
    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, _, _ = dataset_copy.get_data_loaders(args)
    knn_acc_results = []

    # define model
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))
    
    # get data loaders for each task
    train_loaders, memory_loaders, test_loaders = [], [], []
    for t in range(dataset.N_TASKS):
        tr, me, te = dataset.get_data_loaders(args)
        train_loaders.append(tr)
        memory_loaders.append(me)
        test_loaders.append(te)

    # find checkpoints
    checkpoints = []
    for root, dirs, files in os.walk(args.log_dir):
        for file in files:
            if file.endswith('.pth'):
                checkpoints.append(os.path.join(root, file))
                
    checkpoints.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

    eval_tids = [j for j in range(dataset.N_TASKS)]
    for t in range(dataset.N_TASKS):

        # load checkpoint
        save_dict = torch.load(checkpoints[t], map_location='cpu')
        msg = model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k}, strict=True)
        print('load backbone from: ', checkpoints[t])
        
        # test current model on all tasks
        knn_acc_list = []
        for i in eval_tids:
            acc, _ = knn_monitor(model.net.module.backbone, dataset, memory_loaders[i], test_loaders[i],
                                 device, args.cl_default, task_id=i, k=min(args.train.knn_k, len(eval_tids)))
            knn_acc_list.append(acc)

        # memorize acc of current model on all tasks
        knn_acc_results.append(knn_acc_list)

    return knn_acc_results
    
if __name__ == "__main__":
    
    # get arguments
    args = get_args()
    
    # run main
    results = main(device=args.device, args=args)
    
    # save results
    summary_file = os.path.join(args.log_dir, f'test_summary.txt')
    save_results(results, summary_file)

    print(f'Summary file has been saved to {summary_file}')