import argparse
import os
import torch
import numpy as np
import torch
import random
import re
import yaml
import shutil
from datetime import datetime
from .conf import set_deterministic

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
    parser.add_argument('-c', '--config-file', required=True, type=str, help="configs/simsiam_c10_cassle.yaml")
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--device', type=str, default='cuda'  if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hide_progress', action='store_false')
    parser.add_argument('--cl_default', action='store_true')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    # check if the trade-off of InfoUCL is valid
    if hasattr(args.train, 'lambda_info'):
        assert 'ifd' in args.model.backbone, 'InfoUCL is only valid when using InfoDrop'
    else:
        args.train.lambda_info = None

    assert not None in [args.log_dir, args.name]
    args.cl_type = 'scl' if args.cl_default else 'ucl'

    args.name = args.cl_type + '_' + args.name + '_' + args.model.cl_model

    assert not None in [args.log_dir, args.name]

    args.log_dir = os.path.join(args.log_dir, 'in-progress_' + datetime.now().__format__('%m-%d-%H-%M-%S') + '-' + args.name)

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')

    # copy config file to log_dir
    shutil.copy2(args.config_file, args.log_dir)
    
    # set seed
    set_deterministic(args.seed)

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
