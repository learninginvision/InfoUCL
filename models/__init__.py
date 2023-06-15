import os
import importlib
from .ssl_models import SimSiam
from .ssl_models import BarlowTwins
from .ssl_models import NNCLR
import torch
from .backbones import resnet18, resnet18_ifd
from .backbones import Info_Dropout as Info_Dropout

def modify_backbone(backbone, dataset, castrate=True):
    if dataset == 'seq-cifar100':
        backbone.n_classes = 100
    elif dataset == 'seq-cifar10':
        backbone.n_classes = 10
    backbone.output_dim = backbone.fc.in_features
    if not castrate:
        backbone.fc = torch.nn.Identity()

    return backbone

def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

def get_model(args, device, len_train_loader, transform):
    # set loss for supervised continual learning
    
    loss = torch.nn.CrossEntropyLoss()

    if args.model.backbone == 'resnet18_ifd':
        Info_Dropout.set_hyperparameter(drop_rate=args.model.infodrop.drop_rate,
                                        temperature=args.model.infodrop.temperature,
                                        band_width=args.model.infodrop.band_width,
                                        radius=args.model.infodrop.radius)
        backbone = eval(f"{args.model.backbone}(dropout_layers={args.model.infodrop.dropout_layers})")
    elif args.model.backbone == 'resnet18':
        backbone = eval(f"{args.model.backbone}()")
    else:
        raise NotImplementedError
    
    backbone = modify_backbone(backbone, args.dataset.name, args.cl_default)

    if args.model.name == 'simsiam':
        ssl_model = SimSiam(backbone).to(device)
    elif args.model.name == 'barlowtwins':
        ssl_model = BarlowTwins(backbone, device).to(device)
    elif args.model.name == 'nnclr':
        ssl_model = NNCLR(backbone).to(device)
    else:
        raise NotImplementedError
    
    if args.model.proj_layers is not None:
        ssl_model.projector.set_layers(args.model.proj_layers)

    names = {}
    for model in get_all_models():
        mod = importlib.import_module('models.' + model)
        class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
        names[model] = getattr(mod, class_name)
    
    return names[args.model.cl_model](ssl_model, loss, args, len_train_loader, transform)