import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.utils.continual_dataset import ContinualDataset
from augmentations import get_aug
from torchvision import transforms
from utils.conf import base_path

from PIL import Image
import os
import os.path
import numpy as np
from copy import deepcopy


class IDIGITSDataset(Dataset):
    def __init__(
            self,
            root: str, 
            domain: str,
            split: str,
            transform: transforms=None,
            target_transform: transforms=None,

        ) -> bool:

        self.root = root
        self.domain = domain
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        DOMAIN_NAME = ["mnist", "mnistm", "synnum", "svhn"]

        if split == "train" or split == "test":
            assert domain in DOMAIN_NAME, "if split is {}, domain should be in {}".format(split, DOMAIN_NAME)
        elif split == "memory":
            assert domain == None, "if split is {}, domain should be None".format(split)
        else:
            raise ValueError("Split {} not recognized.".format(split))

        flag = 'train' if split in ['train', 'memory'] else 'test' # train/test
        
        if domain is not None:
            self.data = np.load(os.path.join(self.root, flag, '{}_images.npy'.format(domain)))
            self.targets = np.load(os.path.join(self.root, flag, '{}_labels.npy'.format(domain)))
        else:
            self.data = []
            self.targets = []
            for d in DOMAIN_NAME:
                self.data.append(np.load(os.path.join(self.root, flag, '{}_images.npy'.format(d))))
                self.targets.append(np.load(os.path.join(self.root, flag, '{}_labels.npy'.format(d))))
            
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)

        if len(self.targets.shape) == 2:
            self.targets = np.squeeze(self.targets, axis=1)


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class SequentialIDIGITS(ContinualDataset):

    NAME = 'seq-idigits'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 4
    DOMAIN_NAME = ["mnist", "mnistm", "synnum", "svhn"] # define the domain arrival order

    def get_data_loaders(self, args):
        transform = get_aug(train=True, **args.aug_kwargs)
        test_transform = get_aug(train=False, train_classifier=False, **args.aug_kwargs)

        data_path = base_path() + 'IDIGITS'

        train_dataset = IDIGITSDataset(data_path, self.DOMAIN_NAME[self.i], 'train', transform=transform)
        train_loader = DataLoader(train_dataset,
                                batch_size=args.train.batch_size, shuffle=True, num_workers=4)

        memory_dataset = IDIGITSDataset(data_path, None, 'memory', transform=test_transform)
        memory_loader = DataLoader(memory_dataset,
                                batch_size=args.train.batch_size, shuffle=False, num_workers=4)

        test_dataset = IDIGITSDataset(data_path, self.DOMAIN_NAME[self.i], 'test', transform=test_transform)
        test_loader = DataLoader(test_dataset,
                                batch_size=args.train.batch_size, shuffle=False, num_workers=4)
        
        self.train_loaders.append(train_loader)
        self.memory_loaders.append(memory_loader)
        self.test_loaders.append(test_loader)

        self.i += 1
        return train_loader, memory_loader, test_loader


    def get_transform(self, args):
        idigit_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]

        if args.cl_default:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*idigit_norm)
                ])
        else:
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*idigit_norm)
                ])

        return transform

    def not_aug_dataloader(self, batch_size):
        raise NotImplementedError