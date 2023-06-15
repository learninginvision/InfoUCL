from torchvision import transforms
from PIL import Image

imagenet_norm = [[0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2615]]

class Transform_single():
    def __init__(self, image_size, train, normalize=imagenet_norm):
        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)
