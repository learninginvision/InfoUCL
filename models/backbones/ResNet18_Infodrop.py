# The following code is adapted from the file "resnet.py" of the library: https://github.com/bfshi/InfoDrop/domain_generalization,
# which is available under the terms of the GNU General Public License v3.0.

import torch
import torch.nn as nn
import os
# https://raw.githubusercontent.com/huyvnphan/PyTorch_CIFAR10/master/cifar10_models/resnet.py
__all__ = ['ResNetIFD', 'resnet_ifd18', 'resnet_ifd34', 'resnet_ifd50', 'resnet_ifd101',
           'resnet_ifd152', 'resnext_ifd50_32x4d', 'resnext_ifd101_32x8d']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

# We made minor modifications to InfoDrop in `https://github.com/bfshi/InfoDrop/domain_generalization/models/resnet.py`: 
# (1) We designed the hyperparameters of InfoDrop as public attributes of the class for easier parameter setting; 
# (2) We changed the type of padder used in InfoDrop to `ReflectionPad2d`, which can enhance InfoDrop's performance on small-sized images;
class Info_Dropout(nn.Module):
    
    drop_rate = 1.25
    temperature = 0.03
    band_width = 1.0
    radius = 2
    
    def __init__(self, indim, outdim, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, if_pool=False, pool_kernel_size=2, pool_stride=None,
                 pool_padding=0, pool_dilation=1):
        super(Info_Dropout, self).__init__()
        if groups != 1:
            raise ValueError('InfoDropout only supports groups=1')

        self.indim = indim
        self.outdim = outdim
        self.if_pool = if_pool

        self.patch_sampling_num = 9

        self.all_one_conv_indim_wise = nn.Conv2d(self.patch_sampling_num, self.patch_sampling_num,
                                                 kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation,
                                                 groups=self.patch_sampling_num, bias=False)
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise = nn.Conv2d(self.patch_sampling_num, outdim, kernel_size=1, padding=0, bias=False)
        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


        if if_pool:
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding, pool_dilation)

        # the original padding is zero padding, which is not suitable for small-sized images
        self.padder = nn.ReflectionPad2d((padding + self.radius, padding + self.radius + 1,
                                         padding + self.radius, padding + self.radius + 1))

    def initialize_parameters(self):
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False

    @classmethod
    def set_hyperparameter(cls, 
                           drop_rate=1.25, 
                           temperature=0.03,
                           band_width=1.0,
                           radius=2):
        cls.drop_rate = drop_rate
        cls.temperature = temperature
        cls.band_width = band_width
        cls.radius = radius
    
    @classmethod
    def get_hyperparameter(cls):
        return {'drop rate':cls.drop_rate, 
                'temperature':cls.temperature, 
                'band width':cls.band_width, 
                'radius':cls.radius}

    def forward(self, x_old, x, use_info_drop=False):
        if not use_info_drop:
            return x

        with torch.no_grad():
            distances = []
            padded_x_old = self.padder(x_old)
            sampled_i = torch.randint(-self.radius, self.radius + 1, size=(self.patch_sampling_num,)).to(torch.int).tolist()
            sampled_j = torch.randint(-self.radius, self.radius + 1, size=(self.patch_sampling_num,)).to(torch.int).tolist()
            for i, j in zip(sampled_i, sampled_j):
                tmp = padded_x_old[:, :, self.radius: -self.radius - 1, self.radius: -self.radius - 1] - \
                      padded_x_old[:, :, self.radius + i: -self.radius - 1 + i,
                      self.radius + j: -self.radius - 1 + j]
                distances.append(tmp.clone())
            distance = torch.cat(distances, dim=1)
            batch_size, _, h_dis, w_dis = distance.shape
            distance = (distance**2).view(-1, self.indim, h_dis, w_dis).sum(dim=1).view(batch_size, -1, h_dis, w_dis)
            distance = self.all_one_conv_indim_wise(distance)
            distance = torch.exp(
                -distance / distance.mean() / 2 / self.band_width ** 2)  # using mean of distance to normalize
            prob = (self.all_one_conv_radius_wise(distance) / self.patch_sampling_num) ** (1 / self.temperature)

            if self.if_pool:
                prob = -self.pool(-prob)  # min pooling of probability
            prob /= (prob.sum(dim=(-2, -1), keepdim=True) + 1e-16) # add a small number to avoid dividing by zero

            batch_size, channels, h, w = x.shape

            random_choice = random_sample(prob, sampling_num=int(self.drop_rate * h * w))

            random_mask = torch.ones((batch_size * channels, h * w), device=x.device)
            random_mask[torch.arange(batch_size * channels, device=x.device).view(-1, 1), random_choice] = 0

        return x * random_mask.view(x.shape)


class BasicBlockIFD(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, if_dropout=False):
        super(BasicBlockIFD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.if_dropout = if_dropout
        if if_dropout:
            self.info_dropout1 = Info_Dropout(inplanes, planes, kernel_size=3, stride=stride,
                                              padding=1, groups=1, dilation=1)
            self.info_dropout2 = Info_Dropout(planes, planes, kernel_size=3, stride=1,
                                              padding=1, groups=1, dilation=1)


    def forward(self, x, use_info_dropout=False):
        identity = x

        x_old = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.if_dropout:
            out = self.info_dropout1(x_old, out, use_info_dropout)

        x_old = out.clone()
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.downsample is None and self.if_dropout:
            out = self.info_dropout2(x_old, out, use_info_dropout)

        return out


class BottleneckIFD(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, if_dropout=False):
        super(BottleneckIFD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.if_dropout = if_dropout
        if if_dropout:
            self.info_dropout1 = Info_Dropout(inplanes, planes, kernel_size=3, stride=stride,
                                              padding=1, groups=1, dilation=1)
            self.info_dropout2 = Info_Dropout(planes, planes, kernel_size=3, stride=1,
                                              padding=1, groups=1, dilation=1)

    def forward(self, x, use_info_dropout=False):
        identity = x

        x_old = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.if_dropout:
            out = self.info_dropout1(x_old, out, use_info_dropout)

        x_old = out.clone()
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.if_dropout:
            out = self.info_dropout2(x_old, out, use_info_dropout)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Layer(nn.Module):
    def __init__(self, setting, block, planes, blocks, stride=1, dilate=False, if_dropout=False):
        super(Layer, self).__init__()
        norm_layer = setting._norm_layer
        downsample = None
        previous_dilation = setting.dilation
        if dilate:
            setting.dilation *= stride
            stride = 1
        if stride != 1 or setting.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(setting.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        self.layers = nn.ModuleList()
        self.layers.append(block(setting.inplanes, planes, stride, downsample, setting.groups,
                                 setting.base_width, previous_dilation, norm_layer, if_dropout=if_dropout))
        setting.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            self.layers.append(block(setting.inplanes, planes, groups=setting.groups,
                                     base_width=setting.base_width, dilation=setting.dilation,
                                     norm_layer=norm_layer))

    def forward(self, x, use_info_dropout=False):
        for block in self.layers:
            x = block(x, use_info_dropout)
        return x


class ResNetIFD(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout_layers=0.5):
        super(ResNetIFD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.dropout_layers = dropout_layers
        
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.n_classes = num_classes
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        ## CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3 -> 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        ## END
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        if dropout_layers > 0:
            self.info_dropout = Info_Dropout(3, self.inplanes, kernel_size=3, stride=1, padding=1, if_pool=False)
        
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], if_dropout=(dropout_layers>=1))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], if_dropout=(dropout_layers>=2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], if_dropout=(dropout_layers>=3))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], if_dropout=(dropout_layers>=4))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckIFD):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockIFD):
                    nn.init.constant_(m.bn2.weight, 0)

        # This is for Info-Dropout initialization
        for m in self.modules():
            if isinstance(m, Info_Dropout):
                # print(m.drop_rate, m.temperature, m.band_width, m.radius)
                m.initialize_parameters()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, if_dropout=False):

        return Layer(setting=self, block=block, planes=planes, blocks=blocks, stride=stride, dilate=dilate, if_dropout=if_dropout)
    
    def embed(self, x, use_info_dropout=False):
        x_old = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        # x = self.maxpool(x)
        
        if self.dropout_layers > 0:
            x1 = self.info_dropout(x_old, x1, use_info_dropout)
            
        x2 = self.layer1(x1, use_info_dropout)
        x3 = self.layer2(x2, use_info_dropout)
        x4 = self.layer3(x3, use_info_dropout)
        x5 = self.layer4(x4, use_info_dropout)

        x = self.avgpool(x5)
        x = x.reshape(x.size(0), -1)
        return x, [x1, x2, x3, x4, x5]

    def forward(self, x, use_info_dropout=False, return_features=False, plot_metrics=False):
        x, features_list = self.embed(x, use_info_dropout)
        
        if return_features:
            return x
        else:
            x = self.fc(x)
        
        if plot_metrics:
            return x, features_list
        return x

    def get_params(self):
        params = []
        for pp in list(self.parameters()):
          # print(pp[0])
          # if pp.grad is not None:
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self):
        grads = []
        for pp in list(self.parameters()):
            # if pp.grad is not None:
            if pp.grad is None:grads.append(torch.zeros_like(pp).view(-1))
            else:grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNetIFD(block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(script_dir + '/state_dicts/'+arch+'.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model


def resnet_ifd18(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlockIFD, [2, 2, 2, 2], pretrained, progress, device,
                   **kwargs)


def resnet_ifd34(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlockIFD, [3, 4, 6, 3], pretrained, progress, device,
                   **kwargs)


def resnet_ifd50(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', BottleneckIFD, [3, 4, 6, 3], pretrained, progress, device,
                   **kwargs)


def resnet_ifd101(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', BottleneckIFD, [3, 4, 23, 3], pretrained, progress, device,
                   **kwargs)


def resnet_ifd152(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', BottleneckIFD, [3, 8, 36, 3], pretrained, progress, device,
                   **kwargs)


def resnext_ifd50_32x4d(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', BottleneckIFD, [3, 4, 6, 3],
                   pretrained, progress, device, **kwargs)


def resnext_ifd101_32x8d(pretrained=False, progress=True, device='cpu', **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', BottleneckIFD, [3, 4, 23, 3],
                   pretrained, progress, device, **kwargs)


if __name__ == "__main__":
    model = resnet_ifd18()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 11173962
    y = model(torch.randn(64, 3, 32, 32), 1)
    print(y.size(), model)

