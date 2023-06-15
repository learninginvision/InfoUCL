# The following code is adapted from the file "barlowtwins.py" of the library: https://github.com/divyam3897/UCL,
# which is available under the terms of the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss


class BarlowTwins(nn.Module):
    def __init__(self, backbone, device):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor_info = prediction_MLP()
        self.D = self.criterion = BarlowTwinsLoss(device=device)
    
    def forward(self, x1, x2, lambda_info=None):
        
        z1, z2 = self.encoder(x1), self.encoder(x2)
        
        # SSL loss
        L = self.criterion(z1, z2)

        # InfoDrop loss
        if lambda_info == None:
            return {'loss': L, 'loss_info': torch.tensor(0.).to(L.device)}
        else:
            b, p, h_info = self.backbone, self.projector, self.predictor_info

            with torch.no_grad():
                z1_id, z2_id = p(b(x1, use_info_dropout=True)), p(b(x2, use_info_dropout=True))

            L_info = self.criterion(h_info(z1), z1_id) / 2 + self.criterion(h_info(z2), z2_id) / 2
            
            L_all = L + lambda_info * L_info
            
            return {'loss': L_all, 'loss_info': lambda_info * L_info}


if __name__ == "__main__":
    model = BarlowTwins()
    model = torch.nn.DataParallel(model).cuda()
    x1 = torch.randn((128, 3, 32, 32))
    x2 = torch.randn_like(x1)

    for i in range(50):
        model.forward(x1, x2).backward()
    print("forward backwork check")