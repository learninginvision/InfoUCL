# This program uses the following open source libraries:
# Lightly: https://github.com/lightly-ai/lightly (MIT License)
# UCL: https://github.com/divyam3897/UCL (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
import torch.distributed as dist
from .utils import NNMemoryBankModule


def rank() -> int:
    """Returns the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def nnclr_D(out0, out1, temperature=0.5):
    device = out0.device
    batch_size, _ = out0.shape
    out0 = F.normalize(out0, dim=1)
    out1 = F.normalize(out1, dim=1)

    diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

    # calculate similiarities
    # here n = batch_size and m = batch_size * world_size
    logits_00 = torch.einsum('nc,mc->nm', out0, out0) / temperature
    logits_01 = torch.einsum('nc,mc->nm', out0, out1) / temperature
    logits_10 = torch.einsum('nc,mc->nm', out1, out0) / temperature
    logits_11 = torch.einsum('nc,mc->nm', out1, out1) / temperature
    logits_00 = logits_00[~diag_mask].view(batch_size, -1)
    logits_11 = logits_11[~diag_mask].view(batch_size, -1)
    # concatenate logits
    # the logits tensor in the end has shape (2*n, 2*m-1)
    logits_0100 = torch.cat([logits_01, logits_00], dim=1)
    logits_1011 = torch.cat([logits_10, logits_11], dim=1)
    logits = torch.cat([logits_0100, logits_1011], dim=0)

    # create labels
    labels = torch.arange(batch_size, device=device, dtype=torch.long)
    labels = labels + rank() * batch_size
    labels = labels.repeat(2)

    loss = F.cross_entropy(logits, labels, reduction='mean')
    
    return loss


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


class NNCLR(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
        self.predictor_info = prediction_MLP()
        self.memory_bank = NNMemoryBankModule(size=4096)
        self.D = nnclr_D
        
    def forward(self, x1, x2, lambda_info=None):
        b, p, h, h_info = self.backbone, self.projector, self.predictor, self.predictor_info
        
        z1, z2 = p(b(x1)), p(b(x2))
        p1, p2 = h(z1), h(z2)

        p1_info, p2_info = h_info(z1), h_info(z2) 

        z1, z2 = z1.detach(), z2.detach()
        z1, z2 = self.memory_bank(z1, update=False), self.memory_bank(z2, update=True)

        # SSL loss
        L_base = 0.5 * (self.D(z1, p2) + self.D(z2, p1))

        # Infodrop loss
        if lambda_info == None:
            return {'loss': L_base.mean(), 'loss_info': torch.tensor(0.).to(L_base.device)}
        else:
            with torch.no_grad():
                z1_id, z2_id = p(b(x1, use_info_dropout=True)), p(b(x2, use_info_dropout=True))
            
            L_info = self.D(z1_id, p1_info) / 2 + self.D(z2_id, p2_info) / 2
                            
            L = L_base + lambda_info * L_info 

            return {'loss': L.mean(), 'loss_info': lambda_info * L_info.mean()}

    def get_params(self):
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)


    def get_grads(self):
        grads = []
        for pp in list(self.parameters()):
            # if pp.grad is not None:
            if pp.grad is None:grads.append(torch.zeros_like(pp).view(-1))
            else:grads.append(pp.grad.view(-1))
        return torch.cat(grads)


if __name__ == "__main__":
    model = NNCLR()
    model = torch.nn.DataParallel(model).cuda()
    x1 = torch.randn((128, 3, 32, 32))
    x2 = torch.randn_like(x1)

    for i in range(50):
        model.forward(x1, x2).backward()
    print("forward backwork check")