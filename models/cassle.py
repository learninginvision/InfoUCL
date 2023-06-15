from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from models.ssl_models.simsiam import prediction_MLP
import torch
from copy import deepcopy

class CaSSLe(ContinualModel):
    NAME = 'cassle'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, model, loss, args, len_train_loader, transform):
        super(CaSSLe, self).__init__(model, loss, args, len_train_loader, transform)
        self.previous_net = None
        self.distill_predictor = prediction_MLP().to(self.device)
        self.opt.add_param_group({'name': 'predictor', 
                                  'params': self.distill_predictor.parameters(), 
                                  'lr': args.train.base_lr*args.train.batch_size/256})
        assert args.cl_default == False, 'CaSSLe only supports unsupervised continual learning'
        
    def penalty(self, x1, x2):
        '''page 4: https://arxiv.org/pdf/2112.04215.pdf
        Compute the penalty term for the distillation loss,
        Distillation loss. 
        '''
        if self.previous_net is None:return torch.tensor(0.0).to(self.device)
        with torch.no_grad():
            frozen_f = self.previous_net.module.encoder
            frozen_z1, frozen_z2 = frozen_f(x1), frozen_f(x2)
            
        f, h = self.net.module.encoder, self.distill_predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        if self.args.model.name == 'simsiam':
            distill_loss = (self.net.module.D(p1, frozen_z1) + 
                            self.net.module.D(p2, frozen_z2)) / 2
        elif self.args.model.name == 'barlowtwins':
            distill_loss = (self.net.module.D(p1, frozen_z1) + 
                            self.net.module.D(p2, frozen_z2)) / 2
        elif self.args.model.name == 'nnclr':
            # following: https://github.com/DonkeyShot21/cassle/blob/main/cassle/distillers/contrastive.py
            p = torch.cat([p1,p2], dim=0)
            frozen_z = torch.cat([frozen_z1, frozen_z2], dim=0)
            distill_loss = self.net.module.D(p, frozen_z) 
        else:
            raise NotImplementedError
        return distill_loss
    
    def end_task(self, dataset):
        self.previous_net = deepcopy(self.net)
        for param in self.previous_net.parameters():
            param.requires_grad = False
        
    def observe(self, inputs1, labels, inputs2, notaug_inputs):

        self.opt.zero_grad()
        inputs1 = inputs1.to(self.device, non_blocking=True)
        inputs2 = inputs2.to(self.device, non_blocking=True)
        
        data_dict = self.net.forward(inputs1, inputs2,
                                     lambda_info=self.args.train.lambda_info)
        data_dict['penalty'] = self.args.train.alpha * self.penalty(inputs1, inputs2)
        data_dict['loss'] = data_dict['loss'].mean()
        loss = data_dict['loss'] + data_dict['penalty']
        loss.backward()
        self.opt.step()
        data_dict.update({'lr': self.args.train.base_lr})

        return data_dict
