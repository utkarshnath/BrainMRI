from helper import *
from run import Runner, Learn
from IPython.core.debugger import set_trace
from torch import tensor, nn, optim
from callback import *
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import partial
from torch.autograd import gradcheck
from schedulers import combine_schedules, sched_cos
#from model import xresnet18,xresnet50,xresnet101
import time
from optimizers import *
import argparse
from unet import UNet
from unetAdjoint import UNetAdjoint
from adjointNetwork import AdjointDiceLoss,AdjointLoss

parser = argparse.ArgumentParser(description='Adjoint Network')
parser.add_argument('--lr', type=int, default=0.001, help='')
parser.add_argument('--is_sgd', type=str, default='False', help='')
parser.add_argument('--is_adjoint_training', type=str, default='True',help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--compression_factor', type=int, default=4, help='')
parser.add_argument('--masking_factor', type=float, default=None, help='')
parser.add_argument('--image_size', type=int, default=32, help='')
parser.add_argument('--classes', type=int, default=100, help='')
parser.add_argument("--epoch", type=int, default=100, help="")
parser.add_argument("--resnet", type=int, default=50, help="")
parser.add_argument("--dataset", type=str, default='cifar100', help="")
parser.add_argument("--default_config", type=str, default='True', help="")
args = parser.parse_args()

def load_model(model, state_dict_file_path=None):
    if state_dict_file_path is not None:
        model.load_state_dict(torch.load(state_dict_file_path))
    return model


def dataset_resize(image_size,x): return x.view(-1, 3, image_size, image_size)

class AdjointLoss1(nn.Module):
    def __init__(self,weight=0.1):
        super().__init__()
        self.weight = torch.tensor((weight,1)).cuda()
        self.loss = nn.NLLLoss(weight=self.weight)
    def forward(self, output, target):
        log_preds = F.log_softmax(output, dim=1)
        nll = self.loss(log_preds, target)
        return nll

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, output, target):
        output = F.softmax(output, dim=1)
        #output = output[:,1]
        #target = target.reshape(-1)
        
        #output = output[:,0].view(-1)
        #target = target.view(-1)
        intersection = (output[:,1] * target).sum()
        denominator = (output[:,1] + target).sum()
        loss = (2.0*intersection + self.eps)/(denominator+self.eps)
        return 1.0 - loss
        #print(loss.shape)
        #dsc = (2. * intersection + self.smooth) / (
        #    output.sum() + target.sum() + self.smooth
        #)
        #return 1. - dsc
'''
class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_true):
        #print(y_pred.shape,y_true.shape)
        #assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
'''
if __name__ == "__main__":
   start = time.time()
   device = torch.device('cuda',0)

   path = '/scratch/un270/brain-mri/kaggle_3m'
   data = load_data(path, 256, args.batch_size)
   is_sgd = False
   is_individual_training = True
   epoch = args.epoch
   lr = 1e-3
 
   if is_sgd:
      lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e4)])
      lr_scheduler = ParamScheduler('lr', lr_sched,using_torch_optim=True)
      cbfs = [NormalizeCallback(device),lr_scheduler,CudaCallback()]
   else:
      lr_sched = combine_schedules([0.1, 0.9], [sched_cos(lr/10., lr), sched_cos(lr, lr/1e4)])
      beta1_sched = combine_schedules([0.1, 0.9], [sched_cos(0.95, 0.85), sched_cos(0.85, 0.95)])
      lr_scheduler = ParamScheduler('lr', lr_sched)
      beta1_scheduler = ParamScheduler('beta1', beta1_sched)
      cbfs = [NormalizeCallback(device),lr_scheduler,beta1_scheduler,CudaCallback(device),SaveModelCallback('individual-big')]

   if is_individual_training:
      loss_func = DiceLoss()
      #loss_func = AdjointLoss1()
      model = UNet(3,2,compression_factor=1)
      #total_params = sum(p.numel() for p in model.parameters())
      #print(total_params)
      cbfs+=[AvgStatsCallback(metrics=[])]
   else:
      loss_func = AdjointDiceLoss(0)
      cbfs+=[lossScheduler(),AvgStatsCallback(metrics=[accuracy_large,accuracy_small])]
      model = UNetAdjoint(3,2,compression_factor=1)
   end = time.time()
   print("Loaded model", end-start)

   if is_sgd:
      opt = optim.SGD(model.parameters(),lr)
   else:
      opt = StatefulOptimizer(model.parameters(), [weight_decay, adam_step],stats=[AverageGrad(), AverageSqGrad(), StepCount()], lr=0.001, wd=1e-2, beta1=0.9, beta2=0.99, eps=1e-6)

   learn = Learn(model,opt,loss_func, data)
   

   run = Runner(learn,cbs = cbfs)
   run.fit(epoch)
