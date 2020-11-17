from torch.nn.parameter import Parameter
import torch
from torch.autograd import Function
from torch import tensor, nn
import math
import torch.nn.functional as F
import time

def test(a,b,cmp,cname=None):
    if cname is None: cname=cmp.__name__
    assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)

class conv2dFirstLayer(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,*kargs,**kwargs):
        super(conv2dFirstLayer, self).__init__(in_channels,out_channels,kernel_size,padding,stride,*kargs, **kwargs)
        self.padding = (padding,padding)
        self.stride = (stride,stride)

    def forward(self,input):
        #print('First',input.shape)
        #print('input',input[0][0][0])
        #print('weight',self.weight[0][0])
        a = F.conv2d(input,self.weight,self.bias,self.stride,self.padding,self.dilation, self.groups)
        concatinatedTensor = torch.cat([a, a], dim=0)
        return concatinatedTensor

class conv2dAdjoint(nn.Conv2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask_layer,compression_factor=1,masking_factor=None,*kargs,**kwargs):
        super(conv2dAdjoint, self).__init__(in_channels,out_channels,kernel_size,padding,stride,*kargs, **kwargs)
        self.padding = (padding,padding)
        self.stride = (stride,stride)
        self.mask_layer = mask_layer
        self.out_channels = out_channels
        self.compression_factor = compression_factor
        if masking_factor!=None:
           self.mask = randomShape(kernel_size,kernel_size,masking_factor)
        else:
          self.mask = 1
        self.isFirst = True

    def forward(self,input):
        #print('Second',input.shape)
        l,_,_,_ = input.shape
        a = F.conv2d(input[:l//2],self.weight,self.bias,self.stride,self.padding)
        if self.mask_layer:
           b = F.conv2d(input[l//2:],self.weight*self.mask,self.bias,self.stride,self.padding)
           b[:,self.out_channels//self.compression_factor:] = 0
           concatinatedTensor = torch.cat([a, b], dim=0)
        else:
           concatinatedTensor = torch.cat([a, a], dim=0)

        return concatinatedTensor

class ConvTranspose2dAdjoint(nn.ConvTranspose2d):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride,mask_layer,compression_factor=1,masking_factor=None,*kargs,**kwargs):
        super(ConvTranspose2dAdjoint, self).__init__(in_channels,out_channels,kernel_size,stride,padding,*kargs, **kwargs)
        self.padding = (padding,padding)
        self.stride = (stride,stride)
        self.mask_layer = mask_layer
        self.out_channels = out_channels
        self.compression_factor = compression_factor
        if masking_factor!=None:
           self.mask = randomShape(kernel_size,kernel_size,masking_factor)
        else:
          self.mask = 1
        self.isFirst = True

    def forward(self,input):
        l,_,_,_ = input.shape
        a = F.conv_transpose2d(input[:l//2],self.weight,self.bias,self.stride,self.padding)
        if self.mask_layer:
           b = F.conv_transpose2d(input[l//2:],self.weight*self.mask,self.bias,self.stride,self.padding)
           b[:,self.out_channels//self.compression_factor:] = 0
           concatinatedTensor = torch.cat([a, b], dim=0)
        else:
           concatinatedTensor = torch.cat([a, a], dim=0)

        return concatinatedTensor
 
class batchNorm(nn.Module):
    def __init__(self,num_features,*kargs,**kwargs):
        super(batchNorm,self).__init__(*kargs,**kwargs)
        self.num_features = num_features
        self.bn1 = nn.BatchNorm2d(num_features)
        self.bn2 = nn.BatchNorm2d(num_features)

    def forward(self,input):
        l,_,_,_ = input.shape
        a = self.bn1(input[:l//2])
        d = self.bn2(input[l//2:])
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor

class linear(nn.Linear):
    def __init__(self,in_features, out_features, parts=4, bias=True,*kargs,**kwargs):
        super(linear, self).__init__(in_features, out_features, bias=True,*kargs, **kwargs)

    def forward(self,input):
        l,_ = input.shape
        a = F.linear(input[:l//2], self.weight, self.bias)
        d = F.linear(input[l//2:], self.weight, self.bias)
        concatinatedTensor = torch.cat([a, d], dim=0)
        return concatinatedTensor

class AdjointLoss(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        l,_,_,_ = output.shape
        log_preds1 = F.log_softmax(output[:l//2], dim=1)
        #print(log_preds1.shape,target.shape)
        
        nll1 = F.nll_loss(log_preds1, target)
        
        prob1 = F.softmax(output[:l//2], dim=-1)
        prob2 = F.softmax(output[l//2:], dim=-1)
        kl = (prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)
        
        return nll1 + self.alpha * kl.mean()

class AdjointDiceLoss(nn.Module):
    def __init__(self,alpha=1):
        super().__init__()
        self.eps = 1e-6
        self.alpha = alpha

    def forward(self, output, target):
        l,_,_,_ = output.shape
        prob_larger = F.softmax(output[:l//2], dim=1)
        prob_smaller = F.softmax(output[l//2:], dim=1)
        intersection = (prob_larger[:,1] * target).sum()
        denominator = (prob_larger[:,1] + target).sum()
        loss = (2.0*intersection + self.eps)/(denominator+self.eps)
        dsc = 1.0 - loss

        prob_larger = prob_larger.permute(0,2,3,1).reshape(-1,2)
        prob_smaller = prob_smaller.permute(0,2,3,1).reshape(-1,2)
        #kl = F.kl_div(prob_larger, prob_smaller)
        kl = (prob_larger * torch.log(1e-6 + prob_larger/(prob_smaller+1e-6))).sum(-1)
        '''
        out = output[:l//2,0].reshape(-1)
        target = target.reshape(-1)
        intersection = (out * target).sum()
        dsc = (2. * intersection + self.smooth) / (out.sum() + target.sum() + self.smooth)
        dsc = 1. - dsc

        prob1 = output[:l//2]
        prob2 = output[l//2:]
        #kl = (prob1 * torch.log(self.smooth + prob1/(prob2+self.smooth))).sum((2,3))
        kl = F.kl_div(prob1, prob2)
        print(kl,dsc)
        '''
        return dsc + self.alpha*kl.mean()
