from collections import OrderedDict

import torch
import torch.nn as nn
from adjointNetwork import conv2dFirstLayer, conv2dAdjoint, ConvTranspose2dAdjoint, batchNorm

class UNetAdjoint(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, compression_factor=1):
        super(UNetAdjoint, self).__init__()

        features = init_features
        self.encoder1 = UNetAdjoint._firstblock(in_channels, features, name="enc1", compression_factor=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetAdjoint._block(features, features * 2, name="enc2",compression_factor=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetAdjoint._block(features * 2, features * 4, name="enc3",compression_factor=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetAdjoint._block(features * 4, features * 8, name="enc4",compression_factor=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetAdjoint._block(features * 8, features * 16, name="bottleneck",compression_factor=2)

        self.upconv4 = ConvTranspose2dAdjoint(features * 16, features * 8, 2, padding=0, stride=2, compression_factor=2, mask_layer=True)
        self.decoder4 = UNetAdjoint._block((features * 8) * 2, features * 8, name="dec4",compression_factor=2)
        self.upconv3 = ConvTranspose2dAdjoint(features * 8, features * 4, kernel_size=2, stride=2, padding=0, mask_layer=True, compression_factor=2)
        self.decoder3 = UNetAdjoint._block((features * 4) * 2, features * 4, name="dec3", compression_factor=2)
        self.upconv2 = ConvTranspose2dAdjoint(features * 4, features * 2, kernel_size=2, stride=2, padding=0, mask_layer=True, compression_factor=1)
        self.decoder2 = UNetAdjoint._block((features * 2) * 2, features * 2, name="dec2", compression_factor=1)
        self.upconv1 = ConvTranspose2dAdjoint(features * 2, features, kernel_size=2, stride=2, padding=0, mask_layer=True, compression_factor=1)
        self.decoder1 = UNetAdjoint._block(features * 2, features, name="dec1", compression_factor=1)

        self.conv = conv2dAdjoint(features, out_channels,1, padding=0,stride=1,mask_layer=True, compression_factor=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        l,_,_,_ = enc4.shape
        dec4 = torch.cat((torch.cat((dec4[:l//2], enc4[:l//2]),dim=1),torch.cat((dec4[l//2:], enc4[l//2:]),dim=1)),dim=0)
        #dec4 = torch.cat((dec4[:l//2], enc4[:l//2], dec4[l//2:], enc4[l//2:]), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        l,_,_,_ = enc3.shape
        dec3 = torch.cat((torch.cat((dec3[:l//2], enc3[:l//2]),dim=1),torch.cat((dec3[l//2:], enc3[l//2:]),dim=1)),dim=0)
        #dec3 = torch.cat((dec3[:l//2], enc3[:l//2], dec3[l//2:], enc3[l//2:]), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        l,_,_,_ = enc2.shape
        dec2 = torch.cat((torch.cat((dec2[:l//2], enc2[:l//2]),dim=1),torch.cat((dec2[l//2:], enc2[l//2:]),dim=1)),dim=0)
        #dec2 = torch.cat((dec2[:l//2], enc2[:l//2], dec2[l//2:], enc2[l//2:]), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        l,_,_,_ = enc1.shape
        dec1 = torch.cat((torch.cat((dec1[:l//2], enc1[:l//2]),dim=1),torch.cat((dec1[l//2:], enc1[l//2:]),dim=1)),dim=0)
        #dec1 = torch.cat((dec1[:l//2], enc1[:l//2], dec1[l//2:], enc1[l//2:]), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _individualblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels=in_channels, out_channels=features,kernel_size=3,padding=1,bias=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=1,bias=False)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    @staticmethod
    def _firstblock(in_channels, features, name, compression_factor):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", conv2dFirstLayer(in_channels, features,3, stride=1, padding=1)),
                    (name + "norm1", batchNorm(features)),
                    (name + "relu1", nn.ReLU()),
                    (name + "conv2", conv2dAdjoint(features, features,3, stride=1, padding=1, mask_layer=True, compression_factor=compression_factor)),
                    (name + "norm2", batchNorm(features)),
                    (name + "relu2", nn.ReLU()),
                ]
            )
        )

    @staticmethod
    def _block(in_channels, features, name, compression_factor):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", conv2dAdjoint(in_channels, features,3, stride=1, padding=1, mask_layer=True, compression_factor=compression_factor)),
                    (name + "norm1", batchNorm(features)),
                    (name + "relu1", nn.ReLU()),
                    (name + "conv2", conv2dAdjoint(features, features,3, stride=1, padding=1, mask_layer=True, compression_factor=compression_factor)),
                    (name + "norm2", batchNorm(features)),
                    (name + "relu2", nn.ReLU()),
                ]
            )
        )
