import torch
import mri_unet

from mri_unet import *

in_channels = 8
out_channels = 2

model2 = MRI_UNet(in_channels,
                 out_channels,
                 f_maps=64,
                 layer_order='cr',
                 num_groups=0,
                 depth=2,
                 layer_growth=2.0,
                 residual=True,
                 ndims=2)

model3 = MRI_UNet(in_channels,
                 out_channels,
                 f_maps=22,
                 layer_order='cr',
                 num_groups=0,
                 depth=2,
                 layer_growth=2.0,
                 residual=True,
                 ndims=3)

model2.cuda()
model3.cuda()

from torchsummary import summary
summary(model2, (in_channels, 128, 128), batch_size=1)
summary(model3, (in_channels, 128, 128, 128), batch_size=1)
