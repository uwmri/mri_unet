import torch
import mri_unet

from mri_unet import *
from torchsummary import summary

in_channels = 8
out_channels = 2

for complex_kernel in [True, False]:
    for complex_input in [True, False]:
        for ndims in [2,3]:
            model = MRI_UNet(in_channels,
                             out_channels,
                             f_maps=64,
                             layer_order='cr',
                             num_groups=0,
                             depth=2,
                             layer_growth=2.0,
                             residual=True,
                             complex_input=complex_input,
                             complex_kernel=complex_kernel,
                             ndims=ndims)
            model.cuda()

            input_size = (in_channels,)
            for d in range(ndims):
                input_size += (128,)

            if complex_input:
                summary(model, input_size, dtypes=[torch.complex64, ])
            else:
                summary(model, input_size, dtypes=[torch.float32, ])
