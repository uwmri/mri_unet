import unittest
from mri_unet import unet
import torch
from torchsummary import summary

if __name__ == '__main__':
    unittest.main()


class TestMRIUNet(unittest.TestCase):

    def test_unet(self):

        in_channels = 8
        out_channels = 2

        for complex_kernel in [True, False]:
            for complex_input in [True, False]:
                for ndims in [2, 3]:
                    model = unet.UNet(in_channels,
                                      out_channels,
                                      f_maps=64,
                                      layer_order='cr',
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
