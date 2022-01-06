import unittest
from mri_unet import unet
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    unittest.main()


class TestMRIUNet(unittest.TestCase):

    def test_unet(self):

        in_channels = 4
        out_channels = 4
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        for padding in [0, 1]:
            for complex_kernel in [True, False]:
                for complex_input in [True, False]:
                    for ndims in [2, 3]:
                        model = unet.UNet(in_channels,
                                          out_channels,
                                          f_maps=64,
                                          layer_order=['convolution', 'mod relu'],
                                          depth=3,
                                          layer_growth=2.0,
                                          residual=True,
                                          complex_input=complex_input,
                                          complex_kernel=complex_kernel,
                                          ndims=ndims,
                                          padding=padding)
                        model.to(device)

                        input_size = (in_channels,)
                        for d in range(ndims):
                            if padding == 1:
                                input_size += (80,)
                            else:
                                input_size += (120,)

                        for a in range(64, 192, 2):
                            if model.check_spatial_dimensions([a, a, a]):
                                print(f'Input valid = {a}')

                        # if complex_input:
                        #     summary(model, input_size, dtypes=[torch.complex64, ])
                        # else:
                        #     summary(model, input_size, dtypes=[torch.float32, ])

                        # Test a forward pass with rand
                        if complex_input:
                            input = torch.randn( (1,) + input_size, dtype=torch.complex64, device=device)
                        else:
                            input = torch.randn((1,) + input_size, dtype=torch.float32, device=device)
                        output = model(input)
                        loss = torch.mean(torch.abs(F.pad(input, model.output_pad) - output))
                        print(f'{output.shape} {loss.detach().item()}')

