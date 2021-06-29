import torch
import torch.nn as nn

__all__ = ['ComplexReLu', 'ComplexLeakyReLu',
           'ComplexConv', 'ComplexConvTranspose', 'ComplexDepthwiseSeparableConv']


class ComplexReLu(nn.Module):
    '''
    A PyTorch module to apply relu activation on the magnitude of the signal. Phase is preserved
    '''
    def __init__(self):
        super(ComplexReLu, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        mag = torch.abs(input)
        return torch.nn.functional.relu(mag).type(torch.complex64) / (mag + 1e-6) * input


class ComplexLeakyReLu(nn.Module):
    '''
    A PyTorch module to apply leaky relu activation on the magnitude of the signal. Phase is preserved

    Args:
        negative_slope (float): The slope of the negative section of the relu

    '''

    def __init__(self, negative_slope=0.1):
        super(ComplexLeakyReLu, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        mag = torch.abs(input)
        return torch.nn.functional.leaky_relu(mag,
                                              negative_slope=self.negative_slope
                                              ).type(torch.complex64) / (mag + 1e-6) * input


def apply_complex(fr, fi, input, dtype=torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


class ComplexConv(nn.Module):
    '''
    This convolution supporting complex inputs and complex kernels and 2D or 3D convolutions.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all six sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'``
            or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        complex_kernel (bool, optional): If ``True`` declares the kernel as complex and applies it so. Otherwise
            the kernel will be real valued. Default: ``True``
        ndims (int, optional): 2 or 3 specifying 2D or 3D convolutions. Default: 2
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, groups=1, bias=False, complex_kernel=False, ndims=2):
        super(ComplexConv, self).__init__()
        self.complex_kernel = complex_kernel

        if ndims == 2:
            self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                    padding_mode=padding_mode)
            if complex_kernel:
                self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                        padding_mode=padding_mode)
        elif ndims == 3:
            self.conv_r = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                    padding_mode=padding_mode)
            if complex_kernel:
                self.conv_i = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                                        padding_mode=padding_mode)
        else:
            raise ValueError(f'Convolutions must be 2D or 3D passed {ndims}')

    def forward(self, input):
        if self.complex_kernel:
            return apply_complex(self.conv_r, self.conv_i, input)
        else:
            return self.conv_r(input.real) + 1j * self.conv_r(input.imag)


class ComplexConvTranspose(nn.Module):
    '''

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        complex_kernel (bool, optional): If ``True`` declares the kernel as complex and applies it so. Otherwise
            the kernel will be real valued. Default: ``True``
        ndims (int, optional): 2 or 3 specifying 2D or 3D convolutions. Default: 2

    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros',
                 complex_kernel=False, ndims=2):
        super(ComplexConvTranspose, self).__init__()

        self.complex_kernel = complex_kernel

        if ndims == 2:
            self.conv_r = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                             output_padding, groups, bias, dilation, padding_mode)
            if self.complex_kernel:
                self.conv_i = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)
        elif ndims == 3:
            self.conv_r = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding,
                                             output_padding, groups, bias, dilation, padding_mode)
            if self.complex_kernel:
                self.conv_i = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding,
                                                 output_padding, groups, bias, dilation, padding_mode)
        else:
            raise ValueError(f'Convolution transpose must be 2D or 3D passed {ndims}')

    def forward(self, input):
        if self.complex_kernel:
            return apply_complex(self.conv_r, self.conv_i, input)
        else:
            return self.conv_r(input.real) + 1j * self.conv_r(input.imag)


class ComplexDepthwiseSeparableConv(nn.Module):
    '''

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        complex_kernel (bool, optional): If ``True`` declares the kernel as complex and applies it so. Otherwise
            the kernel will be real valued. Default: ``True``
        ndims (int, optional): 2 or 3 specifying 2D or 3D convolutions. Default: 2

    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, groups=1, bias=False, complex_kernel=False, ndims=2):
        super(ComplexDepthwiseSeparableConv, self).__init__()

        self.depthwise = ComplexConv(in_channels, out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     padding_mode=padding_mode,
                                     dilation=dilation,
                                     groups=in_channels,
                                     bias=bias,
                                     complex_kernel=complex_kernel,
                                     ndims=ndims)

        self.pointwise = ComplexConv(in_channels, out_channels,
                                     kernel_size=1,
                                     stride=stride,
                                     padding=padding,
                                     padding_mode=padding_mode,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias,
                                     complex_kernel=complex_kernel,
                                     ndims=ndims)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out