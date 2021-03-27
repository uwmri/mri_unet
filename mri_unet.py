import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MRI_UNet']

def conv2d(in_channels, out_channels, kernel_size, bias=False, padding=1, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)

def conv3d(in_channels, out_channels, kernel_size, bias=False, padding=1, groups=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)

class depthwise_separable_conv2d(nn.Module):
    def __init__(self, nin, nout, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class depthwise_separable_conv3d(nn.Module):
    def __init__(self, nin, nout, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin, kernel_size=3, padding=1, groups=nin, bias=bias)
        self.pointwise = nn.Conv3d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class VarNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.05, affine=True, track_running_stats=True):
        super(VarNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            var = self.running_var

        input = input / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None]

        return input

class VarNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.05, affine=True, track_running_stats=True):
        super(VarNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            var = self.running_var

        input = input / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None]

        return input


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1, ndims=2):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append((f'ReLU{i}', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append((f'LeakyReLU{i}', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append((f'ELU{i}', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            # bias = not ('g' in order or 'b' in order)
            # bias = not ('g' in order)
            if ndims == 2:
                modules.append((f'conv{i}', conv2d(in_channels, out_channels, kernel_size, bias=False, padding=padding)))
            else:
                modules.append((f'conv{i}', conv3d(in_channels, out_channels, kernel_size, bias=False, padding=padding)))
            in_channels = out_channels
        elif char == 'C':
            if ndims == 2:
                modules.append((f'conv{i}', depthwise_separable_conv2d(in_channels, out_channels, bias=False)))
            else:
                modules.append((f'conv{i}', depthwise_separable_conv3d(in_channels, out_channels, bias=False)))

            in_channels = out_channels
        elif char == 'i':
            if ndims == 2:
                modules.append((f'instancenorm{i}', nn.InstanceNorm2d(out_channels)))
            else:
                modules.append((f'instancenorm{i}', nn.InstanceNorm3d(out_channels)))
        elif char == 'g':
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append((f'groupnorm{i}', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            if ndims == 2:
                modules.append((f'batchnorm{i}', nn.BatchNorm2d(out_channels)))
            else:
                modules.append((f'batchnorm{i}', nn.BatchNorm3d(out_channels)))
        elif char == 'v':
            if ndims ==2:
                modules.append((f'varnorm{i}', VarNorm2d(out_channels)))
            else:
                modules.append((f'varnorm{i}', VarNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'C', 'i', v']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv2d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1, ndims=2):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding, ndims=ndims):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm2d+ReLU+Conv2d).
    We use (Conv2d+ReLU+GroupNorm2d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv2d+BatchNorm2d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8, padding=1, ndims=2):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups, padding, ndims))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups, padding, ndims))


class ResBottle(nn.Module):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm2d+ReLU+Conv2d).
    We use (Conv2d+ReLU+GroupNorm2d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv2d+BatchNorm2d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8, padding=1):
        super(ResBottle, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.conv1 = SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, 'crb', num_groups, padding=1)
        # conv2
        self.conv2 = SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, 'c', num_groups, padding=1)

        # Shortcut
        self.convshortcut = SingleConv(conv1_in_channels, conv2_out_channels, 1, 'c',  num_groups, padding=0)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.convshortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += shortcut
        x = self.activation(x)

        return x

class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool2d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, basic_module=DoubleConv, downsample=True, conv_layer_order='crb',
                 num_groups=8, scale_factor=2, ndims=2 ):
        super(Encoder, self).__init__()

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         ndims=ndims)
        if downsample:
            if ndims == 2:
                self.downsample = nn.Conv2d(in_channels,
                                                   in_channels,
                                                   kernel_size=2,
                                                   stride=scale_factor,
                                                   padding=0, bias=False, groups=in_channels)
            else:
                self.downsample = nn.Conv3d(in_channels,
                                                   in_channels,
                                                   kernel_size=2,
                                                   stride=scale_factor,
                                                   padding=0, bias=False, groups=in_channels)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        x = self.downsample(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose2d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose2d, must reverse the MaxPool2d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, add_features, kernel_size=3,
                 scale_factor=2, basic_module=DoubleConv, conv_layer_order='crg', num_groups=8, ndims=2):
        super(Decoder, self).__init__()

        if ndims == 2:
            self.upsample = nn.ConvTranspose2d(in_channels,
                                               in_channels,
                                               kernel_size=2,
                                               stride=scale_factor,
                                               padding=0,
                                               output_padding=0,
                                               bias=False, groups=in_channels)
        else:
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               in_channels,
                                               kernel_size=2,
                                               stride=scale_factor,
                                               padding=0,
                                               output_padding=0,
                                               bias=False, groups=in_channels)

        self.basic_module = basic_module(in_channels + add_features, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         ndims=ndims)

    def forward(self, encoder_features, x):
        # use ConvTranspose2d and summation joining
        print(x.shape)
        x = self.upsample(x)
        print(encoder_features.shape)
        print(x.shape)

        x = torch.cat([encoder_features,x], dim=1)
        x = self.basic_module(x)
        return x

def create_feature_maps(init_channel_number, number_of_fmaps, growth_rate=2.0 ):

    channel_number = init_channel_number
    fmaps = []
    fmaps.append(init_channel_number)
    for k in range(number_of_fmaps-1):
        channel_number = int( channel_number * growth_rate )
        fmaps.append(channel_number)

    return fmaps

class MRI_UNet(nn.Module):
    """
    2DUnet model from
    `"2D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_activation (string): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 f_maps=64,
                 layer_order='crb',
                 num_groups=0,
                 depth=4,
                 layer_growth=2.0,
                 residual=True,
                 ndims=2, **kwargs):

        super(MRI_UNet, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=depth, growth_rate=layer_growth)

        print(f_maps)

        self.residual = residual

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, downsample=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups, ndims=ndims)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups, ndims=ndims)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            add_feature_num = reversed_f_maps[i + 1] # features from past layer
            out_feature_num = reversed_f_maps[i + 1] # features from past layer
            decoder = Decoder(in_feature_num, out_feature_num, add_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups, ndims=ndims)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        if ndims == 2:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1, bias=False)
        else:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1, bias=False)

        # To handle cases when in channels does not equal output channels
        if out_channels != in_channels:
            if ndims == 2:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            else:
                self.residual_conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):

        # Keep x
        input = x

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # Keep skip to end and also downweight to help training
        if self.residual:
            x += self.residual_conv(input)

        return x



