from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_encoder import *
import numpy as np
from collections import OrderedDict


class ConvBlock(nn.Module):

    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):

    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class DepthDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, verbose=False):
        super(DepthDecoder, self).__init__()

        # The scalar values alpha and beta are used to constrain the estimated depth Z(x) to the range [0.01, 100]
        # units where, x is the output from the sigmoid function:
        #
        #   Disparity:  D(x) = (alpha * x + beta)
        #   Depth:      Z(x) = 1. / D(x)
        #
        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.verbose=verbose

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):

        self.outputs = []

        # decoder
        x = input_features[-1]

        if self.verbose:
            print('[ DepthDecoder ][ Input ] x.shape = ', x.shape)
            print('[ DepthDecoder ][ Outputs ]')

        for i in range(4, -1, -1):

            if self.verbose:
                print('\t[ i = {} ]'.format(i))

            x = self.convs[("upconv", i, 0)](x)

            if self.verbose:
                print('\t\t[ convs[(upconv, {:d}, 0)] ] Shape = {}'.format(i, x.shape))

            x = [upsample(x)]

            if self.verbose:
                print('\t\t[ upsample(x) ] Shape = {}'.format(x[0].shape))

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            if self.verbose:
                print('\t\t[ input_features[{:d}] ] Shape = {}'.format(i-1, input_features[i-1].shape))

            x = torch.cat(x, 1)

            if self.verbose:
                print('\t\t[ torch.cat([ upsample(x), input_features[{:d}] ], 1) ] Shape = {}'.format(i-1, x.shape))

            x = self.convs[("upconv", i, 1)](x)

            if self.verbose:
                print('\t\t[ convs[(upconv, {:d}, 1)] ] Shape = {}'.format(i, x.shape))
                print(' ')

            if i in self.scales:
                self.outputs.append(self.alpha * self.sigmoid(self.convs[("dispconv", i)](x)) + self.beta)

        self.outputs = self.outputs[::-1]

        return self.outputs


class DispResNet(nn.Module):

    def __init__(self, num_layers = 18, pretrained=True, verbose=False):
        super(DispResNet, self).__init__()

        # Encoder network.
        self.encoder = ResnetEncoder(
            num_layers=num_layers,
            pretrained=pretrained,
            num_input_images=1
        )

        # Decoded network.
        self.decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(4),
            num_output_channels=1,
            use_skips=True,
            verbose=verbose,
        )

        # Verbose.
        self.verbose = verbose

    def init_weights(self):
        pass

    def forward(self, x):

        if self.verbose:
            print('[ DispResNet ][ Input ] x.shape = ', x.shape)

        # Extract features from the encoder network given input data (e.g., frames in a video clip).
        features = self.encoder(x)

        if self.verbose:
            Nf = len(features)
            print('[ DispResNet ][ Encoder features ]')
            for i in range(Nf):
                print('\t[ {} of {} ] features[{}].shape = {}'.format(i+1, Nf, i, features[i].shape))

        # Compute the outputs from the decoder network given features.
        outputs = self.decoder(features)

        if self.verbose:
            No = len(outputs)
            print('[ DispResNet ][ Decoder outputs ]')
            for i in range(No):
                print('\t[ {} of {} ] outputs[{}].shape = {}'.format(i+1, No, i, outputs[i].shape))
            print('\n')
        
        if self.training:
            return outputs
        else:
            return outputs[0]


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = DispResNet().cuda()
    model.train()

    B = 12

    tgt_img = torch.randn(B, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(B, 3, 256, 832).cuda() for i in range(2)]

    tgt_depth = model(tgt_img)

    print(tgt_depth[0].size())


