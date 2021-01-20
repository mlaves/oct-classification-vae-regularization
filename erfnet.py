# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock (nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)
        
        return F.relu(output+input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=True):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class Encoder2(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channels, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 128))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(128, 0.03, 1))

        self.layers.append(DownsamplerBlock(128, 256))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(256, 0.3, 2))
            self.layers.append(non_bottleneck_1d(256, 0.3, 4))
            self.layers.append(non_bottleneck_1d(256, 0.3, 8))
            self.layers.append(non_bottleneck_1d(256, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(256, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=True):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        # self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(UpsamplerBlock(64, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, out_channels, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class Decoder2 (nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(64, 64))
        self.layers.append(UpsamplerBlock(64, 64))
        self.layers.append(UpsamplerBlock(64, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, out_channels, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class Decoder3 (nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(64, 256))
        self.layers.append(non_bottleneck_1d(256, 0, 1))
        self.layers.append(non_bottleneck_1d(256, 0, 1))
        self.layers.append(non_bottleneck_1d(256, 0, 1))

        self.layers.append(UpsamplerBlock(256, 128))
        self.layers.append(non_bottleneck_1d(128, 0, 1))
        self.layers.append(non_bottleneck_1d(128, 0, 1))
        self.layers.append(non_bottleneck_1d(128, 0, 1))

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, out_channels, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ERFNet
class ErfNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, out_channels=1, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if encoder == None:
            self.encoder = Encoder(num_classes, in_channels=in_channels)
        else:
            self.encoder = encoder
        self.decoder = Decoder(out_channels=out_channels)

    def init_weights(self):
        """ init weights by xavier method """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)
