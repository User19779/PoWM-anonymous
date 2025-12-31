import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import deform_conv2d


class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True, act_func=nn.Tanh()):
        super(ResidualDenseBlock_out, self).__init__()
        # input_channels32
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 32, output, 3, 1, 1, bias=bias)
        # self.conv4 = nn.Conv2d(input + 3 * 32,
        #                        output, 3, 1, 1, bias=bias)
        self.act_func = act_func  # ELU

    def forward(self, x):
        # 
        x1 = self.act_func(self.conv1(x))
        x2 = self.act_func(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act_func(self.conv3(torch.cat((x, x2), 1)))

        return x3


# Edit 0603
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(DeformableConvBlock, self).__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(
            in_channels, offset_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        offset = self.conv_offset(x)
        x = deform_conv2d(input=x, offset=offset, weight=self.conv.weight, bias=self.conv.bias, stride=self.conv.stride,
                          padding=self.conv.padding, dilation=self.conv.dilation,)
        return x


class AttentionDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(AttentionDenseBlock_out, self).__init__()
        self.se_block = SELayer(input)
        self.deformable_conv = DeformableConvBlock(input, input, bias=bias)

        M = 32
        self.conv1 = nn.Conv2d(input,   M, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(2 * input + M, M, 3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(2 * input + 2*M, M, 3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(2 * input + 3*M, output,
                               3, padding=1, bias=bias)

    def forward(self, x):
        identity = x

        x = self.se_block(x)

        x_deform = self.deformable_conv(x)
        x = torch.cat([identity, x_deform], dim=1)
        #  x  channel  2*input

        x1 = self.conv1(x_deform)
        x = torch.cat([x, x1], dim=1)
        x2 = self.conv2(x)
        x = torch.cat([x, x2], dim=1)
        x3 = self.conv3(x)
        x = torch.cat([x, x3], dim=1)
        x4 = self.conv4(x)
        return x4


class AttentionDenseBlock_out_simple(nn.Module):
    def __init__(self, input, output, bias=True):
        super(AttentionDenseBlock_out_simple, self).__init__()
        self.se_block = SELayer(input)
        self.deformable_conv = DeformableConvBlock(input, output, bias=bias)

        M = 32
        # self.conv1 = nn.Conv2d(input, output, 3, padding=1, bias=bias)

        self.act_1 = torch.nn.Tanh()
        self.act_2 = torch.nn.Tanh()

    def forward(self, x):
        res = x
        x = self.se_block(x)

        x = self.deformable_conv(x)
        x = self.act_1.forward(x)

        return x


class AttentionDenseBlock_out_UNet(nn.Module):
    def __init__(self, input, output, bias=True):
        super(AttentionDenseBlock_out_UNet, self).__init__()

        # 
        self.encoder_conv1: nn.Conv2d = nn.Conv2d(
            input, 16, kernel_size=3, stride=1, padding=1)
        self.encoder_tanh1: nn.Tanh = nn.Tanh()
        self.encoder_pool1: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=2, stride=2)

        self.encoder_conv2: nn.Conv2d = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1)
        self.encoder_tanh2: nn.Tanh = nn.Tanh()
        self.encoder_pool2: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=2, stride=2)

        # 
        self.decoder_upsample1: nn.ConvTranspose2d = nn.ConvTranspose2d(
            32, 32, kernel_size=2, stride=2)

        self.decoder_conv1: nn.Conv2d = nn.Conv2d(
            64, 16, kernel_size=3, stride=1, padding=1)
        self.decoder_tanh1: nn.Tanh = nn.Tanh()

        self.decoder_upsample2: nn.ConvTranspose2d = nn.ConvTranspose2d(
            16, 16, kernel_size=2, stride=2)

        self.decoder_conv2: nn.Conv2d = nn.Conv2d(
            32, output, kernel_size=3, stride=1, padding=1)
        self.decoder_tanh2: nn.Tanh = nn.Tanh()

    def forward(self, x):

        res = x
        x1 = self.encoder_tanh1(self.encoder_conv1(x))
        x1_pooled = self.encoder_pool1(x1)

        x2 = self.encoder_tanh2(self.encoder_conv2(x1_pooled))
        x2_pooled = self.encoder_pool2(x2)

        # 
        x3 = self.decoder_upsample1(x2_pooled)
        x3_concat = torch.concat(tensors=(x3, x2), dim=1)
        x3 = self.decoder_tanh1(self.decoder_conv1(x3_concat))

        x4 = self.decoder_upsample2(x3)
        x4_concat = torch.concat(tensors=(x4, x1), dim=1)
        x4 = self.decoder_tanh2(self.decoder_conv2(x4_concat))

        # 
        x = res + x4
        return x
