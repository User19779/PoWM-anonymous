import numpy as np
import torch
import torch.nn as nn

class ConvRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, init_zero=False):
        super(ConvRelu, self).__init__()
        self.init_zero = init_zero
        if self.init_zero:
            self.layers = nn.Conv2d(channels_in, channels_out, 3, stride, padding=1)

        else:
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
                # nn.LeakyReLU(inplace=True)
                nn.Tanh()
            )

    def forward(self, x):
        return self.layers(x)

class DecodeNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(DecodeNet, self).__init__()

        layers = [ConvRelu(in_channels, out_channels, 2)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvRelu(out_channels, out_channels, 2)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class ParallelDecoder(nn.Module):
    def __init__(self, H=128, message_length=64, channels=32):
        super(ParallelDecoder, self).__init__()
        stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        self.message_layers = nn.ModuleList([nn.Sequential(
            ConvRelu(3, channels),
            # ConvRelu(6, channels),  # for frequency domain
            DecodeNet(channels, channels, blocks=stride_blocks),
            ConvRelu(channels, 1, init_zero=True),
        ) for _ in range(4)])
        self.message_layer = ConvRelu(4, 1)

    def forward(self, message_image):
        messages = [message_layer(message_image) for message_layer in self.message_layers]
        message = torch.cat(messages, dim=1)
        message = self.message_layer(message)
        message = message.view(message.shape[0], -1)
        return message
    
class Decoder(nn.Module):
    def __init__(self, H=128, message_length=64, channels=32):
        super(Decoder, self).__init__()
        stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        self.message_layer = nn.Sequential(
            ConvRelu(3, channels),
            # ConvRelu(6, channels),  # for frequency domain
            DecodeNet(channels, channels, blocks=stride_blocks),
            ConvRelu(channels, 1, init_zero=True),
        )

    def forward(self, message_image):

        message = self.message_layer(message_image)
        message = message.view(message_image.shape[0], -1)
        return message
