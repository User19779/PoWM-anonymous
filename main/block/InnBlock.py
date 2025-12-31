import torch
import torch.nn as nn
from main.block.DenseNet import ResidualDenseBlock_out as DenseBlock
from main.block.DenseNet import AttentionDenseBlock_out_simple as AttDenseBlock
# from main.block.DenseNet import AttentionDenseBlock_out_UNet as AttDenseBlock
from main.block.Encoder import Encoder, ParallelEncoder
from main.block.Decoder import Decoder, ParallelDecoder
# import config as c


class Noise_INN_block(nn.Module):
    def __init__(self, clamp: float = 2.0, input_1: int = 3, input_2: int = 9):
        super().__init__()

        self.clamp = clamp
        self.r = DenseBlock(input=input_1, output=input_2)
        self.y = DenseBlock(input=input_1, output=input_2)
        self.f = DenseBlock(input=input_2, output=input_1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = x[0], x[1]
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2

            s1, t1 = self.r(y1), self.y(y1)

            y2 = torch.exp(s1) * x2 + t1
            out = [y1, y2]
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / torch.exp(s1)

            t2 = self.f(y2)
            y1 = x1 - t2
            out = [y1, y2]
        return out


class attention_INN_block(nn.Module):
    def __init__(self, clamp: float = 2.0, input_1: int = 3, input_2: int = 9):
        super().__init__()

        self.clamp = clamp
        self.r = AttDenseBlock(input=input_1, output=input_2)
        self.y = AttDenseBlock(input=input_1, output=input_2)
        self.f = AttDenseBlock(input=input_2, output=input_1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = x[0], x[1]
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2

            s1, t1 = self.r(y1), self.y(y1)

            y2 = torch.exp(s1) * x2 + t1
            out = [y1, y2]
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / torch.exp(s1)

            t2 = self.f(y2)
            y1 = x1 - t2
            out = [y1, y2]
        return out
