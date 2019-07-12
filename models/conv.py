import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ConvBlock(nn.Module):
    def __init__(self, conv, p):
        super().__init__()
        self.conv = conv
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv = weight_norm(self.conv)
        self.act = nn.GLU(1)
        self.dropout = nn.Dropout(p, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class GatedConv(nn.Module):
    def __init__(self, vocabulary, blank=0, name="masr"):
        super().__init__()
        self.blank = blank
        self.vocabulary = vocabulary
        self.name = name
        output_units = len(vocabulary)
        modules = []
        modules.append(ConvBlock(nn.Conv1d(161, 500, 48, 2, 97), 0.2))

        for i in range(7):
            modules.append(ConvBlock(nn.Conv1d(250, 500, 7, 1), 0.3))

        modules.append(ConvBlock(nn.Conv1d(250, 2000, 32, 1), 0.5))

        modules.append(ConvBlock(nn.Conv1d(1000, 2000, 1, 1), 0.5))

        modules.append(weight_norm(nn.Conv1d(1000, output_units, 1, 1)))

        self.cnn = nn.Sequential(*modules)

    def forward(self, x, lens):  # -> B * V * T
        x = self.cnn(x)
        for module in self.modules():
            if type(module) == nn.modules.Conv1d:
                lens = (
                    lens - module.kernel_size[0] + 2 * module.padding[0]
                ) // module.stride[0] + 1
        return x, lens
