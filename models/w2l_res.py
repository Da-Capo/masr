import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, conv, p):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm1d(conv.out_channels)
        self.proj_bn = nn.BatchNorm1d(conv.out_channels)
        self.act = nn.ReLU6()
        self.dropout = nn.Dropout(p, inplace=True)

    def forward(self, x):
        proj_x = x
        x = self.conv(x)
        x = self.bn(x)+self.proj_bn(proj_x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, conv, p):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm1d(conv.out_channels)
        self.act = nn.ReLU6()
        self.dropout = nn.Dropout(p, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class W2LModule(nn.Module):
    def __init__(self, vocabulary, n_input=161, blank=0, name="w2l"):
        super().__init__()
        self.blank = blank
        self.vocabulary = vocabulary
        self.name = name
        output_units = len(vocabulary)
        modules = []
        modules.append(ConvBlock(nn.Conv1d(n_input, 256, 11, 2, 97), 0.8))

        for i in range(3):
            modules.append(ResBlock(nn.Conv1d(256, 256, 11, 1), 0.8))

        modules.append(ConvBlock(nn.Conv1d(256, 384, 17, 1), 0.7))

        for i in range(3):
            modules.append(ResBlock(nn.Conv1d(384, 384, 17, 1), 0.7))

        modules.append(ConvBlock(nn.Conv1d(384, 512, 25, 1), 0.6))

        for i in range(3):
            modules.append(ResBlock(nn.Conv1d(512, 512, 25, 1), 0.6))


        modules.append(ConvBlock(nn.Conv1d(512, 1024, 29, 1), 0.5))
        
        modules.append(ConvBlock(nn.Conv1d(1024, 1024, 1, 1), 0.5))

        modules.append(nn.Conv1d(1024, output_units, 1, 1))

        self.cnn = nn.Sequential(*modules)

    def forward(self, x, lens):  # -> B * V * T
        x = self.cnn(x)
        for module in self.modules():
            if type(module) == nn.modules.Conv1d:
                lens = (
                    lens - module.kernel_size[0] + 2 * module.padding[0]
                ) // module.stride[0] + 1
        return x, lens
