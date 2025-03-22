import torch
import torch.nn as nn
import torch.nn.functional as F


class CWCB(nn.Module):
    def __init__(self, in_channels):
        super(CWCB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # Interleaved concatenation
        b, c, h, w = x1.size()
        x = torch.zeros(b, 2 * c, h, w, device=x1.device)
        x[:, 0::2, :, :] = x1
        x[:, 1::2, :, :] = x2

        # Depthwise convolution
        out = self.conv(x)
        return out


class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))

        if inchannel == 64:
            self.conv3 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(mid_channel),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
                                       nn.BatchNorm2d(mid_channel),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))

            self.conv4 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(mid_channel),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),
                                       nn.BatchNorm2d(mid_channel),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))
        elif inchannel == 128:
            self.conv3 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(mid_channel),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
                                       nn.BatchNorm2d(mid_channel),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))
        elif inchannel == 256:
            self.conv3 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))
        elif inchannel == 512:
            self.conv2 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))
            self.conv3 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))
            self.conv4 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                       nn.BatchNorm2d(inchannel),
                                       nn.ReLU(inplace=True))

        self.convmix = nn.Sequential(nn.Conv2d(inchannel * 4, inchannel, 1, stride=1, bias=False),
                                     nn.BatchNorm2d(inchannel),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(inchannel),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x) if self.conv2 else torch.zeros_like(x1)
        x3 = self.conv3(x) if self.conv3 else torch.zeros_like(x1)
        x4 = self.conv4(x) if self.conv4 else torch.zeros_like(x1)

        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.convmix(x_f)

        return out