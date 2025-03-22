import torch
import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DFEM(nn.Module):
    def __init__(self, inchannels):
        super(DFEM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(inchannels, inchannels // 2),
            nn.ReLU(),
            nn.Linear(inchannels // 2, inchannels)
        )
        self.conv1x1 = DWConv(inchannels, inchannels)
        self.conv1x1_out = DWConv(inchannels * 2, inchannels)
        self.alpha = nn.Parameter(torch.ones(1, inchannels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, inchannels, 1, 1))

    def forward(self, X1, X2):
        # Compute difference
        diff = torch.abs(X1 - X2)

        # Path for diff (Avg pool)
        avgpool_diff = self.avgpool(diff)
        avgpool_diff = avgpool_diff.view(avgpool_diff.size(0), -1)  # 调整形状为 (batch_size, channels)
        mlp_avgpool_diff = self.mlp(avgpool_diff).view(avgpool_diff.size(0), -1, 1,
                                                       1)  # 确保形状为 (batch_size, channels, 1, 1)
        conv_avgpool_diff = self.conv1x1(mlp_avgpool_diff)

        # Path for diff (Max pool)
        maxpool_diff = self.maxpool(diff)
        maxpool_diff = maxpool_diff.view(maxpool_diff.size(0), -1)  # 调整形状为 (batch_size, channels)
        mlp_maxpool_diff = self.mlp(maxpool_diff).view(maxpool_diff.size(0), -1, 1,
                                                       1)  # 确保形状为 (batch_size, channels, 1, 1)
        conv_maxpool_diff = self.conv1x1(mlp_maxpool_diff)

        # Combine paths
        combined_X1 = self.alpha * conv_avgpool_diff + self.beta * conv_maxpool_diff
        combined_X2 = self.alpha * conv_maxpool_diff + self.beta * conv_avgpool_diff

        # Element-wise multiplication and addition
        Y1 = X1 * combined_X1 + X1
        Y2 = X2 * combined_X2 + X2

        return Y1, Y2
