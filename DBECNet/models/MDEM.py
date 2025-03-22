import torch
import torch.nn as nn
import torch.nn.functional as F

class DEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DEM, self).__init__()
        self.dwconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dwconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dwconv3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.gn3 = nn.GroupNorm(32, in_channels)
        self.conv3x3 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # 修改为输入通道数为in_channels
        self.adjust_channels = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)  # 新增的1x1卷积层

    def forward(self, x):
        device = x.device  # 获取输入张量的设备
        # 第一条支路
        out1 = F.relu(self.bn1(self.dwconv1(x)))

        # 第二条支路
        n, c, h, w = x.size()
        out2 = F.relu(nn.LayerNorm([c, h, w]).to(device)(self.dwconv2(x)))

        # 第三条支路
        x_1d = x.view(n, c, -1)
        out3 = F.relu(self.gn3(self.dwconv3(x_1d)))
        out3 = out3.view(n, c, h, w)

        # 拼接
        out = torch.cat((out1, out2, out3), dim=1)

        # 调整通道数
        out = self.adjust_channels(out)

        # 逐元素相乘
        out = out * x

        # Softmax
        out = F.softmax(out, dim=1)

        out = self.conv3x3(out)  # 修改为2D卷积

        return out
# 创建模型实例
in_channels = 512
out_channels = 512
model = DEM(in_channels, out_channels).to('cuda')  # 将模型移动到GPU上

# 输入特征图
input_tensor = torch.randn(1, in_channels, 8, 8).to('cuda')  # 将输入张量移动到GPU上

# 前向传播
output = model(input_tensor)
print(output.shape)  # 输出特征图大小