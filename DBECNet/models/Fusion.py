import torch
import torch.nn as nn
import torch.nn.functional as F

class Diff(nn.Module):
    def __init__(self, in_dim):
        super(Diff, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, F3, F1, F2):
        m_batchsize, C, height, width = F1.size()

        # Use F3 as Q
        query = self.query_conv(F3)
        proj_query = query.view(m_batchsize, -1, width * height).permute(0, 2, 1)

        # Use F1 as K
        key = self.key_conv(F1)
        proj_key = key.view(m_batchsize, -1, width * height)

        # Use F2 as V
        value = self.value_conv(F2)
        proj_value = value.view(m_batchsize, -1, width * height)

        # Scaled dot-product attention
        energy = torch.bmm(proj_query, proj_key) / (C // 8) ** 0.5
        attention = self.softmax(energy)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + F3
        return out
