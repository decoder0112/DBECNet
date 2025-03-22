import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.DFEM import DFEM
from models.Fusion import Diff
from models.CWCB import CWCB
from models.CWCB import MSFF
from models.MDEM import DEM
from models.resnet import resnet18


class HSSNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(HSSNet, self).__init__()
        self.backbone = resnet18(pretrained=True)  # 使用resnet18

        self.mid_d = 64

        # DFEM modules
        self.DFEM5 = DFEM(512)  # 512 8 8
        self.DFEM4 = DFEM(256)  # 256 16 16
        self.DFEM3 = DFEM(128)  # 128 32 32
        self.DFEM2 = DFEM(64)  # 64 64 64

        self.DEM5 = DEM(512, 512)
        self.DEM4 = DEM(256, 256)
        self.DEM3 = DEM(128, 128)
        self.DEM2 = DEM(64, 64)

        self.CWCB5 = CWCB(512)
        self.CWCB4 = CWCB(256)
        self.CWCB3 = CWCB(128)
        self.CWCB2 = CWCB(64)

        self.MSFF5 = MSFF(512, 512)
        self.MSFF4 = MSFF(256, 256)
        self.MSFF3 = MSFF(128, 128)
        self.MSFF2 = MSFF(64, 64)

        # DIFF
        self.Diff5 = Diff(512)
        self.Diff4 = Diff(256)
        self.Diff3 = Diff(128)
        self.Diff2 = Diff(64)


        # 解码器部分
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.channels_cut5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.channels_cut4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.channels_cut3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)

        # DFEM
        W1, H1 = self.DFEM5(x1_5, x2_5)  # 1/32
        W2, H2 = self.DFEM4(x1_4, x2_4)  # 1/16
        W3, H3 = self.DFEM3(x1_3, x2_3)  # 1/8
        W4, H4 = self.DFEM2(x1_2, x2_2)  # 1/4

        S1 = self.DEM5(W1)  # 1/32
        S2 = self.DEM4(W2)  # 1/16
        S3 = self.DEM3(W3)  # 1/8
        S4 = self.DEM2(W4)  # 1/4
        Q1 = self.DEM5(H1)  # 1/32
        Q2 = self.DEM4(H2)  # 1/16
        Q3 = self.DEM3(H3)  # 1/8
        Q4 = self.DEM2(H4)  # 1/4

        C1 = self.CWCB5(x1_5, x2_5)
        C2 = self.CWCB4(x1_4, x2_4)
        C3 = self.CWCB3(x1_3, x2_3)
        C4 = self.CWCB2(x1_2, x2_2)

        G1 = self.MSFF5(C1)
        G2 = self.MSFF4(C2)
        G3 = self.MSFF3(C3)
        G4 = self.MSFF2(C4)

        K1 = self.Diff5(G1, S1, Q1)
        K2 = self.Diff4(G2, S2, Q2)
        K3 = self.Diff3(G3, S3, Q3)
        K4 = self.Diff2(G4, S4, Q4)

        # 解码器部分
        D1 = torch.cat((self.upconv5(K1), K2), dim=1)
        D1 = self.channels_cut5(D1)
        D2 = torch.cat((self.upconv4(D1), K3), dim=1)
        D2 = self.channels_cut4(D2)
        D3 = torch.cat((self.upconv3(D2), K4), dim=1)
        D3 = self.channels_cut3(D3)
        mask = F.interpolate(D3, x1.size()[2:], mode='bilinear', align_corners=True)
        output = self.final_conv(mask)
        output = torch.sigmoid(output)
        return output


if __name__ == '__main__':
    x1 = torch.randn((4, 3, 256, 256)).cuda()
    x2 = torch.randn((4, 3, 256, 256)).cuda()
    model = HSSNet(3, 1).cuda()
    out = model(x1, x2)
    print(out.shape)
