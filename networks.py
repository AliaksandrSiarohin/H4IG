import torch.nn.functional as F
from torch import nn


class DCGenerator(nn.Module):
    def __init__(self, block_expansion, dim_z, **kwargs):
        super(DCGenerator, self).__init__()

        # 4 x 4
        self.fc = nn.Linear(dim_z, (8 * block_expansion) * 4 * 4)
        self.conv0 = nn.Conv2d(8 * block_expansion, 8 * block_expansion, kernel_size=(3, 3), padding=(1, 1))
        self.bn0 = nn.BatchNorm2d(8 * block_expansion)
        # 8 x 8
        self.conv1 = nn.Conv2d(8 * block_expansion, 4 * block_expansion, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(4 * block_expansion)
        # 16 x 16
        self.conv2 = nn.Conv2d(4 * block_expansion, 2 * block_expansion, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(2 * block_expansion)
        # 32 x 32
        self.conv3 = nn.Conv2d(2 * block_expansion, block_expansion, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(block_expansion)
        # 32 x 32
        self.conv4 = nn.Conv2d(block_expansion, 3, kernel_size=(3, 3), padding=(1, 1))

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input, label):
        out = self.fc(input)
        out = out.view(input.shape[0], -1, 4, 4)
        # 4 x 4
        out = self.conv0(out)
        out = self.bn0(out)
        out = F.relu(out, True)
        out = self.upsample(out)
        # 8 x 8
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.upsample(out)
        # 16 x 16
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.upsample(out)
        # 32 x 32
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv4(out)

        out = F.tanh(out)

        return out


class DCDiscriminator(nn.Module):
    def __init__(self, block_expansion, sn=False, **kwargs):
        super(DCDiscriminator, self).__init__()

        # 32 x 32
        self.conv0 = nn.Conv2d(3, block_expansion, kernel_size=(3, 3), padding=(1, 1))
        if sn:
            self.conv0 = nn.utils.spectral_norm(self.conv0)
        # 32 x 32
        self.conv1 = nn.Conv2d(block_expansion, block_expansion * 2, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        if sn:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        # 16 x 16
        self.conv2 = nn.Conv2d(block_expansion * 2, block_expansion * 4, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        if sn:
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        # 8 x 8
        self.conv3 = nn.Conv2d(block_expansion * 4, block_expansion * 8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        if sn:
            self.conv3 = nn.utils.spectral_norm(self.conv3)
        # 4 x 4
        self.fc = nn.Linear(block_expansion * 8, 1)
        if sn:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input, label):
        # 32 x 32
        out = self.conv0(input)
        out = F.leaky_relu(out, 0.2)
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)

        # 16 x 16
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)

        # 8 x 8
        out = self.conv3(out)
        out = F.leaky_relu(out, 0.2)

        # 4 x 4
        out = out.mean(dim=(2, 3))
        out = self.fc(out)

        return out


class ShiftInvariantDiscriminator(nn.Module):
    def __init__(self, block_expansion, sn=False, **kwargs):
        super(ShiftInvariantDiscriminator, self).__init__()

        # 48 x 48, receptive field 3 x 3
        self.conv0 = nn.Conv2d(3, block_expansion, kernel_size=(3, 3))
        if sn:
            self.conv0 = nn.utils.spectral_norm(self.conv0)
        # 46 x 46, receptive field 5 x 5
        self.conv1 = nn.Conv2d(block_expansion, block_expansion * 2, kernel_size=(3, 3))
        if sn:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
        # 44 x 44, receptive field 9 x 9
        self.conv2 = nn.Conv2d(block_expansion * 2, block_expansion * 4, kernel_size=(3, 3), dilation=(2, 2))
        if sn:
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        # 40 x 40, receptive field 17 x 17
        self.conv3 = nn.Conv2d(block_expansion * 4, block_expansion * 8, kernel_size=(3, 3), dilation=(4, 4))
        if sn:
            self.conv3 = nn.utils.spectral_norm(self.conv3)
        # 32 x 32
        self.fc = nn.Linear(block_expansion * 8, 1)
        if sn:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input, label):
        # 48 x 48
        out = F.pad(input, pad=(8, 8, 8, 8), value=-1)

        out = self.conv0(input)
        out = F.leaky_relu(out, 0.2)
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)

        # 44 x 44
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)

        # 40 x 40
        out = self.conv3(out)
        out = F.leaky_relu(out, 0.2)

        # 32 x 32
        out = out.mean(dim=(2, 3))
        out = self.fc(out)

        return out
