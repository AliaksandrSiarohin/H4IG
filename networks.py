import torch.nn.functional as F
from torch import nn
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4MConvP4M, P4ConvP4, P4MConvZ2


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
    def __init__(self, block_expansion, sn=False, shift_invariance=False, rot_invariance=False,
                 flip_invariance=False, **kwargs):
        super(DCDiscriminator, self).__init__()

        self.rot_invariance = rot_invariance
        self.flip_invariance = flip_invariance
        self.shift_invariance = shift_invariance
        self.sn = sn

        assert self.rot_invariance or not self.flip_invariance

        if self.shift_invariance and self.rot_invariance and self.flip_invariance:
            block_expansion //= int(8 ** 0.5)
            self.conv0 = P4MConvZ2(3, block_expansion, kernel_size=3)
            self.conv1 = P4MConvP4M(block_expansion, block_expansion * 2, kernel_size=3)
            self.conv2 = P4MConvP4M(block_expansion * 2, block_expansion * 4, kernel_size=3, dilation=2)
            self.conv3 = P4MConvP4M(block_expansion * 4, block_expansion * 8, kernel_size=3, dilation=4)
        elif self.shift_invariance and self.rot_invariance:
            block_expansion //= 2
            self.conv0 = P4ConvZ2(3, block_expansion, kernel_size=3)
            self.conv1 = P4ConvP4(block_expansion, block_expansion * 2, kernel_size=3)
            self.conv2 = P4ConvP4(block_expansion * 2, block_expansion * 4, kernel_size=3, dilation=2)
            self.conv3 = P4ConvP4(block_expansion * 4, block_expansion * 8, kernel_size=3, dilation=4)
        elif self.shift_invariance:
            self.conv0 = nn.Conv2d(3, block_expansion, kernel_size=3)
            self.conv1 = nn.Conv2d(block_expansion, block_expansion * 2, kernel_size=3)
            self.conv2 = nn.Conv2d(block_expansion * 2, block_expansion * 4, kernel_size=3, dilation=2)
            self.conv3 = nn.Conv2d(block_expansion * 4, block_expansion * 8, kernel_size=3, dilation=4)
        elif self.rot_invariance and self.flip_invariance:
            block_expansion //= int(8 ** 0.5)
            self.conv0 = P4MConvZ2(3, block_expansion, kernel_size=3)
            self.conv1 = P4MConvP4M(block_expansion, block_expansion * 2, kernel_size=3)
            self.conv2 = P4MConvP4M(block_expansion * 2, block_expansion * 4, kernel_size=3)
            self.conv3 = P4MConvP4M(block_expansion * 4, block_expansion * 8, kernel_size=3)
        elif self.rot_invariance:
            block_expansion //= 2
            self.conv0 = P4ConvZ2(3, block_expansion, kernel_size=3)
            self.conv1 = P4ConvP4(block_expansion, block_expansion * 2, kernel_size=3)
            self.conv2 = P4ConvP4(block_expansion * 2, block_expansion * 4, kernel_size=3)
            self.conv3 = P4ConvP4(block_expansion * 4, block_expansion * 8, kernel_size=3)
        else:
            self.conv0 = nn.Conv2d(3, block_expansion, kernel_size=3)
            self.conv1 = nn.Conv2d(block_expansion, block_expansion * 2, kernel_size=3)
            self.conv2 = nn.Conv2d(block_expansion * 2, block_expansion * 4, kernel_size=3)
            self.conv3 = nn.Conv2d(block_expansion * 4, block_expansion * 8, kernel_size=3)

        self.fc = nn.Linear(block_expansion * 8, 1)

        if self.sn:
            self.conv0 = nn.utils.spectral_norm(self.conv0)
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            self.conv3 = nn.utils.spectral_norm(self.conv3)
            self.fc = nn.utils.spectral_norm(self.fc)

    def pool(self, input):
        if len(input.shape) == 5:
            out = F.avg_pool3d(input, kernel_size=(1, 2, 2))
        else:
            out = F.avg_pool2d(input, kernel_size=(2, 2))
        return out

    def pad(self, input, pad_size, is_first=False):
        if self.rot_invariance and not is_first:
            out = F.pad(input, pad=(pad_size, pad_size, pad_size, pad_size, 0, 0) if self.shift_invariance else (1, 1, 1, 1, 0, 0),
                        mode='circular' if self.shift_invariance else 'constant')
        else:
            out = F.pad(input, pad=(pad_size, pad_size, pad_size, pad_size) if self.shift_invariance else (1, 1, 1, 1),
                        mode='circular' if self.shift_invariance else 'constant')
        return out

    def forward(self, input, label):
        out = self.pad(input, 1, is_first=True)
        out = self.conv0(out)

        out = F.leaky_relu(out, 0.2)

        out = self.pad(out, 1)

        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)
        if not self.shift_invariance:
            out = self.pool(out)

        out = self.pad(out, 2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        if not self.shift_invariance:
            out = self.pool(out)

        out = self.pad(out, 4)
        out = self.conv3(out)
        out = F.leaky_relu(out, 0.2)

        if len(out.shape) == 5:
            out = out.mean(dim=(2, 3, 4))
        else:
            out = out.mean(dim=(2, 3))

        #print (out.shape)
        #out = self.fc(out)


        return out


if __name__ == "__main__":
    from chunk_rectangle_dataset import draw_object
    import torch
    import imageio
    import numpy as np

    a = draw_object(pos=(8, 8), flip=True)
    b = draw_object(pos=(8, 8))

    imageio.imsave('1.png', a)
    imageio.imsave('2.png', b)

    a = torch.Tensor(a.astype(np.float64)).unsqueeze(0).permute(0, 3, 1, 2)
    b = torch.Tensor(b.astype(np.float64)).unsqueeze(0).permute(0, 3, 1, 2)

    a = (a - 127.5) / 127.5
    b = (b - 127.5) / 127.5

    module = DCDiscriminator(block_expansion=16, shift_invariance=True, rot_invariance=True,
                             flip_invariance=True, sn=True)
    module = DCDiscriminator(block_expansion=16, shift_invariance=True, rot_invariance=True,
                             flip_invariance=True, sn=True)

    a = module(a, None)
    b = module(b, None)

    print (a.shape)

    #print(a.sum(dim=(2, 3)))

    print(a[0, 0])
    print(b[0, 0])
    print(a - b)
