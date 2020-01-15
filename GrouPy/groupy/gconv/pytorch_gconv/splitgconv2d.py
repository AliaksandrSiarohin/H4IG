import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math
from torch.nn.modules.utils import _pair
from groupy.gconv.make_gconv_indices import *

make_indices_functions = {(1, 4): make_c4_z2_indices,
                          (4, 4): make_c4_p4_indices,
                          (1, 8): make_d4_z2_indices,
                          (8, 8): make_d4_p4m_indices}


def trans_filter(w, inds):
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                               inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    return w_transformed.contiguous()


class SplitGConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, input_stabilizer_size=1, output_stabilizer_size=4):
        super(SplitGConv2D, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size
        self.dilation = dilation

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds)
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)

        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels * self.input_stabilizer_size, input_shape[-2],
                           input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                     padding=self.padding, dilation=self.dilation)
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y


class P4ConvZ2(SplitGConv2D):
    def __init__(self, *args, **kwargs):
        super(P4ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, *args, **kwargs)


class P4ConvP4(SplitGConv2D):
    def __init__(self, *args, **kwargs):
        super(P4ConvP4, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, *args, **kwargs)


class P4MConvZ2(SplitGConv2D):
    def __init__(self, *args, **kwargs):
        super(P4MConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, *args, **kwargs)


class P4MConvP4M(SplitGConv2D):
    def __init__(self, *args, **kwargs):
        super(P4MConvP4M, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)


import math


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, first=False):
        super(BasicBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential()
        conv = P4ConvP4 if not first else P4ConvZ2
        self.layers.add_module('Conv', conv(in_planes, out_planes, \
                                            kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm3d(out_planes))
        self.layers.add_module('ReLU', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class PoolAndCls(nn.Module):
    def __init__(self, nChannels):
        super(PoolAndCls, self).__init__()
        self.cls = nn.Conv1d(nChannels, 1, kernel_size=1)

    def forward(self, feat):
        num_channels = feat.size(1)
        out = F.avg_pool3d(feat, (1, feat.size(3), feat.size(4)))
        out = out.view(-1, num_channels, feat.size(2))
        out = self.cls(out).squeeze(1)
        return out


class GroupNetworkInNetwork(nn.Module):
    def __init__(self, opt):
        super(GroupNetworkInNetwork, self).__init__()

        num_classes = opt['num_classes']
        num_inchannels = opt['num_inchannels'] if ('num_inchannels' in opt) else 3
        num_stages = opt['num_stages'] if ('num_stages' in opt) else 3
        use_avg_on_conv3 = opt['use_avg_on_conv3'] if ('use_avg_on_conv3' in opt) else True

        assert (num_stages >= 3)
        nChannels = 192 // 2
        nChannels2 = 160 // 2
        nChannels3 = 96 // 2

        blocks = [nn.Sequential() for i in range(num_stages)]
        # 1st block
        blocks[0].add_module('Block1_ConvB1', BasicBlock(num_inchannels, nChannels, 5, first=True))
        blocks[0].add_module('Block1_ConvB2', BasicBlock(nChannels, nChannels2, 1))
        blocks[0].add_module('Block1_ConvB3', BasicBlock(nChannels2, nChannels3, 1))
        blocks[0].add_module('Block1_MaxPool', nn.MaxPool3d(kernel_size=(1, 2, 2)))

        # 2nd block
        blocks[1].add_module('Block2_ConvB1', BasicBlock(nChannels3, nChannels, 5))
        blocks[1].add_module('Block2_ConvB2', BasicBlock(nChannels, nChannels, 1))
        blocks[1].add_module('Block2_ConvB3', BasicBlock(nChannels, nChannels, 1))
        blocks[1].add_module('Block2_AvgPool', nn.AvgPool3d(kernel_size=(1, 2, 2)))
        #
        # # 3rd block
        blocks[2].add_module('Block3_ConvB1', BasicBlock(nChannels, nChannels, 3))
        blocks[2].add_module('Block3_ConvB2', BasicBlock(nChannels, nChannels, 1))
        blocks[2].add_module('Block3_ConvB3', BasicBlock(nChannels, nChannels, 1))
        #
        if num_stages > 3 and use_avg_on_conv3:
            blocks[2].add_module('Block3_AvgPool',
                                 nn.AvgPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        for s in range(3, num_stages):
            blocks[s].add_module('Block' + str(s + 1) + '_ConvB1', BasicBlock(nChannels, nChannels, 3))
            blocks[s].add_module('Block' + str(s + 1) + '_ConvB2', BasicBlock(nChannels, nChannels, 1))
            blocks[s].add_module('Block' + str(s + 1) + '_ConvB3', BasicBlock(nChannels, nChannels, 1))

        # global average pooling and classifier
        blocks.append(nn.Sequential())
        blocks[-1].add_module('Classifier', PoolAndCls(nChannels))

        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ['conv' + str(s + 1) for s in range(num_stages)] + ['classifier', ]
        assert (len(self.all_feat_names) == len(self._feature_blocks))

    def _parse_out_keys_arg(self, out_feat_keys):

        # By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1], ] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    'Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {0}.'.format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.

        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        # print (x.shape)
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat + 1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat.view(feat.shape[0], -1, feat.shape[-2],
                                                                feat.shape[-1]) if key != 'classifier' else feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
        return out_feats

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, P4ConvP4) or isinstance(m, P4ConvZ2):
                if m.weight.requires_grad:
                    n = 4 * m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad:
                    m.weight.data.fill_(1)
                if m.bias.requires_grad:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias.requires_grad:
                    m.bias.data.zero_()


def create_model(opt):
    return GroupNetworkInNetwork(opt)


if __name__ == '__main__':
    size = 32
    opt = {'num_classes': 4, 'num_stages': 5}

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(1, 3, size, size).uniform_(-1, 1))

    out = net(x, out_feat_keys=net.all_feat_names)
    for f in range(len(out)):
        print('Output feature {0} - size {1}'.format(
            net.all_feat_names[f], out[f].size()))

    out = net(x)
    print('Final output: {0}'.format(out.size()))

if __name__ == "__main__":
    opt = {}
    opt['num_classes'] = 4
    opt['num_inchannels'] = 3
    layer = GroupNetworkInNetwork(opt)

    img = torch.rand((1, 3, 32, 32))

    img_rot = img.transpose(-1, -2)[..., range(32)[::-1]]

    print(img)
    print(img_rot)

    out = layer(img)
    out_rot = layer(img_rot)
    print(out.shape)

    print(out)
    print('--------------------------')
    print(out_rot)
