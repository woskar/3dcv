import torch
import torch.nn.functional as F
from cfg import *
import torch.nn as nn
def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

########################################################################################

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()
    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x

class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H * stride, W * stride)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route, shortcut and outlayer
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile=None, width=None, height=None, channels=None):
        super(Darknet, self).__init__()
        self.channels = 3
        self.blocks = parse_cfg("configs/config.cfg")
        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky
        self.convLast = nn.Conv2d(125, 64, 3)
        self.convLast = nn.Conv2d(64, 27, 3)

    def forward(self, x):
        ind = -1
        for i in range(0, len(self.blocks)):
            #print(self.blocks[ind]["type"])
            ind = ind + 1
            x = self.models[ind](x)
        return x

    def create_network(self, blocks):
        models = nn.ModuleList()
        prev_filters = self.channels
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        deconv_id = 0
        for block in blocks:
            if block['type'] in ['convolutional', 'deconvolutional']:
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = int((kernel_size - 1) / 2) if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                namesuffix = None
                if block['type'] == 'convolutional':
                    conv_id = conv_id + 1
                    if batch_normalize:
                        model.add_module('conv{0}'.format(conv_id),
                                         nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                        model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                        # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                    else:
                        model.add_module('conv{0}'.format(conv_id),
                                         nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                    namesuffix = conv_id
                elif block['type'] == 'deconvolutional':
                    deconv_id = deconv_id + 1
                    if batch_normalize:
                        model.add_module('deconv{0}'.format(deconv_id),
                                         nn.ConvTranspose2d(prev_filters, filters, kernel_size, stride, pad,
                                                            bias=False))
                        model.add_module('bn{0}'.format(deconv_id), nn.BatchNorm2d(filters))
                        # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                    else:
                        model.add_module('deconv{0}'.format(deconv_id),
                                         nn.ConvTranspose2d(prev_filters, filters, kernel_size, stride, pad))
                    namesuffix = deconv_id
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(namesuffix), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(namesuffix), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = int(stride * stride * prev_filters)
                out_filters.append(prev_filters)
                prev_stride = int(prev_stride * stride)
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = int(prev_stride / stride)
                out_strides.append(prev_stride)
                # models.append(nn.Upsample(scale_factor=stride, mode='nearest'))
                models.append(Upsample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                layerlen = len(layers)
                assert (layerlen >= 1)
                prev_filters = out_filters[layers[0]]
                prev_stride = out_strides[layers[0]]
                if layerlen > 1:
                    assert (layers[0] == ind - 1)
                    for i in range(1, layerlen):
                        prev_filters += out_filters[layers[i]]
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] in ['shortcut', 'outlayer']:
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            else:
                print('unknown type %s' % (block['type']))
        return models

    def load_weights(self, weightfile):
        self.load_state_dict(torch.load(weightfile))

    def save_weights(self, outfile):
        torch.save(self.state_dict(), outfile)