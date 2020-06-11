import functools

import torch
import torch.nn as nn

from models_search.building_blocks_search import decimal2binaryGray

class ConvCell(nn.Module):

    def __init__(self, dim, padding_type, num_skip_in, norm_layer, use_dropout, use_bias, activation=nn.ReLU):
        super(ConvCell, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), activation(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        self.conv1 = nn.Sequential(*conv_block)

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        self.conv2 = nn.Sequential(*conv_block)
        self.relu = activation(True)
        self.num_skip_in = num_skip_in
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1),
                                                              norm_layer(dim),
                                                              activation(True)]) for _ in range(num_skip_in)])

    def set_arch(self, skip_ins, skip_types):
        if self.num_skip_in:
            self.skip_ins = [0] * self.num_skip_in
            for skip_idx, (skip_in, skip_type) in enumerate(zip(decimal2binaryGray(skip_ins, self.num_skip_in)[::-1],
                                                              decimal2binaryGray(skip_types, self.num_skip_in)[::-1])):
                if int(skip_in) != 0:
                    self.skip_ins[-(skip_idx + 1)] = int(skip_type) + 1

    def forward(self, x, skip_ft=None):
        residual = self.conv1(x)
        h = residual

        if self.num_skip_in:
            assert len(self.skip_in_ops) == len(self.skip_ins)
            for skip_flag, ft, skip_in_op in zip(self.skip_ins, skip_ft, self.skip_in_ops):
                if skip_flag == 1:
                    residual = residual + ft
                elif skip_flag == 2:
                    residual = residual + skip_in_op(ft)

        return self.relu(x + self.conv2(residual)), h

class AutoResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', max_skip_num=3):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        assert (n_blocks >= 0)
        self.cur_stage = 0
        self.max_skip_num = max_skip_num
        super(AutoResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.encode = nn.Sequential(*model)

        """ ************ """
        self.resnet_flow = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            self.resnet_flow += [ConvCell(ngf * mult, num_skip_in=min(i, self.max_skip_num), padding_type=padding_type,
                                          norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.resnet_flow = nn.ModuleList(self.resnet_flow)

        """ ************ """

        model = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.decode = nn.Sequential(*model)

    def set_arch(self, arch_id, cur_stage):
        NUM_ARCH = 2

        if not isinstance(arch_id, list):
            arch_id = arch_id.to('cpu').numpy().tolist()
        arch_id = [int(x) for x in arch_id]

        self.cur_stage = cur_stage
        for i in range(self.cur_stage):
            arch_stage = arch_id[i * NUM_ARCH:(i + 1) * NUM_ARCH]
            print(arch_stage)
            self.resnet_flow[i+1].set_arch(*arch_stage)

    def forward(self, input):
        """Standard forward"""
        x = self.encode(input)

        hs = []
        for i, res_block in enumerate(self.resnet_flow[:self.cur_stage+1]):
            x, h = res_block(x, skip_ft=hs)

            if i == self.max_skip_num:
                hs.pop(0)
            hs.append(h)

        x = self.decode(x)
        return x
