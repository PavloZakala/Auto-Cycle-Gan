# -*- coding: utf-8 -*-
# @Date    : 2019-07-31
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from torch import nn
from models.building_blocks import Cell, DisBlock, OptimizedDisBlock, CellDown, CellUp


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=0, short_cut=True)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=1, short_cut=True)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=2, short_cut=True)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out, ))
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)

        return output


class GeneratorED(nn.Module):
    def __init__(self, args):
        super(GeneratorED, self).__init__()
        self.args = args
        # self.ch = args.gf_dim
        # self.bottom_width = args.bottom_width
        gf_dim = 32
        self.down_cell1 = CellDown(3, gf_dim, 'maxpool', skip_in=[], short_cut=True)
        self.down_cell2 = CellDown(gf_dim, gf_dim * 2, 'maxpool', skip_in=[], short_cut=True)
        self.down_cell3 = CellDown(gf_dim * 2, gf_dim * 4, 'maxpool', skip_in=[], short_cut=True)
        # self.down_cell4 = CellDown(gf_dim * 4, gf_dim * 8, 'maxpool', skip_in=[], short_cut=True)
        #
        # self.up_cell4 = CellUp(gf_dim * 8, gf_dim * 4, 'nearest', skip_in=[], short_cut=True)
        self.up_cell3 = CellUp(gf_dim * 4, gf_dim * 2, 'nearest', skip_in=[], short_cut=True)
        self.up_cell2 = CellUp(gf_dim * 2, gf_dim, 'nearest', skip_in=[gf_dim * 2], short_cut=True)
        self.up_cell1 = CellUp(gf_dim, gf_dim, 'nearest', skip_in=[gf_dim])
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(),
            nn.Conv2d(gf_dim, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        h1, x = self.down_cell1(x, [])
        h2, x = self.down_cell2(x, [])
        h3, x = self.down_cell3(x, [])
        # h4, x = self.down_cell4(x, [])

        # _, x = self.up_cell4(x, [])
        _, x = self.up_cell3(x, [])
        _, x = self.up_cell2(x, [h2])
        _, x = self.up_cell1(x, [h1])

        output = self.to_rgb(x)

        return output


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output

if __name__ == '__main__':
    import torch
    gen = GeneratorED()

    img = torch.rand(10, 3, 128, 128)
    y = gen(img)
    print(y.shape)