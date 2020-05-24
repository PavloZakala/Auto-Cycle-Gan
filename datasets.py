# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageDataset(object):
    def __init__(self, args, batch_size, num_workers, cur_img_size=None):
        img_size = cur_img_size if cur_img_size else args.img_size
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'stl10':

            tr = Dt(root=args.data_path, split='train+unlabeled', transform=transform, download=True)
            tr.data = tr.data[:min(args.data_size, len(tr.data))]
            vl = Dt(root=args.data_path, split='test', transform=transform)
            vl.data = vl.data[:min(args.data_size, len(vl.data))]

            self.train = torch.utils.data.DataLoader(
                tr, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                vl, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)

            self.test = self.valid
        else:

            tr = Dt(root=args.data_path, train=True, transform=transform, download=True)
            tr.data = tr.data[:min(args.data_size, len(tr.data))]
            vl = Dt(root=args.data_path, train=False, transform=transform)
            vl.data = vl.data[:min(args.data_size, len(vl.data))]
            self.train = torch.utils.data.DataLoader(
                tr, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                vl, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)

            self.test = self.valid

