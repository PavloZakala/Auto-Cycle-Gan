import torch
import os
import numpy as np
import torch.nn as nn


import time
import cfg
from tqdm import tqdm
from models import create_model
from options.train_options import TrainOptions
from models.autogan_cifar10_b import Discriminator, GeneratorED
from data import create_dataset
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.utils import set_log_dir, save_checkpoint, create_logger

MODEL_DIR = 'D:\\imagenet'

def train(args, dataset, model, logger):


    for i, data in enumerate(dataset):  # inner loop within one epoch

        iter_start_time = time.time()

        model.set_input(data)
        model.optimize_parameters()

        if (i+1) % args.print_freq == 0:
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time)
            message = "[Batch: %d/%d][time: %.3f]" % (i, len(dataset), t_comp)
            for k, v in losses.items():
                message += '[%s: %.3f]' % (k, v)
            tqdm.write(message)

    model.update_learning_rate()

def main():

    args = TrainOptions().parse()  # get training options
    torch.cuda.manual_seed(args.random_seed)

    args.path_helper = set_log_dir('logs', args.name)
    logger = create_logger(args.path_helper['log_path'])

    dataset = create_dataset(args)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(args)      # create a model given opt.model and other options
    model.setup(args)               # regular setup: load and print networks; create schedulers

    for _ in tqdm(range(0, args.n_epochs + args.n_epochs_decay + 1)):
        train(args, dataset, model, logger)



if __name__ == '__main__':
    main()