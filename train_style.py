import time

import torch
from tqdm import tqdm

from utils import utils
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from utils.utils import set_log_dir, create_logger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train(args, epoch, dataset, model, logger):
    for i, data in enumerate(dataset):  # inner loop within one epoch

        iter_start_time = time.time()

        model.set_input(data)
        model.optimize_parameters()

        if (i + 1) % args.print_freq == 0:
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time)
            message = "[Batch: %d/%d][time: %.3f]" % (i, len(dataset), t_comp)
            for k, v in losses.items():
                message += '[%s: %.3f]' % (k, v)
            logger.info(message)
            tqdm.write(message)

    model.update_learning_rate()


def main():
    args = TrainOptions().parse()  # get training options
    torch.cuda.manual_seed(args.random_seed)

    args.path_helper = set_log_dir('logs', args.name)
    logger = create_logger(args.path_helper['log_path'])
    args.logger = logger

    dataset = create_dataset(args)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    logger.info('The number of training images = %d' % dataset_size)

    model = create_model(args)  # create a model given opt.model and other options
    model.setup(args)  # regular setup: load and print networks; create schedulers

    total = 0
    for epoch in tqdm(range(0, args.n_epochs + args.n_epochs_decay + 1)):
        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()

            model.set_input(data)
            model.optimize_parameters()

            if (total + 1) % args.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time)
                message = "[Batch: %d/%d][time: %.3f]" % (i, len(dataset), t_comp)
                for k, v in losses.items():
                    message += '[%s: %.3f]' % (k, v)
                logger.info(message)
                tqdm.write(message)

            if (total + 1) % args.display_freq == 0:
                model.compute_visuals()
                utils.save_current_results(args, model.get_current_visuals(), epoch)

            if (total + 1) % args.save_epoch_freq == 0:
                logger.info('saving the model at the end of epoch %d' % (epoch))
                model.save_networks('latest')
                model.save_networks(epoch)

            total += 1
        model.update_learning_rate()


if __name__ == '__main__':
    main()
