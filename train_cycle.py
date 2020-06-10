from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from data import create_dataset
from models.cycle_gan_model import CycleGANModel
from options.arch_train_options import ArchTrainOptions
from utils.utils import set_log_dir, save_current_results


def cyclgan_train(opt, cycle_gan: CycleGANModel,
                  train_loader,
                  writer_dict):
    cycle_gan.train()

    writer = writer_dict['writer']
    total_iters = 0
    t_data = 0.0

    for epoch in trange(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        train_steps = writer_dict['train_steps']
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            cycle_gan.set_input(data)
            cycle_gan.optimize_parameters()

            if (i + 1) % opt.print_freq == 0:
                losses = cycle_gan.get_current_losses()
                t_comp = (time.time() - iter_start_time)
                message = "GAN: [Ep: %d/%d]" % (epoch, opt.n_epochs + opt.n_epochs_decay)
                message += "[Batch: %d/%d][time: %.3f][data: %.3f]" % (epoch_iter, len(train_loader), t_comp, t_data)
                for k, v in losses.items():
                    message += '[%s: %.3f]' % (k, v)
                tqdm.write(message)

            if (total_iters + 1) % opt.display_freq == 0:
                cycle_gan.compute_visuals()
                save_current_results(opt, cycle_gan.get_current_visuals(), train_steps)

            if (total_iters + 1) % opt.save_latest_freq == 0:
                tqdm.write('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'latest'
                cycle_gan.save_networks(save_suffix)

            iter_data_time = time.time()

        if (epoch + 1) % opt.save_epoch_freq == 0:
            cycle_gan.save_networks('latest')
            cycle_gan.save_networks(epoch)

        tqdm.write('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        writer.add_scalars('Train/discriminator', {
            "A": float(cycle_gan.loss_D_A),
            "B": float(cycle_gan.loss_D_B),
        }, train_steps)
        writer.add_scalars('Train/generator', {
            "A": float(cycle_gan.loss_G_A),
            "B": float(cycle_gan.loss_G_B),
        }, train_steps)
        writer.add_scalars('Train/cycle', {
            "A": float(cycle_gan.loss_cycle_A),
            "B": float(cycle_gan.loss_cycle_B),
        }, train_steps)
        writer.add_scalars('Train/idt', {
            "A": float(cycle_gan.loss_idt_A),
            "B": float(cycle_gan.loss_idt_B),
        }, train_steps)

        writer_dict['train_steps'] += 1
        cycle_gan.update_learning_rate()


MODEL_DIR = 'D:\\imagenet'


def main():
    opt = ArchTrainOptions().parse()
    torch.cuda.manual_seed(12345)

    opt.path_helper = set_log_dir(opt.checkpoints_dir, opt.name)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('The number of training images = %d' % len(dataset))

    cycle_gan = CycleGANModel(opt)
    cycle_gan.setup(opt)
    cycle_gan.set_arch(opt.arch, opt.n_resnet-1)

    writer_dict = {"writer": SummaryWriter(opt.path_helper['log_path']),
                   'train_steps': 0}

    # for i, data in tqdm(enumerate(dataset)):
    #     cycle_gan.set_input(data)
    #     cycle_gan.forward()
    #     cycle_gan.compute_visuals()
    #     save_current_results(opt, cycle_gan.get_current_visuals(), i)


    cyclgan_train(opt, cycle_gan, dataset, writer_dict)


if __name__ == '__main__':
    main()
