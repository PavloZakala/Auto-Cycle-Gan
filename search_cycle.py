from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data import create_dataset
from models.cycle_gan_model import CycleGANModel
from models_search.cycle_controller import CycleControllerModel
from options.search_options import SearchOptions
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.inception_score import _init_inception
from utils.utils import set_log_dir, save_current_results, RunningStats, save_checkpoint

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class GrowCtrler(object):
    def __init__(self, grow_step):
        self.grow_step = grow_step

    def cur_stage(self, search_iter):
        return search_iter // self.grow_step + 1


def cyclgan_train(opt, cycle_gan: CycleGANModel,
                  cycle_controller: CycleControllerModel,
                  train_loader,
                  g_loss_history: RunningStats,
                  d_loss_history: RunningStats,
                  writer_dict):
    cycle_gan.train()
    cycle_controller.eval()

    dynamic_reset = False
    writer = writer_dict['writer']
    total_iters = 0
    t_data = 0.0

    for epoch in range(opt.shared_epoch):
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

            cycle_controller.forward()

            cycle_gan.set_input(data)
            cycle_gan.optimize_parameters()

            g_loss_history.push(cycle_gan.loss_G.item())
            d_loss_history.push(cycle_gan.loss_D_A.item() +
                                cycle_gan.loss_D_B.item())

            if (i + 1) % opt.print_freq == 0:
                losses = cycle_gan.get_current_losses()
                t_comp = (time.time() - iter_start_time)
                message = "GAN: [Ep: %d/%d]" % (epoch, opt.shared_epoch)
                message += "[Batch: %d/%d][time: %.3f][data: %.3f]" % (epoch_iter, len(train_loader), t_comp, t_data)
                for k, v in losses.items():
                    message += '[%s: %.3f]' % (k, v)
                tqdm.write(message)

            if (total_iters + 1) % opt.display_freq == 0:
                cycle_gan.compute_visuals()
                save_current_results(opt, cycle_gan.get_current_visuals(), train_steps)

            if g_loss_history.is_full():
                if g_loss_history.get_var() < opt.dynamic_reset_threshold \
                        or d_loss_history.get_var() < opt.dynamic_reset_threshold:
                    dynamic_reset = True
                    tqdm.write("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            if (total_iters + 1) % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
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

        cycle_gan.update_learning_rate()

    return dynamic_reset


def controller_train(opt, cycle_gan: CycleGANModel,
                     cycle_controller: CycleControllerModel,
                     writer_dict):
    writer = writer_dict['writer']

    # train mode
    cycle_controller.train()

    # eval mode
    cycle_gan.eval()
    iter_start_time = time.time()
    for i in range(0, opt.ctrl_step):
        controller_step = writer_dict['controller_steps']

        cycle_controller.step_A()
        cycle_controller.step_B()

        if (i + 1) % opt.print_freq_controller == 0:
            losses = cycle_controller.get_current_losses()
            t_comp = (time.time() - iter_start_time)
            iter_start_time = time.time()
            message = "Cont: [Ep: %d/%d]" % (i, opt.ctrl_step) + "[{}][{}]".format(cycle_controller.arch_A,
                                                                                   cycle_controller.arch_B)
            message += "[time: %.3f]" % (t_comp)
            for k, v in losses.items():
                message += '[%s: %.3f]' % (k, v)
            tqdm.write(message)
        # write
        writer.add_scalars('Controller/loss', {
            "A": cycle_controller.loss_A.item(),
            "B": cycle_controller.loss_B.item()
        }, controller_step)

        writer.add_scalars('Controller/discriminator', {
            "A": cycle_controller.loss_D_A.item(),
            "B": cycle_controller.loss_D_B.item()
        }, controller_step)
        writer.add_scalars('Controller/inception_score', {
            "A": cycle_controller.loss_IS_A.item(),
            "B": cycle_controller.loss_IS_B.item()
        }, controller_step)

        writer.add_scalars('Controller/adv', {
            "A": cycle_controller.loss_adv_A,
            "B": cycle_controller.loss_adv_B
        }, controller_step)
        writer.add_scalars('Controller/entropy', {
            "A": cycle_controller.loss_entropy_A,
            "B": cycle_controller.loss_entropy_B
        }, controller_step)
        writer.add_scalars('Controller/reward', {
            "A": cycle_controller.loss_reward_A,
            "B": cycle_controller.loss_reward_B
        }, controller_step)

        writer_dict['controller_steps'] = controller_step + 1


MODEL_DIR = 'D:\\imagenet'


def main():
    opt = SearchOptions().parse()
    torch.cuda.manual_seed(12345)

    _init_inception(MODEL_DIR)
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    start_search_iter = 0
    cur_stage = 1

    grow_ctrler = GrowCtrler(opt.grow_step)

    if opt.load_path:
        print(f'=> resuming from {opt.load_path}')
        assert os.path.exists(opt.load_path)
        checkpoint_file = os.path.join(opt.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location={'cuda:0': 'cpu'})
        # set controller && its optimizer
        cur_stage = checkpoint['cur_stage']
        start_search_iter = checkpoint["search_iter"]
        opt.path_helper = checkpoint['path_helper']

        cycle_gan = CycleGANModel(opt)
        cycle_gan.setup(opt)

        cycle_controller = CycleControllerModel(opt, cur_stage=cur_stage)
        cycle_controller.setup(opt)
        cycle_controller.set(cycle_gan)

        cycle_gan.load_from_state(checkpoint["cycle_gan"])
        cycle_controller.load_from_state(checkpoint["cycle_controller"])

    else:
        opt.path_helper = set_log_dir(opt.checkpoints_dir, opt.name)

        cycle_gan = CycleGANModel(opt)
        cycle_gan.setup(opt)

        cycle_controller = CycleControllerModel(opt, cur_stage=cur_stage)
        cycle_controller.setup(opt)
        cycle_controller.set(cycle_gan)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('The number of training images = %d' % len(dataset))

    writer_dict = {"writer": SummaryWriter(opt.path_helper['log_path']),
                   'controller_steps': start_search_iter * opt.ctrl_step,
                   'train_steps': start_search_iter * opt.shared_epoch}

    g_loss_history = RunningStats(opt.dynamic_reset_window)
    d_loss_history = RunningStats(opt.dynamic_reset_window)

    grow_steps = [int(1.3 * opt.grow_step ** i) for i in range(1, opt.n_resnet - 1)]
    opt.max_search_iter = sum(grow_steps)

    for search_iter in tqdm(range(int(start_search_iter), int(opt.max_search_iter))):
        tqdm.write(f"<start search iteration {search_iter}>")
        cycle_controller.reset()

        if search_iter in grow_steps:
            cur_stage = grow_ctrler.cur_stage(search_iter + 1)
            tqdm.write(f'=> grow to stage {cur_stage}')
            prev_archs_A, prev_hiddens_A = cycle_controller.get_topk_arch_hidden_A()
            prev_archs_B, prev_hiddens_B = cycle_controller.get_topk_arch_hidden_B()

            del cycle_controller

            cycle_controller = CycleControllerModel(opt, cur_stage)
            cycle_controller.setup(opt)
            cycle_controller.set(cycle_gan, prev_hiddens_A, prev_hiddens_B, prev_archs_A, prev_archs_B)

        dynamic_reset = cyclgan_train(opt, cycle_gan, cycle_controller, dataset,
                                      g_loss_history, d_loss_history, writer_dict)

        controller_train(opt, cycle_gan, cycle_controller, writer_dict)

        if dynamic_reset:
            tqdm.write('re-initialize share GAN')
            del cycle_gan
            cycle_gan = CycleGANModel(opt)
            cycle_gan.setup(opt)

        save_checkpoint({
            'cur_stage': cur_stage,
            'search_iter': search_iter + 1,
            'cycle_gan': cycle_gan.save_networks(epoch=search_iter),
            'cycle_controller': cycle_controller.save_networks(epoch=search_iter),
            'path_helper': opt.path_helper
        }, False, opt.path_helper['ckpt_path'])

    final_archs_A, _ = cycle_controller.get_topk_arch_hidden_A()
    final_archs_B, _ = cycle_controller.get_topk_arch_hidden_B()
    print(f"discovered archs: {final_archs_A}")
    print(f"discovered archs: {final_archs_B}")


if __name__ == '__main__':
    main()
