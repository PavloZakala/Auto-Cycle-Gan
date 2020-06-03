import operator

import numpy as np
import torch
import os

from data import create_dataset
from models import networks
from models.base_model import BaseModel
from models_search.controller import Controller
from utils.inception_score import get_inception_score
from utils.utils import load_saves


class CycleControllerModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--ctrl_lr', type=float, default=3.5e-4, help='adam: ctrl learning rate')
            parser.add_argument('--lr_decay', action='store_true', help='learning rate decay or not')
            parser.add_argument('--beta1', type=float, default=0.0,
                                help='adam: decay of first order momentum of gradient')
            parser.add_argument('--beta2', type=float, default=0.9,
                                help='adam: decay of first order momentum of gradient')
            parser.add_argument('--ctrl_sample_batch', type=int, default=1,
                                help='sample size of controller of each step')
            parser.add_argument('--baseline_decay', type=float, default=0.9,
                                help='baseline decay rate in RL')
            parser.add_argument('--entropy_coeff', type=float, default=1e-3,
                                help='to encourage the exploration')

        return parser

    @staticmethod
    def get_score(opt, netG, netD, dataloader, loss, data_name):

        # eval mode
        netG.eval()
        img_list = list()
        ds = list()
        with torch.no_grad():
            for data in dataloader:
                imgs = data[data_name]
                if len(opt.gpu_ids) != 0:
                    imgs = imgs.cuda()

                conv_imgs = netG(imgs)

                d = [float(loss(im.unsqueeze(0), True)) for im in conv_imgs]
                conv_imgs = conv_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                        torch.uint8).numpy()
                img_list.extend(list(conv_imgs))
                ds.extend(d)

        mean_is, std_is = get_inception_score(img_list, splits=1)

        mean_d = 1 / np.mean(ds)
        std_d = 1 / np.std(ds)

        return mean_is, mean_d * 3.0

    def __init__(self, opt, cur_stage):

        BaseModel.__init__(self, opt)

        self.loss_names = ["A", "B",
                           'D_A', 'D_B',
                           'IS_A', 'IS_B',
                           'reward_A', 'reward_B',
                           "adv_A", "adv_B",
                           "entropy_A", "entropy_B"]

        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']

        self.visual_names = visual_names_A + visual_names_B

        self.cur_stage = cur_stage
        self.model_names = ["C_A", "C_B"]
        self.ctrl_sample_batch = opt.ctrl_sample_batch

        self.netC_A = Controller(opt, self.cur_stage)
        self.netC_B = Controller(opt, self.cur_stage)

        self.netD_A = networks.define_D(3, 64, "basic", norm='instance')
        self.netD_B = networks.define_D(3, 64, "basic", norm='instance')
        load_saves(self.netD_A, "res", "D_A", os.path.join(opt.path, "pre_mod"))
        load_saves(self.netD_B, "res", "D_B", os.path.join(opt.path, "pre_mod"))
        self.loss = networks.GANLoss("lsgan")

        self.prev_hiddens_A = None
        self.prev_archs_A = None

        self.prev_hiddens_B = None
        self.prev_archs_B = None

        if len(self.gpu_ids) != 0:
            self.netC_A = self.netC_A.cuda()
            self.netC_A = self.netC_A.cuda()

        networks.init_weights(self.netC_A, opt.init_type, opt.init_gain)
        networks.init_weights(self.netC_A, opt.init_type, opt.init_gain)

        self.optimizers_names = ["A", "B"]
        self.optimizerA = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netC_A.parameters()),
                                           opt.ctrl_lr, (0.0, 0.9))
        self.optimizerB = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netC_A.parameters()),
                                           opt.ctrl_lr, (0.0, 0.9))

        self.optimizers.append(self.optimizerA)
        self.optimizers.append(self.optimizerB)

        self.valid_dataloader = create_dataset(opt)
        self.baseline_decay = opt.baseline_decay
        self.entropy_coeff = opt.entropy_coeff

        self.netG_A = None
        self.netG_B = None

        self.baseline_A = None
        self.baseline_B = None

    def set(self, cycle_gan,
            prev_hiddens_A=None,
            prev_hiddens_B=None,
            prev_archs_A=None,
            prev_archs_B=None):

        self.netG_A = cycle_gan.netG_A
        self.netG_B = cycle_gan.netG_B

        self.prev_hiddens_A = prev_hiddens_A
        self.prev_hiddens_B = prev_hiddens_B

        self.prev_archs_A = prev_archs_A
        self.prev_archs_B = prev_archs_B

    def reset(self):
        self.baseline_A = None
        self.baseline_B = None

    def forward(self):
        arch = self.netC_A.sample(1, prev_hiddens=self.prev_hiddens_A, prev_archs=self.prev_archs_A,
                                  cpu=len(self.gpu_ids) == 0)[0][0]

        self.netG_A.set_arch(arch, self.netC_A.cur_stage)

        arch = self.netC_B.sample(1, prev_hiddens=self.prev_hiddens_B, prev_archs=self.prev_archs_B,
                                  cpu=len(self.gpu_ids) == 0)[0][0]

        self.netG_B.set_arch(arch, self.netC_B.cur_stage)

    def save_networks(self, epoch):
        state_dict = super().save_networks(epoch)
        state_dict["prev_hiddens_A"] = self.prev_hiddens_A
        state_dict["prev_hiddens_B"] = self.prev_hiddens_B

        state_dict["prev_archs_A"] = self.prev_archs_A
        state_dict["prev_archs_B"] = self.prev_archs_B

        return state_dict

    def load_from_state(self, opt_state_dict):
        super().load_from_state(opt_state_dict)
        self.prev_hiddens_A = opt_state_dict["prev_hiddens_A"]
        self.prev_hiddens_B = opt_state_dict["prev_hiddens_B"]

        self.prev_archs_A = opt_state_dict["prev_archs_A"]
        self.prev_archs_B = opt_state_dict["prev_archs_B"]

    def set_input(self, input):

        self.baseline_A = input['baseline_A']
        self.baseline_B = input['baseline_B']
        if len(self.gpu_ids) != 0:
            self.baseline_A = self.baseline_A.cuda()
            self.baseline_B = self.baseline_B.cuda()

    def step_A(self):
        archs, selected_log_probs, entropies = self.netC_A.sample(self.ctrl_sample_batch,
                                                                  prev_hiddens=self.prev_hiddens_A,
                                                                  prev_archs=self.prev_archs_A,
                                                                  cpu=len(self.gpu_ids) == 0)

        cur_batch_rewards = []
        for arch in archs:
            self.netG_A.set_arch(arch, self.cur_stage)
            is_score, d_score = self.get_score(self.opt, self.netG_A, self.netD_A, self.valid_dataloader, self.loss,
                                               "A")
            cur_batch_rewards.append(is_score + d_score)
            self.loss_IS_A = is_score
            self.loss_D_A = d_score
        self.arch_A = archs.tolist()
        cur_batch_rewards = torch.tensor(cur_batch_rewards, requires_grad=False)
        if len(self.gpu_ids) != 0:
            cur_batch_rewards = torch.tensor(cur_batch_rewards, requires_grad=False).cuda()
        cur_batch_rewards = cur_batch_rewards.unsqueeze(-1) + self.entropy_coeff * entropies
        if self.baseline_A is None:
            self.baseline_A = cur_batch_rewards
        else:
            self.baseline_A = self.baseline_decay * self.baseline_A.detach() + (
                    1 - self.baseline_decay) * cur_batch_rewards
        adv = cur_batch_rewards - self.baseline_A

        # policy loss
        self.loss_A = -selected_log_probs * adv
        self.loss_A = self.loss_A.sum()

        # update controller
        self.optimizerA.zero_grad()
        self.loss_A.backward()
        self.optimizerA.step()

        self.loss_reward_A = cur_batch_rewards.mean().item()
        self.loss_adv_A = adv.mean().item()
        self.loss_entropy_A = entropies.mean().item()

    def step_B(self):
        archs, selected_log_probs, entropies = self.netC_B.sample(self.ctrl_sample_batch,
                                                                  prev_hiddens=self.prev_hiddens_B,
                                                                  prev_archs=self.prev_archs_B,
                                                                  cpu=len(self.gpu_ids) == 0)

        cur_batch_rewards = []
        for arch in archs:
            self.netG_B.set_arch(arch, self.cur_stage)
            is_score, d_score = self.get_score(self.opt, self.netG_B, self.netD_B, self.valid_dataloader, self.loss,
                                               "B")
            cur_batch_rewards.append(is_score + d_score)
            self.loss_IS_B = is_score
            self.loss_D_B = d_score
        self.arch_B = archs.tolist()
        cur_batch_rewards = torch.tensor(cur_batch_rewards, requires_grad=False)
        if len(self.gpu_ids) != 0:
            cur_batch_rewards = torch.tensor(cur_batch_rewards, requires_grad=False).cuda()
        cur_batch_rewards = cur_batch_rewards.unsqueeze(-1) + self.entropy_coeff * entropies
        if self.baseline_B is None:
            self.baseline_B = cur_batch_rewards
        else:
            self.baseline_B = self.baseline_decay * self.baseline_B.detach() + (
                    1 - self.baseline_decay) * cur_batch_rewards
        adv = cur_batch_rewards - self.baseline_B

        # policy loss
        self.loss_B = -selected_log_probs * adv
        self.loss_B = self.loss_B.sum()

        # update controller
        self.optimizerB.zero_grad()
        self.loss_B.backward()
        self.optimizerB.step()

        self.loss_reward_B = cur_batch_rewards.mean().item()
        self.loss_adv_B = adv.mean().item()
        self.loss_entropy_B = entropies.mean().item()

    def get_topk_arch_hidden_A(self):

        self.netC_A.eval()
        cur_stage = self.netC_A.cur_stage
        archs, _, _, hiddens = self.netC_A.sample(self.opt.num_candidate, with_hidden=True,
                                                  prev_archs=self.prev_archs_A,
                                                  prev_hiddens=self.prev_hiddens_A,
                                                  cpu=len(self.gpu_ids) == 0)
        hxs, cxs = hiddens
        arch_idx_perf_table = {}
        for arch_idx in range(len(archs)):
            self.netG_A.set_arch(archs[arch_idx], cur_stage)
            is_score, d_score = self.get_score(self.opt, self.netG_A, self.netD_A, self.valid_dataloader,
                                               self.loss, "A")
            arch_idx_perf_table[arch_idx] = is_score + d_score
        topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:self.opt.topk]
        topk_archs = []
        topk_hxs = []
        topk_cxs = []
        for arch_idx_perf in topk_arch_idx_perf:
            arch_idx = arch_idx_perf[0]
            topk_archs.append(archs[arch_idx])
            topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
            topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

        return topk_archs, (topk_hxs, topk_cxs)

    def get_topk_arch_hidden_B(self):

        self.netC_B.eval()
        cur_stage = self.netC_B.cur_stage
        archs, _, _, hiddens = self.netC_B.sample(self.opt.num_candidate, with_hidden=True,
                                                  prev_archs=self.prev_archs_B,
                                                  prev_hiddens=self.prev_hiddens_B,
                                                  cpu=len(self.gpu_ids) == 0)
        hxs, cxs = hiddens
        arch_idx_perf_table = {}
        for arch_idx in range(len(archs)):
            self.netG_B.set_arch(archs[arch_idx], cur_stage)
            is_score, d_score = self.get_score(self.opt, self.netG_B, self.netD_B, self.valid_dataloader,
                                               self.loss, "B")
            arch_idx_perf_table[arch_idx] = is_score + d_score
        topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:self.opt.topk]
        topk_archs = []
        topk_hxs = []
        topk_cxs = []
        for arch_idx_perf in topk_arch_idx_perf:
            arch_idx = arch_idx_perf[0]
            topk_archs.append(archs[arch_idx])
            topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
            topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

        return topk_archs, (topk_hxs, topk_cxs)

    def optimize_parameters(self):

        self.step_A()
        self.step_B()
