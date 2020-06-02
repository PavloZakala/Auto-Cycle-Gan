from .train_options import TrainOptions


class SearchOptions(TrainOptions):

    def initialize(self, parser):
        TrainOptions.initialize(self, parser)

        parser.add_argument('--shared_epoch', type=int, default=15,
                            help='the number of epoch to train the shared gan at each search iteration')
        parser.add_argument('--grow_step', type=int, default=25,
                            help='which iteration to grow the image size from 8 to 16')
        parser.add_argument('--max_search_iter', type=int, default=90,
                            help='max search iterations of this algorithm')
        parser.add_argument('--topk', type=int, default=5,
                            help='preserve topk models architectures after each stage')
        parser.add_argument('--ctrl_sample_batch', type=int, default=1,
                            help='sample size of controller of each step')
        parser.add_argument('--hid_size', type=int, default=100,
                            help='the size of hidden vector')
        parser.add_argument('--ctrl_lr', type=float, default=3.5e-4,
                            help='adam: ctrl learning rate')
        parser.add_argument('--baseline_decay', type=float, default=0.9,
                            help='baseline decay rate in RL')
        parser.add_argument('--entropy_coeff', type=float, default=1e-3,
                            help='to encourage the exploration')
        parser.add_argument('--ctrl_step', type=int, default=30,
                            help='number of steps to train the controller at each search iteration')
        parser.add_argument('--dynamic_reset_window', type=int, default=100,
                            help='the window size')
        parser.add_argument('--dynamic_reset_threshold', type=float, default=1e-3,
                            help='var threshold')
        parser.add_argument('--write_train_step', type=int, default=100,
                            help='the size of hidden vector')
        parser.add_argument('--load_path', type=str,
                            help='The reload model path')
        parser.add_argument('--print_freq_controller', type=int, default=10,
                            help='frequency of showing training results on console')
        parser.add_argument('--num_candidate', type=int, default=10,
                            help='number of candidate architectures to be sampled')
        return parser
