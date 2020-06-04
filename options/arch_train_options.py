from .train_options import TrainOptions


class ArchTrainOptions(TrainOptions):

    def initialize(self, parser):
        TrainOptions.initialize(self, parser)

        # parser.add_argument('--entropy_coeff', type=float, default=1e-3,
        #                     help='to encourage the exploration')
        parser.add_argument('--write_train_step', type=int, default=100,
                            help='the size of hidden vector')
        # parser.add_argument('--load_path', type=str,
        #                     help='The reload model path')
        parser.add_argument('--path', default=".", type=str, help='number of candidate architectures to be sampled')
        parser.add_argument('--arch', nargs='+', type=int,
                            help='the vector of a discovered architecture')
        return parser
