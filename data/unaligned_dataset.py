import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import pickle as plt


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    def load_data(self, path):

        with open(path, 'rb') as f:
            data = plt.load(f)
        return data

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        if opt.valid:
            self.dir_A = os.path.join(opt.dataroot, 'validA')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, 'validB')  # create a path '/path/to/data/trainB'
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        dataA = self.load_data("{}.plt".format(self.dir_A))
        dataB = self.load_data("{}.plt".format(self.dir_B))

        self.A_images = [img for _, img in dataA]
        self.B_images = [img for _, img in dataB]

        self.A_paths = [os.path.join(self.dir_A, name) for name, _ in dataA]
        self.B_paths = [os.path.join(self.dir_B, name) for name, _ in dataB]

        self.A_size = min(len(self.A_images), int(opt.max_dataset_size))  # get the size of dataset A
        self.B_size = min(len(self.B_images), int(opt.max_dataset_size))  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A = self.transform_A(self.A_images[index % self.A_size])
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B = self.transform_B(self.B_images[index_B])

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
