import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import join
from .utils import load_nifti_img, check_exceptions, is_image_file

import torchvision
import skimage.transform
import torchsample.transforms as ts

import torch


class SplitTCIA3DDataset(data.Dataset):
    def __init__(self, root_dir, split, data_splits, im_dim = None, transform=None, preload_data=False):
        super(SplitTCIA3DDataset, self).__init__()
        
        self.im_dim = im_dim

        # list_dir = []

        self.image_filenames = []
        self.target_filenames = []

        for i in data_splits:
            # list_dir.append(join(root_dir, i))

            image_dir = join(root_dir, self.im_dim, i, 'image')
            # print("\n\n\n", image_dir,"\n\n\n")
            target_dir = join(root_dir, '512_512_256', i, 'label')


            self.image_filenames  += [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
            self.target_filenames += [join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)]

        self.image_filenames = sorted(self.image_filenames)
        self.target_filenames = sorted(self.target_filenames)

        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [load_nifti_img(ii, dtype=np.int16)[0] for ii in self.image_filenames]
            self.raw_labels = [load_nifti_img(ii, dtype=np.uint8)[0] for ii in self.target_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input, _ = load_nifti_img(self.image_filenames[index], dtype=np.int16)
            target, _ = load_nifti_img(self.target_filenames[index], dtype=np.uint8)
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # handle exceptions

        

        # if self.im_dim != None:
        #     input = skimage.transform.resize(input, self.im_dim)

        #check_exceptions(input, target)
        if self.transform:
            input, _ = self.transform(input, np.ones(input.shape) )
            _, target = self.transform(np.ones(target.shape), target)


        # if self.im_dim != None:
        #     input = input.numpy()
        #     print("||||||  SHAPE", input.shape)
        #     input = skimage.transform.resize(input, self.im_dim)
        #     print("||||||  SHAPE", input.shape)
        #     input = torch.Tensor(input)
        #     print("||||||  SHAPE", input.shape)

        print("shapes : ", input.shape, target.shape)

        return input, target

    def __len__(self):
        return len(self.image_filenames)