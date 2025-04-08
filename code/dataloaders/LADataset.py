import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose
from torchio import transforms as tiot
from copy import deepcopy

class Pancreas(Dataset):
    """ Pancreas Dataset """
    def __init__(self, data_dir, list_dir, split, reverse=False, logging=None, augment=False):
        self.data_dir = data_dir + "/pancreas_data"
        self.list_dir = list_dir
        self.split = split
        self.reverse = reverse
        self.augment = augment

        tr_transform = Compose([
            RandomCrop((96, 96, 96)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((96, 96, 96)),
            ToTensor()
        ])


        if split == 'train_lab':
            data_path = os.path.join(list_dir,'train_lab_12.txt')
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = os.path.join(list_dir,'train_unlab_12.txt')
            self.transform = tr_transform
            print("unlab transform")            
        else:
            data_path = os.path.join(list_dir,'test.txt')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()


        self.image_list = [item.replace('\n', '') for item in self.image_list]
        self.image_list = [os.path.join(self.data_dir, f"PANCREAS_{item}") for item in self.image_list]

        logging.info("{} set: total {} samples".format(split, len(self.image_list)))
        logging.info("total {} samples".format(self.image_list))

    def __len__(self):
        # return len(self.image_list)
        if (self.split == "train_lab") | (self.split == "train_unlab"):
            return len(self.image_list) * 10
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:].astype(np.float32)
        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        img = deepcopy(image_)
        if self.augment:
            img_s = deepcopy(image_)
            if random.random() < 0.8:
                img_s = tiot.RandomBiasField()(img_s)
                img_s = tiot.RandomGamma((-0.5, 0.5))(img_s)
            if random.random() < 0.5:
                img_s = tiot.RandomBlur((0.1, 2))(img_s)
            return img.float(), img_s.float(), label_.long()

        return image_.float(), label_.long()

    """ LA Dataset """
class LAHeart(Dataset):

    def __init__(self, data_dir, list_dir, split, reverse=False, logging=None, augment=False):
        self.data_dir = data_dir + "/2018LA_Seg_Training Set"
        self.list_dir = list_dir
        self.split = split
        self.reverse = reverse
        self.augment = augment

        tr_transform = Compose([
            RandomCrop((112, 112, 80)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((112, 112, 80)),
            ToTensor()
        ])

        if split == 'train_lab':
            data_path = os.path.join(list_dir,'train_lab8.txt')
            self.transform = tr_transform
        elif split == 'train_unlab':
            data_path = os.path.join(list_dir,'train_unlab8.txt')
            self.transform = tr_transform
            print("unlab transform")            
        else:
            data_path = os.path.join(list_dir,'test.txt')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        self.image_list = [os.path.join(self.data_dir, item, "mri_norm2.h5") for item in self.image_list]

        logging.info("{} set: total {} samples".format(split, len(self.image_list)))
        logging.info("total {} samples".format(self.image_list))

    def __len__(self):
        if (self.split == "train_lab") | (self.split == "train_unlab"):
            return len(self.image_list) * 10
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        if self.reverse:
            image_path = self.image_list[len(self.image_list) - idx % len(self.image_list) - 1]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        img = deepcopy(image_)
        if self.augment:
            img_s = deepcopy(image_)
            if random.random() < 0.5:
                img_s = tiot.RandomBiasField()(img_s)
            else:
                img_s = tiot.RandomGamma((-0.5, 0.5))(img_s)
            return img.float(), img_s.float(), label_.long()
        
        return img.float(), label_.long()

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, samples):
        image, label = samples[0], samples[1]
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        samples = image, label
        return samples


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]


if __name__ == '__main__':
    data_dir = '../../../Datasets/LA_dataset'
    list_dir = '../datalist/LA'
    labset = LAHeart(data_dir, list_dir,split='lab')
    unlabset = LAHeart(data_dir,list_dir,split='unlab')
    trainset = LAHeart(data_dir,list_dir,split='train')
    testset = LAHeart(data_dir, list_dir,split='test')

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    train_sample = trainset[0] 
    test_sample = testset[0]

    print(len(labset), lab_sample['image'].shape, lab_sample['label'].shape)  # 16 torch.Size([1, 112, 112, 80]) torch.Size([112, 112, 80])
    print(len(unlabset), unlab_sample['image'].shape, unlab_sample['label'].shape) # 64 torch.Size([1, 112, 112, 80]) torch.Size([112, 112, 80])
    print(len(trainset), train_sample['image'].shape, train_sample['label'].shape) # 80 torch.Size([1, 112, 112, 80]) torch.Size([112, 112, 80])
    print(len(testset), test_sample['image'].shape, test_sample['label'].shape) # 20 torch.Size([1, 112, 112, 80]) torch.Size([112, 112, 80])


    labset = LAHeart(data_dir, list_dir,split='lab', aug_times=5)
    unlabset = LAHeart(data_dir,list_dir,split='unlab', aug_times=5)
    trainset = LAHeart(data_dir,list_dir,split='train', aug_times=5)
    testset = LAHeart(data_dir, list_dir,split='test', aug_times=5)

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    train_sample = trainset[0] 
    test_sample = testset[0]

    print(len(labset), lab_sample['image'].shape, lab_sample['label'].shape)  # 80 torch.Size([1, 112, 112, 80]) torch.Size([112, 112, 80])
    print(len(unlabset), unlab_sample['image'].shape, unlab_sample['label'].shape) # 320 torch.Size([1, 112, 112, 80]) torch.Size([112, 112, 80])
    print(len(trainset), train_sample['image'].shape, train_sample['label'].shape) # 400 torch.Size([1, 112, 112, 80]) torch.Size([112, 112, 80])
    print(len(testset), test_sample['image'].shape, test_sample['label'].shape) # 20 torch.Size([1, 112, 112, 80]) torch.Size([112, 112, 80])