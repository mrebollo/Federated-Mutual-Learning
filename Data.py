import torch
import os.path
from torchvision.datasets import utils, MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from PIL import Image

from Partitions import partition_train_indices


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        utils.makedir_exist_ok(self.raw_folder)
        utils.makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)


def build_loader(dataset, batch_size, ratio, seed=13):
    if ratio >= 1.0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_samples = max(1, int(len(dataset) * ratio))
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)


def Dataset(args):
    trainset, testset = None, None
    print("Loading dataset {}...".format(args.dataset))
    print("Download: {}".format(args.download))

    if args.dataset in ('cifar10', 'cifar100'):
        if args.dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            args.classes = 10
        else:
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
            args.classes = 100

        tra_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if args.dataset == 'cifar10':
            trainset = CIFAR10(root="~/data", train=True, download=args.download, transform=tra_trans)
            testset = CIFAR10(root="~/data", train=False, download=args.download, transform=val_trans)
        elif args.dataset == 'cifar100':
            trainset = CIFAR100(root="~/data", train=True, download=args.download, transform=tra_trans)
            testset = CIFAR100(root="~/data", train=False, download=args.download, transform=val_trans)

    elif args.dataset in ('femnist', 'mnist'):
        tra_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        val_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if args.dataset == 'femnist':
            trainset = FEMNIST(root='~/data', train=True, download=args.download, transform=tra_trans)
            testset = FEMNIST(root='~/data', train=False, download=args.download, transform=val_trans)
        if args.dataset == 'mnist':
            trainset = MNIST(root='~/data', train=True, download=args.download, transform=tra_trans)
            testset = MNIST(root='~/data', train=False, download=args.download, transform=val_trans)

    return trainset, testset


class Data(object):

    def __init__(self, args):
        self.args = args
        self.trainset, self.testset = None, None
        trainset, testset = Dataset(args)
        self.trainset, self.testset = trainset, testset

        train_indices = partition_train_indices(
            args.dataset,
            trainset.targets,
            args.split,
            args.iid,
            seed=13,
        )

        self.test_all = build_loader(testset, args.batchsize, 1.0)
        self.train_loader = [DataLoader(Subset(trainset, idx), batch_size=args.batchsize, shuffle=True, num_workers=0)
                             for idx in train_indices]
        self.test_loader = build_loader(testset, args.batchsize, args.val_ratio)
