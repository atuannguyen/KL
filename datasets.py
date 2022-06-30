
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder, USPS, SVHN
from torchvision.transforms.functional import rotate

#from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
#from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "RotatedMNIST",
    # Big images
    "PACS",
    "VisDA17",
    "MNISTUSPS",
    "SVHNMNIST"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    EPOCHS = 100             # Default, if train with epochs, check performance every epoch.
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                                               resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = self.augment_transform
            else:
                env_transform = self.transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)


class PACS(MultipleEnvironmentImageFolder):
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class VisDA17(MultipleEnvironmentImageFolder):
    EPOCHS = 20
    ENVIRONMENTS = ["train", "validation"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "visda17/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class MNISTUSPS(MultipleDomainDataset):
    ENVIRONMENTS = ["MNIST", "USPS"]
    def __init__(self, root, test_envs, hparams):
        self.datasets = []
        self.input_shape = (1, 28, 28,)
        self.num_classes = 10
        
        transform = transforms.Compose([
                        transforms.Resize((28,28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])


        for envs in self.ENVIRONMENTS:
            dataset_tr = eval(envs)(root, train=True, transform=transform, download=True)
            dataset_te = eval(envs)(root, train=False, transform=transform, download=True)
            self.datasets.append(torch.utils.data.ConcatDataset([dataset_tr,dataset_te]))

class SVHNMNIST(MultipleDomainDataset):
    ENVIRONMENTS = ["SVHN", "MNIST"]
    EPOCHS = 100
    def __init__(self, root, test_envs, hparams):
        self.datasets = []
        self.input_shape = (3, 28, 28,)
        self.num_classes = 10
        
        transform_gray = transforms.Compose([
                        transforms.Resize((28,28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
        transform = transforms.Compose([
                        transforms.Resize((28,28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])


        # SVHN
        dataset_tr = SVHN(root, split='train', transform=transform, download=True)
        dataset_te = SVHN(root, split='test', transform=transform, download=True)
        self.datasets.append(torch.utils.data.ConcatDataset([dataset_tr,dataset_te]))

        # MNIST
        dataset_tr = MNIST(root, train=True, transform=transform_gray, download=True)
        dataset_te = MNIST(root, train=False, transform=transform_gray, download=True)
        self.datasets.append(torch.utils.data.ConcatDataset([dataset_tr,dataset_te]))



