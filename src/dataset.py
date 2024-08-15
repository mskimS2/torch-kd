import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from typing import Any, Dict


def get_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    if config["dataset"] == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    elif config["dataset"] == "mnist":
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        ),
    }
