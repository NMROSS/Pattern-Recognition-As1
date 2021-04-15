from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose

train_set = 'Data/mnist-png-format/train'
test_set = 'Data/mnist-png-format/train'


def load_mnist(batch_size, workers):
'''
      Load MNIST data returns tuple (Train, Validation, Test)
'''
    transforms = Compose([
                        ToTensor(),           
                        Normalize(mean=(0.5, 0.5, 0.5,), std=(0.5, 0.5, 0.5)), 
                        ])

    train_dataset = ImageFolder(train_set, transform=transforms)
    test_dataset = ImageFolder(train_set, transform=transforms)

    # Split Train dataset set into two (Train=85/Validation=15% split)
    train_size = int(len(train_dataset) * 0.85) 
    validation_size = (len(train_dataset) - train_size) 
    train, validation = random_split(train_dataset, [train_size, validation_size])
    
    train_dl = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = workers)
    val_dl   = DataLoader(validation, batch_size = batch_size, shuffle = True, num_workers = workers)
    test_dl  = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = workers)

    
    return train_dl, val_dl, test_dl