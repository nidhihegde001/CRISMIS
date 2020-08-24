from dataset import Cosmic_Detection
from torchvision import datasets, models, transforms
import torch

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}



'''
Provide path to dataset
Format: PascalVOC

              root
  /     /            \      \ 
train train_annots  valid  valid_annots

'''
data_dir = 'root/'

crismis_dataset = {x: Cosmic_Detection(None, x, data_transforms[x]) 
                  for x in ['train', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(crismis_dataset[x], batch_size=1,
                                             shuffle=True, num_workers=0, pin_memory = True)
              for x in ['train', 'valid']}

dataset_sizes = {x: len(crismis_dataset[x]) for x in ['train', 'valid']}