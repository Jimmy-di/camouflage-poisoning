import torchvision
# Overwrite getitem method to obtain the index of the images when iterating through the images

from torch.utils.data import Dataset, DataLoader


class CIFAR10(Dataset):
    def __init__(self, train, transform):
        self.cifar10 = torchvision.datasets.CIFAR10(
                        root='./data', train=train, download=True, transform=transform)
        self.targets = self.cifar10.targets
        self.classes = self.cifar10.classes
        self.data = self.cifar10.data
        
  
    # Overloaded the getitem method to return index as well
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        return data, target, index
    
    # Method to get all images' indices from a certain class without iterating through the loader
    def get_index(self, target_label):
      index_list = []
      for index, label in enumerate(self.targets):
        if label == target_label:
          index_list.append(index)
      return index_list

    def __len__(self):
        return len(self.cifar10)

    def remove(self, remove_list):
      mask = np.ones(len(self.cifar10), dtype=bool)
      mask[remove_list] = False
      data = self.data[mask]
