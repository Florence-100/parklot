import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from typing import Sequence

class SubsetSampler(Sampler[int]):
    indices = Sequence[int]
    
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices
        
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

def subset_sampler(dataset, keep_ratio=0.1, random=False):
    n = len(dataset)
    m = int(n * keep_ratio)
    indices = np.random.permutation(n)[:m]
    return SubsetRandomSampler(indices) if random else SubsetSampler(indices)


if __name__ == "__main__":
    
    from torch.utils.data import DataLoader, Dataset
    class ToyDataset(Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
    toydata = ToyDataset(np.arange(50))
    sampler = SubsetSampler([2, 5, 7, 10, 11, 20, 23, 24])
    toydataloder = DataLoader(toydata, batch_size=5, sampler=sampler)