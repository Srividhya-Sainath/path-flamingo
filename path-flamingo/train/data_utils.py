"""
Util functions for initializing webdataset objects
"""

from math import gcd
from dataclasses import dataclass
from multiprocessing import Value
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value
    
# Helper function to compute LCM
## Maine add kiya hai. 
## Yeh TCGA aur GTEX dataset ka number same hona chahiye har epoch mein. Isliye
def lcm(x, y):
    return (x * y) // gcd(x, y)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
