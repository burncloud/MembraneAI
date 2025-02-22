from typing import TypeVar, Optional, Iterable
from . import Sampler, Dataset

T_co = TypeVar('T_co', covariant=True)
class DistributedSampler(Sampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int]=..., rank: Optional[int]=...): ...
    def __iter__(self) -> Iterable[int]: ...
    def __len__(self) -> int: ...
    def set_epoch(self, epoch: int) -> None: ...
