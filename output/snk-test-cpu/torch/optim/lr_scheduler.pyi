from typing import Iterable, Any, Optional
from .optimizer import Optimizer

class _LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int=...) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
    def get_lr(self) -> float: ...
    def step(self, epoch: int) -> None: ...

class LambdaLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, lr_lambda: float, last_epoch: int=...) -> None: ...

class StepLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float=..., last_epoch: int=...) -> None:...

class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, milestones: Iterable[int], gamma: float=..., last_epoch: int=...) -> None: ...

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int=...) -> None: ...

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float, last_epoch: int=...) -> None: ...

class ReduceLROnPlateau:
    in_cooldown: bool

    def __init__(self, optimizer: Optimizer, mode: str=..., factor: float=..., patience: int=..., verbose: bool=..., threshold: float=..., threshold_mode: str=..., cooldown: int=..., min_lr: float=..., eps: float=...) -> None: ...
    def step(self, metrics: Any, epoch: Optional[int]=...) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict): ...