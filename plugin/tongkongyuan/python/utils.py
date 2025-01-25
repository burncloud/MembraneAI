import math
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import PIL, albumentations as A, cv2
import random,os,numpy as np, torch
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
SMOOTH = 1e-6

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Config:
    # config scope
    seed = 2020
    device = "cuda"

    show_valid = True

    epochs = 200
    batch_size = 64

    height = 288
    width = 512



    train_transforms = A.Compose([
        A.Resize(height=height, width=width, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=1, border_mode=cv2.BORDER_CONSTANT,
                           value=0),
        A.OneOf([
            A.GridDistortion(distort_limit=0.2, border_mode=cv2.BORDER_CONSTANT, p=1),
            A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, p=1)
        ], p=0.3),

        A.OneOf([
            A.RandomGamma(gamma_limit=(60, 120), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1)
        ], p=0.5),
    ])

    valid_transforms = A.Compose([
        A.Resize(height=height, width=width, p=1),
    ])


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class EarlyStopping:
    def __init__(self, patience=20, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, states_dict, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, states_dict, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, states_dict, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, saved_model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(saved_model.state_dict(), model_path)
        self.val_score = epoch_score


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_kfold(n_splits=5):
    pass
    # todo
    raise NotImplementedError


def make_train_val(image_names, test_size=0.2, seed=42):
    dummy_y = [1] * len(image_names)
    X_train, X_test, _, _ = train_test_split(image_names, dummy_y, test_size = test_size, random_state = seed)
    return X_train, X_test


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch