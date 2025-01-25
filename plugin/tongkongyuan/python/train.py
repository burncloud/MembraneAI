import os
import sys


import os, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from PIL import Image
from PIL import ImageDraw
import json
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc

from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch.fpn import FPN
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from typing import Optional, Union
from skimage.measure import find_contours


from utils import *
from loss import lovasz_hinge


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


class LossFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = FocalLoss(gamma=2)

    def forward(self, x, gt):
        loss = self.fn(x, gt)
        loss += lovasz_hinge(x.squeeze(), gt.squeeze())
        return loss

loss_fn = LossFn()


class EyeDataset(Dataset):
    def __init__(self, image_names, transforms=None, test_mode=False):
        self.image_names = image_names
        self.transforms = transforms
        self.test_mode = test_mode

    def __len__(self):
        return len(self.image_names)

    def parse_json(self, json_name, image_shape):
        """
        todo 只处理瞳孔缘
        """
        data = json.load(open(json_name, encoding='gbk'))
        shapes = data['shapes']

        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        draw = PIL.ImageDraw.Draw(mask)

        for shape in shapes:
            if shape['label'] != "瞳孔缘": continue;
            points = shape['points']
            xy = [tuple(point) for point in points]
            assert len(xy) > 2, "Polygon must have points more than 2"
            draw.polygon(xy=xy, outline=1, fill=1)

        mask = np.array(mask, dtype=np.uint8)
        return mask

    def normalize(self, image):
        image = image.astype(np.float32)
        image = image / 255.
        image -= 0.5
        image /= 0.225
        return np.transpose(image, axes=[2, 0, 1])

    def __getitem__(self, item):
        image_name = self.image_names[item]
        json_name = image_name.replace("jpg", "json")
        mask_name = image_name.replace("jpg", "png")
        if not self.test_mode:
            image = np.array(Image.open(image_name))
            try:
                mask = self.parse_json(json_name, image.shape[:2])
            except FileNotFoundError:
                mask = np.array(Image.open(mask_name))
            ###########################################
            ## DEBUG, To make sure your mask is compatiable with image
            if 0:
                plt.imshow(np.concatenate([image, mask[..., None].repeat(3, axis=-1) * 255], axis=1))
                plt.show()
                plt.close()
            ###########################################

            if self.transforms is not None:
                aug = Config.valid_transforms(image=image, mask=mask)
                image = aug['image']
                mask = aug['mask']

            image = image[:, 64:448, :]
            mask = mask[:, 64:448]

            result = {
                'image': self.normalize(image),
                "mask_gt": mask[None, ...],  # [1,h,w]
                'image_org': image,
            }
        else:
            image = np.array(Image.open(image_name))
            if self.transforms is not None:
                aug = Config.valid_transforms(image=image)
                image = aug['image']
            image = image[:, 64:448, :]
            image_org = image.copy()
            result = {
                'image': self.normalize(image),
                'image_org': image_org,
            }
        return result


def run():
    image_names = glob.glob("./data/data/*.jpg")
    train_image_names, valid_image_names = make_train_val(image_names, test_size=0.2, seed=Config.seed)
    image_names2 = glob.glob("./data/data_mask/*.jpg")
    train_image_names2, valid_image_names2 = make_train_val(image_names2, test_size=0.2, seed=Config.seed)

    train_image_names += train_image_names2
    valid_image_names += valid_image_names2

    print("Train sample num is {}    Valid sample num is {} ".format(len(train_image_names), len(valid_image_names)))

    train_dataset = EyeDataset(train_image_names, transforms=Config.train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_dataset = EyeDataset(valid_image_names, transforms=Config.valid_transforms)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    model = FPN(encoder_name='mobilenet_v2', encoder_weights='imagenet', classes=1)
    model.to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, min_lr=1e-6, patience=3,
                                                           verbose=True)

    es = EarlyStopping(patience=9, mode='min', delta=1e-6)
    for epoch in range(Config.epochs):
        ## train
        model.train()
        optimizer.zero_grad()
        losses = AverageMeter()
        tk = tqdm(train_dataloader)
        for bi, data in enumerate(tk):
            images = data['image'].to(Config.device)
            gt = data['mask_gt'].to(Config.device)

            outputs = model(images)
            loss = loss_fn(outputs, gt)
            losses.update(loss.item(), n=images.size(0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tk.set_postfix(loss=losses.avg)

        ## val
        model.eval()
        tk = valid_dataloader
        losses = AverageMeter()
        with torch.no_grad():
            for bi, data in enumerate(tk):
                images = data['image'].to(Config.device)
                gt = data['mask_gt'].to(Config.device)

                outputs = model(images)
                loss = loss_fn(outputs, gt)
                losses.update(loss.item(), n=images.size(0))

        result = {
            'loss': losses.avg,
        }
        for k, v in result.items():
            print(f"{k} is : {v}")

        es(result['loss'], model, f"ckpt/model.pth")

        scheduler.step(result['loss'])

        if es.early_stop:
            print("Early Stopping Training !!!!!!!!<<<<")
            break

    image_names = glob.glob("./data/data/*.jpg")
    image_names2 = glob.glob("./data/data_mask/*.jpg")

    image_names += image_names2

    train_dataset = EyeDataset(image_names, transforms=Config.train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-5

    model.load_state_dict(torch.load("ckpt/model.pth"))
    for epoch in range(5):
        ## train
        model.train()
        optimizer.zero_grad()
        losses = AverageMeter()
        tk = tqdm(train_dataloader)
        for bi, data in enumerate(tk):
            images = data['image'].to(Config.device)
            gt = data['mask_gt'].to(Config.device)

            outputs = model(images)
            loss = loss_fn(outputs, gt)
            losses.update(loss.item(), n=images.size(0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tk.set_postfix(loss=losses.avg)

    torch.save(model.state_dict(), "ckpt/model_0.pth")
    for epoch in range(4):
        ## train
        model.train()
        optimizer.zero_grad()
        losses = AverageMeter()
        tk = tqdm(train_dataloader)
        for bi, data in enumerate(tk):
            images = data['image'].to(Config.device)
            gt = data['mask_gt'].to(Config.device)

            outputs = model(images)
            loss = loss_fn(outputs, gt)
            losses.update(loss.item(), n=images.size(0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tk.set_postfix(loss=losses.avg)
        torch.save(model.state_dict(), f"ckpt/model_{epoch + 1}.pth")

    state_dict = {k: v / 5. for k, v in torch.load("ckpt/model_0.pth").items()}
    for e in range(4):
        temp_dict = torch.load(f"ckpt/model_{e + 1}.pth")
        for k, v in temp_dict.items():
            state_dict[k] += v / 5.
    torch.save(state_dict, "ckpt/model_avg.pth")



if __name__ == '__main__':
    seed_torch(42)

    torch.cuda.empty_cache()

    run()




