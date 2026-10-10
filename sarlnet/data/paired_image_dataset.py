import os
import random

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from sarlnet.utils.img_util import imread, pil_to_tensor, scandir


class PairedImageDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.lq_paths = scandir(opt.get("dataroot_lq"))
        self.gt_paths = scandir(opt.get("dataroot_gt"))
        self.gt_map = {os.path.basename(p): p for p in self.gt_paths}
        self.lq_size = opt.get("lq_size")
        self.gt_size = opt.get("gt_size")
        self.use_hflip = opt.get("use_hflip", False)
        self.use_rot = opt.get("use_rot", False)
        if not self.lq_paths:
            raise FileNotFoundError(f"No LQ images found in {opt.get('dataroot_lq')}")

    def __len__(self):
        return len(self.lq_paths)

    def _resize(self, img, size):
        if size is None:
            return img
        return TF.resize(img, [size, size], interpolation=TF.InterpolationMode.BICUBIC)

    def _augment_pair(self, lq, gt):
        if self.use_hflip and random.random() < 0.5:
            lq = TF.hflip(lq)
            gt = TF.hflip(gt) if gt is not None else gt
        if self.use_rot:
            k = random.randint(0, 3)
            if k:
                angle = 90 * k
                lq = TF.rotate(lq, angle)
                gt = TF.rotate(gt, angle) if gt is not None else gt
        return lq, gt

    def __getitem__(self, index):
        lq_path = self.lq_paths[index]
        name = os.path.basename(lq_path)
        gt_path = self.gt_map.get(name)
        lq = self._resize(imread(lq_path), self.lq_size)
        gt = self._resize(imread(gt_path), self.gt_size) if gt_path else None
        lq, gt = self._augment_pair(lq, gt)
        data = {"lq": pil_to_tensor(lq), "lq_path": lq_path, "key": os.path.splitext(name)[0]}
        if gt is not None:
            data["gt"] = pil_to_tensor(gt)
            data["gt_path"] = gt_path
            data["is_synth"] = True
        else:
            data["gt"] = TF.resize(lq, [self.gt_size, self.gt_size], interpolation=TF.InterpolationMode.BICUBIC)
            data["gt"] = pil_to_tensor(data["gt"])
            data["gt_path"] = ""
            data["is_synth"] = False
        return data

