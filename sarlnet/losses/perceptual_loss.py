import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, use_vgg=True):
        super().__init__()
        self.vgg = None
        if use_vgg:
            try:
                from torchvision.models import vgg16, VGG16_Weights

                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[:16].eval()
                for p in vgg.parameters():
                    p.requires_grad = False
                self.vgg = vgg
            except Exception:
                self.vgg = None

    def forward(self, sr, gt):
        sr = sr.clamp(0, 1)
        gt = gt.clamp(0, 1)
        if self.vgg is None:
            sr_low = F.avg_pool2d(sr, 4, 4)
            gt_low = F.avg_pool2d(gt, 4, 4)
            return F.l1_loss(sr_low, gt_low, reduction="none").flatten(1).mean(dim=1)
        self.vgg = self.vgg.to(sr.device)
        return F.l1_loss(self.vgg(sr), self.vgg(gt), reduction="none").flatten(1).mean(dim=1)

