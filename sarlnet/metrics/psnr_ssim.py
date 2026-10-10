import math

import torch
import torch.nn.functional as F


def calculate_psnr(sr, gt, eps=1e-8):
    sr = sr.detach().clamp(0, 1)
    gt = gt.detach().clamp(0, 1)
    mse = F.mse_loss(sr, gt).item()
    return 20 * math.log10(1.0 / math.sqrt(mse + eps))


def calculate_ssim(sr, gt, eps=1e-8):
    sr = sr.detach().clamp(0, 1)
    gt = gt.detach().clamp(0, 1)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(sr, 11, 1, 5)
    mu_y = F.avg_pool2d(gt, 11, 1, 5)
    sigma_x = F.avg_pool2d(sr * sr, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(gt * gt, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(sr * gt, 11, 1, 5) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return ssim.mean().item()

