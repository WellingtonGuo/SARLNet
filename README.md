# SARLNet training code

This folder contains a BasicSR-style, trainable implementation of the SARLNet protocol described in the paper:

- x4 remote-sensing super-resolution
- synthetic paired warm-start for 20 epochs
- PPO actor-critic reward fine-tuning
- action space: restart sampling plus U-Net feature manipulation
- reward: Pareto-weighted MANIQA/MUSIQ/CLIP-IQA scores with optional perceptual shaping
- evaluation: PSNR/SSIM and no-reference IQA scores

The code is intentionally self-contained. If `pyiqa` is installed, MANIQA/MUSIQ/CLIP-IQA are used directly. Otherwise the training script falls back to deterministic proxy IQA scores so the full pipeline remains runnable.

## Train

Edit paths in `options/train_sarlnet.yml`, then run:

```bash
python train.py -opt options/train_sarlnet.yml
```

Expected paired directory layout:

```text
datasets/AID/train/LR/*.png
datasets/AID/train/HR/*.png
datasets/real/train/LR/*.png
```

HR files are optional for real-world data. For synthetic data, LR/HR filenames should match.

## Test

```bash
python test.py -opt options/test_sarlnet.yml
```

The default options are conservative so the model can run on ordinary hardware. For the paper setting, use `gt_size: 512`, `lq_size: 128`, `batch_size_per_gpu: 4`, `total_epoch: 100`, and `warmup_epoch: 20`.

