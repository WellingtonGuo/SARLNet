import os

from sarlnet.data.build_dataset import build_dataset
from sarlnet.models.sarlnet_model import SARLNetModel
from sarlnet.utils.img_util import save_img
from sarlnet.utils.logger import get_root_logger
from sarlnet.utils.options import get_device, mkdir_and_rename, parse_options


def main():
    opt = parse_options()
    root = opt.get("path", {}).get("results_root", os.path.join("results", opt["name"]))
    mkdir_and_rename(root)
    logger = get_root_logger(os.path.join(root, "test.log"))
    device = get_device()
    dataset = build_dataset(opt["datasets"]["test"])
    train_like_opt = {
        "train": {"reward": {"metrics": [m for m in opt.get("val", {}).get("metrics", []) if m not in ("psnr", "ssim")]}},
        "network_g": opt["network_g"],
    }
    model = SARLNetModel({**opt, **train_like_opt}, device)
    model.load_network(opt.get("path", {}).get("pretrain_network_g"), strict=True)

    sums = {}
    for i in range(len(dataset)):
        data = dataset[i]
        batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v for k, v in data.items()}
        sr, scores = model.validate_batch(batch)
        for k, v in scores.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        if opt.get("val", {}).get("save_img", True):
            save_img(sr, os.path.join(root, "images", f"{data['key']}.png"))
    logger.info("Test " + ", ".join(f"{k}: {v / len(dataset):.4f}" for k, v in sums.items()))


if __name__ == "__main__":
    main()

