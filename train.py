import os
import time

from tqdm import tqdm

from sarlnet.data.build_dataset import build_dataset, build_train_loader
from sarlnet.models.sarlnet_model import SARLNetModel
from sarlnet.utils.img_util import save_img
from sarlnet.utils.logger import get_root_logger
from sarlnet.utils.options import get_device, mkdir_and_rename, parse_options, set_random_seed


def main():
    opt = parse_options()
    set_random_seed(opt.get("manual_seed", 123))
    root = opt["path"].get("experiments_root", os.path.join("experiments", opt["name"]))
    mkdir_and_rename(root)
    logger = get_root_logger(os.path.join(root, "train.log"))
    device = get_device()
    logger.info(f"Using device: {device}")

    train_loader = build_train_loader(opt)
    val_dataset = None
    if "val" in opt.get("datasets", {}) and opt["datasets"]["val"].get("dataroot_lq"):
        try:
            val_dataset = build_dataset(opt["datasets"]["val"])
        except FileNotFoundError:
            logger.info("Validation dataset not found; validation is skipped.")

    model = SARLNetModel(opt, device)
    model.load_network(opt.get("path", {}).get("pretrain_network_g"))

    total_epoch = opt["train"].get("total_epoch", 100)
    print_freq = opt.get("logger", {}).get("print_freq", 20)
    save_freq = opt.get("logger", {}).get("save_checkpoint_freq", 5)
    val_freq = opt.get("val", {}).get("val_freq", 1)
    current_iter = 0

    for epoch in range(1, total_epoch + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epoch}")
        for data in pbar:
            current_iter += 1
            model.feed_data(data)
            model.optimize_parameters(epoch)
            if current_iter % print_freq == 0:
                msg = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in model.log_dict.items())
                pbar.set_postfix_str(msg)
        model.scheduler.step()

        if val_dataset is not None and epoch % val_freq == 0:
            scores_sum = {}
            count = min(len(val_dataset), 16)
            for i in range(count):
                sr, scores = model.validate_batch({k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v for k, v in val_dataset[i].items()})
                for k, v in scores.items():
                    scores_sum[k] = scores_sum.get(k, 0.0) + float(v)
                if opt.get("val", {}).get("save_img", True) and i < 4:
                    save_img(sr, os.path.join(root, "visualization", f"epoch{epoch:03d}_{i}.png"))
            logger.info("Validation " + ", ".join(f"{k}: {v / count:.4f}" for k, v in scores_sum.items()))

        if epoch % save_freq == 0 or epoch == total_epoch:
            model.save(epoch, current_iter, root)
            logger.info(f"Saved checkpoint at epoch {epoch}.")

    logger.info(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S')}.")


if __name__ == "__main__":
    main()

