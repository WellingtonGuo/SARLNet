from torch.utils.data import ConcatDataset, DataLoader

from sarlnet.data.paired_image_dataset import PairedImageDataset


def build_dataset(opt):
    dataset_type = opt.get("type", "PairedImageDataset")
    if dataset_type != "PairedImageDataset":
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return PairedImageDataset(opt)


def build_train_loader(opt):
    datasets = []
    for key in ("train_synth", "train_real"):
        if key in opt["datasets"] and opt["datasets"][key].get("dataroot_lq"):
            try:
                datasets.append(build_dataset(opt["datasets"][key]))
            except FileNotFoundError:
                if key == "train_synth":
                    raise
    if not datasets:
        raise RuntimeError("No train dataset is available.")
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    train_opt = opt["train"]
    return DataLoader(
        dataset,
        batch_size=train_opt.get("batch_size_per_gpu", 4),
        shuffle=True,
        num_workers=train_opt.get("num_worker_per_gpu", 4),
        pin_memory=True,
        drop_last=True,
    )

