import torch
import torch.nn.functional as F


class IQAEvaluator:
    def __init__(self, metric_names, device, use_pyiqa=True):
        self.metric_names = [m.lower() for m in metric_names]
        self.device = device
        self.models = {}
        if use_pyiqa:
            try:
                import pyiqa

                for name in self.metric_names:
                    pyiqa_name = "clipiqa" if name in ("clip-iqa", "clipiqa") else name
                    self.models[name] = pyiqa.create_metric(pyiqa_name, device=device).eval()
            except Exception:
                self.models = {}

    @torch.no_grad()
    def __call__(self, img):
        img = img.detach().clamp(0, 1)
        if not self.metric_names:
            return img.new_zeros((img.size(0), 0))
        scores = []
        for name in self.metric_names:
            if name in self.models:
                value = self.models[name](img)
                scores.append(value.reshape(img.size(0), -1).mean(dim=1))
            else:
                scores.append(self._proxy_score(img, name))
        return torch.stack(scores, dim=1)

    def _proxy_score(self, img, name):
        gray = img.mean(dim=1, keepdim=True)
        dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        sharp = (dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3))) * 0.5
        contrast = gray.flatten(1).std(dim=1)
        mean = gray.flatten(1).mean(dim=1)
        saturation = img.flatten(2).std(dim=1).mean(dim=1)
        if name == "musiq":
            return (0.55 * contrast + 0.45 * sharp).clamp(0, 1)
        if name in ("clipiqa", "clip-iqa"):
            natural = 1.0 - (mean - 0.5).abs() * 2.0
            return (0.45 * natural + 0.35 * saturation + 0.2 * sharp).clamp(0, 1)
        return (0.5 * sharp + 0.5 * contrast).clamp(0, 1)

