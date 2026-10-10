import torch
import torch.nn.functional as F


class RunningNormalizer:
    def __init__(self, num_metrics, eps=1e-8, device="cpu"):
        self.min = torch.full((num_metrics,), float("inf"), device=device)
        self.max = torch.full((num_metrics,), float("-inf"), device=device)
        self.records = []
        self.eps = eps

    def update(self, scores):
        scores = scores.detach()
        self.min = torch.minimum(self.min, scores.amin(dim=0))
        self.max = torch.maximum(self.max, scores.amax(dim=0))
        self.records.append(self.normalize(scores).detach().cpu())
        if len(self.records) > 2048:
            self.records = self.records[-2048:]

    def normalize(self, scores):
        return (scores - self.min) / (self.max - self.min + self.eps)

    def pareto_best(self, fallback):
        if not self.records:
            return fallback.detach()
        records = torch.cat(self.records, dim=0).to(fallback.device)
        return records.max(dim=0).values.unsqueeze(0).expand_as(fallback)


class ParetoReward:
    def __init__(self, num_metrics, metric_weights, lambda_dir=0.05, tau=0.1, eps=1e-8, device="cpu"):
        self.metric_weights = torch.tensor(metric_weights, dtype=torch.float32, device=device)
        self.metric_weights = self.metric_weights / self.metric_weights.sum()
        self.lambda_dir = lambda_dir
        self.tau = tau
        self.eps = eps
        self.normalizer = RunningNormalizer(num_metrics, eps=eps, device=device)

    def compute(self, prev_scores, next_scores, perceptual_penalty=None, perceptual_weight=0.0):
        self.normalizer.update(torch.cat([prev_scores, next_scores], dim=0))
        prev_norm = self.normalizer.normalize(prev_scores)
        next_norm = self.normalizer.normalize(next_scores)
        best = self.normalizer.pareto_best(prev_norm)
        deficiency = (best - prev_norm).clamp_min(0) + self.eps
        w = deficiency / deficiency.sum(dim=1, keepdim=True)
        delta = next_norm - prev_norm
        q = F.softmax(delta / self.tau, dim=1)
        kl = (w * ((w + self.eps).log() - (q + self.eps).log())).sum(dim=1)
        improve = (w * delta).sum(dim=1)
        quality = (next_norm * self.metric_weights).sum(dim=1)
        reward = improve + quality - self.lambda_dir * kl
        if perceptual_penalty is not None and perceptual_weight > 0:
            reward = reward - perceptual_weight * perceptual_penalty.detach().view_as(reward)
        return reward, w.detach()

