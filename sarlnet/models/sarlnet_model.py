import os

import torch
import torch.nn.functional as F

from sarlnet.archs.sarlnet_arch import SARLNet
from sarlnet.losses.perceptual_loss import PerceptualLoss
from sarlnet.metrics.iqa import IQAEvaluator
from sarlnet.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from sarlnet.rewards.pareto_reward import ParetoReward


class SARLNetModel:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        net_opt = dict(opt["network_g"])
        net_opt.pop("type", None)
        self.net_g = SARLNet(**net_opt).to(device)
        train_opt = opt.get("train", {})
        self.optimizer = torch.optim.AdamW(
            self.net_g.parameters(),
            lr=train_opt.get("lr", 1e-3),
            betas=tuple(train_opt.get("betas", [0.9, 0.999])),
            weight_decay=train_opt.get("weight_decay", 1e-4),
        )
        total_epoch = max(train_opt.get("total_epoch", 100), 1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epoch)
        reward_opt = train_opt.get("reward", {})
        metrics = reward_opt.get("metrics", ["maniqa", "musiq", "clipiqa"])
        self.iqa = IQAEvaluator(metrics, device, reward_opt.get("use_pyiqa", True))
        self.reward = ParetoReward(
            len(metrics),
            reward_opt.get("metric_weights", [1 / len(metrics)] * len(metrics)),
            lambda_dir=reward_opt.get("lambda_dir", 0.05),
            tau=reward_opt.get("tau", 0.1),
            eps=reward_opt.get("eps", 1e-8),
            device=device,
        )
        self.perceptual = PerceptualLoss(use_vgg=False).to(device)
        self.log_dict = {}
        self.lq = None
        self.gt = None
        self.is_synth = None

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        self.gt = data["gt"].to(self.device)
        self.is_synth = data.get("is_synth", torch.ones(self.lq.size(0), dtype=torch.bool)).to(self.device)

    def perceptual_weight(self, epoch):
        train_opt = self.opt.get("train", {})
        warmup = train_opt.get("warmup_epoch", 20)
        total = max(train_opt.get("total_epoch", 100) - warmup, 1)
        base = train_opt.get("perceptual_weight", 0.1)
        if epoch <= warmup:
            return base
        return base * max(0.0, 1.0 - (epoch - warmup) / total)

    def optimize_parameters(self, epoch):
        if epoch <= self.opt["train"].get("warmup_epoch", 20):
            return self._optimize_warmup(epoch)
        return self._optimize_ppo(epoch)

    def _optimize_warmup(self, epoch):
        self.optimizer.zero_grad(set_to_none=True)
        sr = self.net_g(self.lq)
        pixel = F.l1_loss(sr, self.gt)
        per = self.perceptual(sr, self.gt).mean()
        loss = pixel + self.perceptual_weight(epoch) * per
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt["train"].get("grad_clip", 1.0))
        self.optimizer.step()
        self.log_dict = {"loss": loss.item(), "pixel": pixel.item(), "per": per.item(), "stage": "warmup"}

    def _optimize_ppo(self, epoch):
        ppo_opt = self.opt["train"].get("ppo", {})
        gamma = ppo_opt.get("gamma", 0.99)
        clip_range = ppo_opt.get("clip_range", 0.2)
        value_weight = ppo_opt.get("value_weight", 0.5)
        entropy_weight = ppo_opt.get("entropy_weight", 0.01)
        ppo_epoch = ppo_opt.get("ppo_epoch", 4)
        pref0 = torch.full((self.lq.size(0), 3), 1 / 3, device=self.device)
        sr, traj, init, lq_up = self.net_g.rollout(self.lq, pref0, explore=True)

        rewards = []
        prefs = []
        prev_rewards = []
        last_reward = torch.zeros(self.lq.size(0), device=self.device)
        with torch.no_grad():
            for item in traj:
                prev_img = self.net_g.decode(item["x"])
                next_img = self.net_g.decode(item["x_next"])
                prev_scores = self.iqa(prev_img)
                next_scores = self.iqa(next_img)
                per_penalty = None
                if self.is_synth.any():
                    per_penalty = self.perceptual(next_img, self.gt)
                    per_penalty = per_penalty * self.is_synth.float()
                reward, pref = self.reward.compute(
                    prev_scores,
                    next_scores,
                    perceptual_penalty=per_penalty,
                    perceptual_weight=self.perceptual_weight(epoch),
                )
                rewards.append(reward.detach())
                prefs.append(pref.detach())
                prev_rewards.append(last_reward.detach())
                last_reward = reward.detach()

        returns = []
        running = torch.zeros_like(rewards[-1])
        for reward in reversed(rewards):
            running = reward + gamma * running
            returns.insert(0, running)
        returns = torch.stack(returns)
        old_values = torch.stack([item["value"].detach() for item in traj])
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        total_loss_value = 0.0
        for _ in range(ppo_epoch):
            self.optimizer.zero_grad(set_to_none=True)
            policy_losses = []
            value_losses = []
            entropies = []
            for i, item in enumerate(traj):
                new_logp, value, entropy = self.net_g.recompute_logp_value(item, init, lq_up, prefs[i], prev_rewards[i])
                ratio = torch.exp(new_logp - item["old_logp"])
                adv = advantages[i]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
                policy_losses.append(-torch.minimum(unclipped, clipped).mean())
                value_losses.append(F.mse_loss(value, returns[i]))
                entropies.append(entropy.mean())
            loss = torch.stack(policy_losses).mean()
            loss = loss + value_weight * torch.stack(value_losses).mean()
            loss = loss - entropy_weight * torch.stack(entropies).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt["train"].get("grad_clip", 1.0))
            self.optimizer.step()
            total_loss_value = loss.item()

        self.log_dict = {
            "loss": total_loss_value,
            "reward": torch.stack(rewards).mean().item(),
            "sr_mean": sr.detach().mean().item(),
            "stage": "ppo",
        }

    def validate_batch(self, data):
        self.net_g.eval()
        with torch.no_grad():
            lq = data["lq"].to(self.device)
            gt = data["gt"].to(self.device)
            sr = self.net_g(lq)
            scores = {"psnr": calculate_psnr(sr, gt), "ssim": calculate_ssim(sr, gt)}
            iqa_scores = self.iqa(sr).mean(dim=0).detach().cpu().tolist()
            for name, score in zip(self.opt.get("train", {}).get("reward", {}).get("metrics", ["maniqa", "musiq", "clipiqa"]), iqa_scores):
                scores[name] = score
        self.net_g.train()
        return sr, scores

    def save(self, epoch, current_iter, root):
        model_dir = os.path.join(root, "models")
        state_dir = os.path.join(root, "training_states")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)
        torch.save(self.net_g.state_dict(), os.path.join(model_dir, f"net_g_{epoch:03d}.pth"))
        torch.save(self.net_g.state_dict(), os.path.join(model_dir, "net_g_latest.pth"))
        torch.save(
            {"epoch": epoch, "iter": current_iter, "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()},
            os.path.join(state_dir, "latest.state"),
        )

    def load_network(self, path, strict=True):
        if path and os.path.isfile(path):
            state = torch.load(path, map_location=self.device)
            self.net_g.load_state_dict(state, strict=strict)

