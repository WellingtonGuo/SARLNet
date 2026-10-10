import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / max(half - 1, 1))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels=0):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.emb = nn.Linear(emb_channels, channels * 2) if emb_channels else None

    def forward(self, x, emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.emb is not None and emb is not None:
            gamma, beta = self.emb(emb).chunk(2, dim=1)
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SemanticDiscrepancyBranch(nn.Module):
    def __init__(self, channels, num_tokens=8):
        super().__init__()
        self.token_proj = nn.Conv2d(channels, num_tokens, 1)
        enc_layer = nn.TransformerEncoderLayer(channels, 4, channels * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def tokens(self, feat):
        b, c, h, w = feat.shape
        attn = self.token_proj(feat).flatten(2).softmax(dim=-1)
        flat = feat.flatten(2).transpose(1, 2)
        return torch.bmm(attn, flat)

    def forward(self, cur, init):
        cur_t = self.tokens(cur)
        init_t = self.tokens(init)
        mixed = self.encoder(torch.cat([cur_t, init_t], dim=1))
        cur_ctx = mixed[:, : cur_t.size(1)].mean(dim=1)[:, :, None, None]
        init_ctx = mixed[:, cur_t.size(1) :].mean(dim=1)[:, :, None, None]
        diff = torch.abs((cur + cur_ctx) - (init + init_ctx))
        return self.refine(diff)


class SDPB(nn.Module):
    def __init__(self, latent_channels=4, channels=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.sdb = SemanticDiscrepancyBranch(channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, channels // 2, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, 3, 1, 1),
        )
        self.out = nn.Conv2d(channels * 2, channels, 1)

    def _edge(self, x):
        gray = x.mean(dim=1, keepdim=True)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8
        ky = kx.transpose(2, 3)
        return torch.cat([F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)], dim=1)

    def forward(self, cur, init):
        cur_f = self.proj(cur)
        init_f = self.proj(init)
        mask = self.sdb(cur_f, init_f)
        edge_diff = torch.abs(self._edge(cur) - self._edge(init))
        spatial = self.spatial(edge_diff) * mask
        return self.out(torch.cat([torch.abs(cur_f - init_f), spatial], dim=1))


class ContentEncoder(nn.Module):
    def __init__(self, img_channels=3, channels=64):
        super().__init__()
        self.img = nn.Sequential(nn.Conv2d(img_channels, channels, 3, 1, 1), nn.SiLU(), ResBlock(channels))
        self.edge = nn.Sequential(nn.Conv2d(2, channels, 3, 1, 1), nn.SiLU(), ResBlock(channels))
        self.match = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, 3, 1, 1), nn.SiLU(), nn.Conv2d(channels * 2, channels * 2, 3, 1, 1))

    def _edge(self, x):
        gray = x.mean(dim=1, keepdim=True)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8
        ky = kx.transpose(2, 3)
        return torch.cat([F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)], dim=1)

    def forward(self, lq_up):
        img_f = self.img(lq_up)
        edge_f = self.edge(self._edge(lq_up))
        gamma, beta = self.match(torch.cat([img_f, edge_f], dim=1)).chunk(2, dim=1)
        return img_f * (1 + gamma) + beta


class PolicyNet(nn.Module):
    def __init__(self, latent_channels=4, cond_channels=64, hidden=128, num_restart=4, num_metrics=3):
        super().__init__()
        self.num_restart = num_restart
        in_ch = latent_channels + cond_channels
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),
            nn.SiLU(),
            ResBlock(64),
            nn.Conv2d(64, hidden, 4, 2, 1),
            nn.SiLU(),
            ResBlock(hidden),
            nn.Conv2d(hidden, hidden, 4, 2, 1),
            nn.SiLU(),
            ResBlock(hidden),
        )
        self.meta = nn.Sequential(nn.Linear(1 + 1 + num_metrics, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU())
        self.restart_logits = nn.Linear(hidden, num_restart)
        self.cont_mean = nn.Linear(hidden, 3)
        self.cont_logstd = nn.Parameter(torch.zeros(3))
        self.value = nn.Linear(hidden, 1)

    def forward(self, x, cond, t_norm, pref, prev_reward):
        h = self.enc(torch.cat([x, cond], dim=1)).mean(dim=(2, 3))
        meta = self.meta(torch.cat([t_norm[:, None], prev_reward[:, None], pref], dim=1))
        h = self.head(torch.cat([h, meta], dim=1))
        logits = self.restart_logits(h)
        mean = torch.tanh(self.cont_mean(h))
        std = self.cont_logstd.exp().clamp(0.05, 1.0).expand_as(mean)
        value = self.value(h).squeeze(1)
        return logits, mean, std, value


class ActionUNet(nn.Module):
    def __init__(self, latent_channels=4, cond_channels=64, base=64, emb_channels=128):
        super().__init__()
        self.in_conv = nn.Conv2d(latent_channels + cond_channels, base, 3, 1, 1)
        self.down = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.mid = ResBlock(base * 2, emb_channels)
        self.up = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.out_res = ResBlock(base, emb_channels)
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, latent_channels, 3, 1, 1)
        self.action_emb = nn.Sequential(nn.Linear(3 + emb_channels, emb_channels), nn.SiLU(), nn.Linear(emb_channels, emb_channels))
        self.mod = nn.Linear(3, base * 2)

    def forward(self, x, content, t_embed, cont_action):
        content = F.interpolate(content, size=x.shape[-2:], mode="bilinear", align_corners=False)
        action_e = self.action_emb(torch.cat([cont_action, t_embed], dim=1))
        h = self.in_conv(torch.cat([x, content], dim=1))
        gamma, beta = self.mod(cont_action).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        skip = h
        h = self.down(h)
        h = self.mid(h, action_e)
        h = self.up(h) + skip
        h = self.out_res(h, action_e)
        return self.out_conv(F.silu(self.out_norm(h)))


class SARLNet(nn.Module):
    def __init__(self, scale=4, img_channels=3, base_channels=64, latent_channels=4, num_steps=8, restart_intervals=None):
        super().__init__()
        self.scale = scale
        self.num_steps = num_steps
        self.restart_intervals = restart_intervals or [0, 1, 2, 4]
        self.latent_channels = latent_channels
        self.encoder = nn.Conv2d(img_channels, latent_channels, 3, 1, 1)
        self.decoder = nn.Sequential(nn.Conv2d(latent_channels, base_channels, 3, 1, 1), nn.SiLU(), ResBlock(base_channels), nn.Conv2d(base_channels, img_channels, 3, 1, 1), nn.Sigmoid())
        self.sdpb = SDPB(latent_channels, base_channels)
        self.ce = ContentEncoder(img_channels, base_channels)
        self.policy = PolicyNet(latent_channels, base_channels, base_channels * 2, len(self.restart_intervals), 3)
        self.unet = ActionUNet(latent_channels, base_channels, base_channels, base_channels * 2)
        betas = torch.linspace(1e-4, 0.02, num_steps + max(self.restart_intervals) + 2)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def encode_lq(self, lq):
        lq_up = F.interpolate(lq, scale_factor=self.scale, mode="bicubic", align_corners=False).clamp(0, 1)
        return self.encoder(lq_up), lq_up

    def decode(self, z):
        return self.decoder(z).clamp(0, 1)

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        acp = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return acp.sqrt() * x0 + (1 - acp).sqrt() * noise, noise

    def _restart(self, x, step, restart_idx):
        intervals = torch.tensor(self.restart_intervals, device=x.device)
        add = intervals[restart_idx]
        t = torch.full_like(add, step)
        t_add = torch.minimum(t + add, torch.full_like(t, self.alphas_cumprod.numel() - 1))
        acp_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        acp_add = self.alphas_cumprod[t_add].view(-1, 1, 1, 1)
        rho = (acp_add / acp_t).clamp(0, 1)
        return rho.sqrt() * x + (1 - rho).sqrt() * torch.randn_like(x)

    def distributions(self, x, init, lq_up, step, pref, prev_reward):
        cond = self.sdpb(x, init)
        t_norm = torch.full((x.size(0),), step / max(self.num_steps - 1, 1), device=x.device)
        logits, mean, std, value = self.policy(x, cond, t_norm, pref, prev_reward)
        return Categorical(logits=logits), Normal(mean, std), value

    def denoise_from_restarted(self, x, lq_up, step, cont_action):
        content = self.ce(lq_up)
        t_embed = timestep_embedding(torch.full((x.size(0),), step, device=x.device), content.size(1) * 2)
        eps_pred = self.unet(x, content, t_embed, cont_action)
        beta = self.betas[min(step, self.betas.numel() - 1)]
        mean = x - beta.sqrt() * eps_pred
        sigma = beta.sqrt().clamp_min(1e-4)
        x_next = mean + sigma * torch.randn_like(mean)
        return x_next, mean, sigma

    def transition(self, x, init, lq_up, step, restart_idx, cont_action):
        x_restart = self._restart(x, step, restart_idx)
        x_next, mean, sigma = self.denoise_from_restarted(x_restart, lq_up, step, cont_action)
        return x_next, mean, sigma, x_restart

    def transition_log_prob(self, x_next, mean, sigma):
        dist = Normal(mean, sigma)
        return dist.log_prob(x_next).flatten(1).mean(dim=1)

    def rollout(self, lq, pref=None, explore=True):
        init, lq_up = self.encode_lq(lq)
        b = lq.size(0)
        device = lq.device
        pref = pref if pref is not None else torch.full((b, 3), 1 / 3, device=device)
        prev_reward = torch.zeros(b, device=device)
        x = init
        traj = []
        for step in reversed(range(1, self.num_steps + 1)):
            cat, norm, value = self.distributions(x, init, lq_up, step, pref, prev_reward)
            restart_idx = cat.sample() if explore else cat.probs.argmax(dim=1)
            cont_raw = norm.rsample() if explore else norm.mean
            cont_action = cont_raw.tanh()
            action_logp = cat.log_prob(restart_idx) + norm.log_prob(cont_raw).sum(dim=1)
            entropy = cat.entropy() + norm.entropy().sum(dim=1)
            x_prev, mean, sigma, x_restart = self.transition(x, init, lq_up, step, restart_idx, cont_action)
            trans_logp = self.transition_log_prob(x_prev, mean, sigma)
            traj.append(
                {
                    "x": x.detach(),
                    "x_next": x_prev.detach(),
                    "x_restart": x_restart.detach(),
                    "step": step,
                    "restart_idx": restart_idx.detach(),
                    "cont_raw": cont_raw.detach(),
                    "cont_action": cont_action.detach(),
                    "old_logp": (action_logp + trans_logp).detach(),
                    "old_action_logp": action_logp.detach(),
                    "value": value,
                    "entropy": entropy,
                }
            )
            x = x_prev
        return self.decode(x), traj, init.detach(), lq_up.detach()

    def recompute_logp_value(self, item, init, lq_up, pref, prev_reward):
        x = item["x"]
        step = item["step"]
        cat, norm, value = self.distributions(x, init, lq_up, step, pref, prev_reward)
        action_logp = cat.log_prob(item["restart_idx"]) + norm.log_prob(item["cont_raw"]).sum(dim=1)
        entropy = cat.entropy() + norm.entropy().sum(dim=1)
        _, mean, sigma = self.denoise_from_restarted(item["x_restart"], lq_up, step, item["cont_action"])
        trans_logp = self.transition_log_prob(item["x_next"], mean, sigma)
        return action_logp + trans_logp, value, entropy

    def forward(self, lq):
        sr, _, _, _ = self.rollout(lq, explore=False)
        return sr

