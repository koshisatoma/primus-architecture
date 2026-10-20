import time

import torch


class WMPlanner:
    """Latent-space CEM using an ASTRA-style world model at about 10 Hz."""

    def __init__(self, astra_ckpt, z_dim=128, act_dim=29, H=8, K=64):
        self.H, self.K = H, K
        self.z_dim, self.act_dim = z_dim, act_dim
        self.model_path = astra_ckpt
        self.net = torch.load(astra_ckpt, map_location='cpu', weights_only=False)
        self.net.eval()

    @torch.no_grad()
    def plan(self, z0, base_action):
        """Return the best first action offset and wall-clock planning time in ms."""
        t0 = time.time()
        if isinstance(base_action, (list, tuple)):
            base_action = torch.as_tensor(base_action, dtype=torch.float32)
        else:
            base_action = base_action.to(torch.float32)

        mean = torch.zeros(self.H, self.act_dim, dtype=base_action.dtype)
        best, best_r = None, -1e9
        for _ in range(3):
            noise = torch.randn(self.K, self.H, self.act_dim, dtype=base_action.dtype) * 0.3
            cand = mean[None] + noise
            rewards = torch.zeros(self.K, dtype=base_action.dtype)
            z = z0[None].repeat(self.K, 1, 1) if z0.dim() == 2 else z0[None].repeat(self.K, 1)
            for h in range(self.H):
                action = base_action[None, None] + cand[:, h:h + 1]
                za = torch.cat([z[:, -1:], action], dim=-1)
                plat_out = self.net.plat(za)
                if isinstance(plat_out, tuple):
                    mu, lv = plat_out
                else:
                    mu, lv = plat_out.chunk(2, dim=-1)
                r = self.net.rpred(za)
                rewards += r.squeeze(-1).squeeze(-1)
                z = torch.cat([z, mu], dim=1)
            j = rewards.argmax()
            if rewards[j] > best_r:
                best_r, best = float(rewards[j]), cand[j]
            topk = rewards.topk(8).indices
            mean = 0.3 * mean + 0.7 * cand[topk].mean(dim=0)
        dt = (time.time() - t0) * 1000.0
        return best[0], dt


if __name__ == '__main__':
    print('import OK — plug astra_grounding.pt at launch')
