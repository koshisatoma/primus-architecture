import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, latent_dim, batch_first=True)

    def forward(self, values):
        encoded, _ = self.gru(values)
        return encoded


class ASTRA(nn.Module):
    """Source grounding, target encoder, and state/reward prediction heads."""

    def __init__(self, absolute_dim, action_dim, observation_dim, latent_dim=128):
        super().__init__()
        self.source_encoder = Encoder(absolute_dim + action_dim, latent_dim)
        self.transition_head = nn.Linear(latent_dim + action_dim, 2 * latent_dim)
        self.reward_head = nn.Linear(latent_dim + action_dim, 1)
        self.absolute_head = nn.Linear(
            latent_dim + action_dim + absolute_dim,
            absolute_dim,
        )
        self.target_encoder = Encoder(observation_dim, latent_dim)

    def source(self, absolute, actions):
        latent = self.source_encoder(torch.cat([absolute, actions], dim=-1))
        latent_action = torch.cat([latent, actions], dim=-1)
        mean, log_variance = self.transition_head(latent_action).chunk(2, dim=-1)
        predicted_reward = self.reward_head(latent_action)
        return latent, mean, log_variance, predicted_reward

    def corrected_absolute(self, latent, actions, simulated_next):
        values = torch.cat([latent, actions, simulated_next], dim=-1)
        return self.absolute_head(values)


def maximum_mean_discrepancy(source, target, sigma=1.0):
    def kernel(left, right):
        distances = torch.cdist(left, right).square()
        return torch.exp(-distances / (2.0 * sigma**2))

    return (
        kernel(source, source).mean()
        - 2.0 * kernel(source, target).mean()
        + kernel(target, target).mean()
    )


def load_episodes(path):
    episodes = []
    with np.load(path, allow_pickle=False) as archive:
        indices = sorted(
            {
                key.removeprefix('abs_')
                for key in archive.files
                if key.startswith('abs_')
            },
            key=int,
        )
        if not indices:
            raise ValueError(f'No episodes found in {path}')

        required = ('abs', 'act', 'rew', 'sim_next', 'tgt_next', 'obs')
        for index in indices:
            episode = {}
            for name in required:
                key = f'{name}_{index}'
                if key not in archive:
                    raise ValueError(f'Missing {key!r} in {path}')
                episode[name] = torch.as_tensor(
                    archive[key],
                    dtype=torch.float32,
                )
            lengths = {value.shape[0] for value in episode.values()}
            if len(lengths) != 1 or next(iter(lengths)) < 2:
                raise ValueError(
                    f'Episode {index} arrays must have matching lengths >= 2'
                )
            episodes.append(episode)
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/episodes.npz')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--align_epochs', type=int, default=20)
    parser.add_argument('--out', default='data/astra_grounding.pt')
    args = parser.parse_args()
    if args.epochs < 1 or args.align_epochs < 0:
        raise ValueError('--epochs must be positive and --align_epochs non-negative')

    episodes = load_episodes(args.data)
    model = ASTRA(
        episodes[0]['abs'].shape[-1],
        episodes[0]['act'].shape[-1],
        episodes[0]['obs'].shape[-1],
    )
    source_parameters = (
        list(model.source_encoder.parameters())
        + list(model.transition_head.parameters())
        + list(model.reward_head.parameters())
        + list(model.absolute_head.parameters())
    )
    optimizer = torch.optim.Adam(source_parameters, lr=1e-4)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for episode in episodes:
            absolute = episode['abs'].unsqueeze(0)
            actions = episode['act'].unsqueeze(0)
            latent, mean, log_variance, predicted_reward = model.source(
                absolute[:, :-1],
                actions[:, :-1],
            )
            next_latent = model.source(
                absolute[:, 1:],
                actions[:, 1:],
            )[0].detach()
            transition_loss = 0.5 * (
                log_variance
                + torch.square(next_latent - mean) * torch.exp(-log_variance)
            ).mean()
            reward_loss = F.mse_loss(
                predicted_reward[..., 0],
                episode['rew'][:-1].unsqueeze(0),
            )
            corrected_absolute = model.corrected_absolute(
                latent,
                actions[:, :-1],
                episode['sim_next'][:-1].unsqueeze(0),
            )
            absolute_loss = F.mse_loss(
                corrected_absolute,
                episode['tgt_next'][:-1].unsqueeze(0),
            )
            loss = transition_loss + reward_loss + 0.5 * absolute_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        if epoch % 10 == 0:
            print(
                f'ground epoch {epoch} loss {total_loss / len(episodes):.4f}',
                flush=True,
            )

    for parameter in source_parameters:
        parameter.requires_grad_(False)
    alignment_optimizer = torch.optim.Adam(
        model.target_encoder.parameters(),
        lr=1e-4,
    )
    for epoch in range(args.align_epochs):
        total_loss = 0.0
        for episode in episodes:
            source_latent = model.source_encoder(
                torch.cat([episode['abs'], episode['act']], dim=-1).unsqueeze(0)
            ).detach()
            target_latent = model.target_encoder(
                episode['obs'].unsqueeze(0)
            )
            loss = maximum_mean_discrepancy(
                source_latent.reshape(-1, source_latent.shape[-1]),
                target_latent.reshape(-1, target_latent.shape[-1]),
            )
            alignment_optimizer.zero_grad()
            loss.backward()
            alignment_optimizer.step()
            total_loss += float(loss.detach())

        if epoch % 5 == 0:
            print(
                f'align epoch {epoch} mmd {total_loss / len(episodes):.4f}',
                flush=True,
            )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'format_version': 1,
            'architecture': 'ASTRA',
            'absolute_dim': episodes[0]['abs'].shape[-1],
            'action_dim': episodes[0]['act'].shape[-1],
            'observation_dim': episodes[0]['obs'].shape[-1],
            'latent_dim': model.source_encoder.gru.hidden_size,
            'model_state_dict': model.state_dict(),
        },
        output,
    )
    print(f'saved {output}')


if __name__ == '__main__':
    main()
