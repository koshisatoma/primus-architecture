"""Optional PyTorch utilities for EWC penalties and per-skill adapters."""

import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SkillAdapter(nn.Module):
    """A small per-skill low-rank residual adapter."""

    def __init__(self, base_dim: int, rank: int = 8):
        super().__init__()
        if base_dim < 1 or rank < 1:
            raise ValueError('base_dim and rank must be positive')
        self.lora_A = nn.Parameter(torch.randn(base_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, base_dim))
        self.scaling = 1.0 / rank

    def forward(self, values):
        return values @ (self.lora_A @ self.lora_B) * self.scaling


class AntiForgetting:
    """Store Fisher estimates and per-task reference weights for EWC."""

    def __init__(self, checkpoint_dir='data/ewc_checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.fisher_info = {}
        self.optimal_params = {}
        self.skill_adapters = nn.ModuleDict()

    @staticmethod
    def _safe_task_name(task_name):
        if not isinstance(task_name, str) or not re.fullmatch(
            r'[A-Za-z0-9_-]{1,64}', task_name
        ):
            raise ValueError('task_name may contain only letters, digits, _ and -')
        return task_name

    def compute_fisher(self, model, dataloader, task_name):
        task_name = self._safe_task_name(task_name)
        fisher = {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        sample_count = 0
        was_training = model.training
        model.eval()
        try:
            for batch in dataloader:
                model.zero_grad(set_to_none=True)
                output = model(batch['input'])
                loss = F.cross_entropy(output, batch['target'])
                loss.backward()
                batch_size = int(batch['target'].shape[0])
                sample_count += batch_size
                for name, parameter in model.named_parameters():
                    if name in fisher and parameter.grad is not None:
                        fisher[name] += parameter.grad.detach().square() * batch_size
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was_training)
        if sample_count == 0:
            raise ValueError('Cannot compute Fisher information from an empty dataset')
        fisher = {name: value / sample_count for name, value in fisher.items()}
        self.fisher_info[task_name] = fisher
        self.save_optimal_params(model, task_name)
        torch.save(fisher, self.checkpoint_dir / f'{task_name}_fisher.pt')
        return fisher

    def save_optimal_params(self, model, task_name):
        task_name = self._safe_task_name(task_name)
        parameters = {
            name: value.detach().cpu().clone()
            for name, value in model.named_parameters()
        }
        self.optimal_params[task_name] = parameters
        torch.save(parameters, self.checkpoint_dir / f'{task_name}_optimal.pt')

    def ewc_penalty(self, model, task_name, lambda_ewc: float = 1000.0):
        task_name = self._safe_task_name(task_name)
        if lambda_ewc < 0:
            raise ValueError('lambda_ewc must be non-negative')
        fisher = self.fisher_info.get(task_name)
        if fisher is None:
            fisher_path = self.checkpoint_dir / f'{task_name}_fisher.pt'
            if not fisher_path.exists():
                return next(model.parameters()).new_zeros(())
            fisher = torch.load(
                fisher_path,
                map_location='cpu',
                weights_only=True,
            )
            self.fisher_info[task_name] = fisher
        optimum = self.optimal_params.get(task_name)
        if optimum is None:
            optimum_path = self.checkpoint_dir / f'{task_name}_optimal.pt'
            if not optimum_path.exists():
                raise FileNotFoundError(
                    f'Missing optimal parameter snapshot for task {task_name}'
                )
            optimum = torch.load(
                optimum_path,
                map_location='cpu',
                weights_only=True,
            )
            self.optimal_params[task_name] = optimum
        penalty = next(model.parameters()).new_zeros(())
        for name, parameter in model.named_parameters():
            if name in fisher and name in optimum:
                penalty = penalty + (
                    fisher[name].to(parameter.device)
                    * (parameter - optimum[name].to(parameter.device)).square()
                ).sum()
        return float(lambda_ewc) * penalty

    def create_skill_adapter(self, skill_name: str, base_dim: int, rank: int = 8):
        skill_name = self._safe_task_name(skill_name)
        if skill_name not in self.skill_adapters:
            self.skill_adapters[skill_name] = SkillAdapter(base_dim, rank)
        return self.skill_adapters[skill_name]

    def save_skill_adapter(self, skill_name: str):
        skill_name = self._safe_task_name(skill_name)
        if skill_name not in self.skill_adapters:
            raise KeyError(f'No adapter registered for skill {skill_name}')
        path = self.checkpoint_dir / f'{skill_name}_lora.pt'
        torch.save(self.skill_adapters[skill_name].state_dict(), path)
        return path

    def load_skill_adapter(
        self, skill_name: str, base_dim: int, rank: int = 8
    ) -> SkillAdapter | None:
        skill_name = self._safe_task_name(skill_name)
        path = self.checkpoint_dir / f'{skill_name}_lora.pt'
        if not path.exists():
            return None
        adapter = SkillAdapter(base_dim, rank)
        state = torch.load(path, map_location='cpu', weights_only=True)
        adapter.load_state_dict(state)
        self.skill_adapters[skill_name] = adapter
        return adapter


_instance = None


def anti_forgetting() -> AntiForgetting:
    global _instance
    if _instance is None:
        _instance = AntiForgetting()
    return _instance
