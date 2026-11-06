"""PPO-Clip teaching example: PyTorch only, CPU by default, no Gym dependency.

Run: python examples/ppo_core.py
Test: python -m unittest discover -s tests -p 'test_ppo_core.py' -v

The policy is an unbounded diagonal Gaussian. The synthetic rollout demonstrates
data flow and parameter updates, not performance on a benchmark. Real collectors
must supply the value of the actual successor observation BEFORE any reset.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math

import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal


def _layer(in_features: int, out_features: int, gain: float) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _trunk(obs_dim: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        _layer(obs_dim, hidden_size, math.sqrt(2)), nn.Tanh(),
        _layer(hidden_size, hidden_size, math.sqrt(2)), nn.Tanh(),
    )


class ActorCritic(nn.Module):
    """Two functions, optionally sharing features; inputs always have a batch axis."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64,
                 *, shared: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.backbone = _trunk(obs_dim, hidden_size) if shared else nn.Identity()
        self.actor = (
            _layer(hidden_size, action_dim, 0.01) if shared else
            nn.Sequential(_trunk(obs_dim, hidden_size), _layer(hidden_size, action_dim, 0.01))
        )
        self.critic = (
            _layer(hidden_size, 1, 1.0) if shared else
            nn.Sequential(_trunk(obs_dim, hidden_size), _layer(hidden_size, 1, 1.0))
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _features(self, obs: Tensor) -> Tensor:
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(f"obs must have shape [B, {self.obs_dim}], including B=1")
        return self.backbone(obs)

    def distribution(self, obs: Tensor) -> Independent:
        mean = self.actor(self._features(obs))
        # Independent sums log_prob and entropy across the action dimensions.
        return Independent(Normal(mean, self.log_std.exp().expand_as(mean)), 1)

    def value(self, obs: Tensor) -> Tensor:
        return self.critic(self._features(obs)).squeeze(-1)

    def evaluate(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if actions.shape != (obs.shape[0], self.action_dim):
            raise ValueError("actions must have shape [B, action_dim]")
        dist = self.distribution(obs)
        return dist.log_prob(actions), dist.entropy(), self.value(obs)

    @torch.no_grad()
    def sample(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dist = self.distribution(obs)
        actions = dist.sample()
        return actions, dist.log_prob(actions), self.value(obs)


@torch.no_grad()
def compute_gae(rewards: Tensor, values: Tensor, next_values: Tensor,
                terminated: Tensor, truncated: Tensor, *, gamma: float = 0.99,
                gae_lambda: float = 0.95) -> tuple[Tensor, Tensor]:
    """Return raw advantages and lambda-returns, both [T, N] and detached.

    All arrays describe the SAME transitions. rewards[t] is r_(t+1).
    values[t] is the rollout-time V(s_t). next_values[t] is V(s_(t+1))
    for the actual successor, including the final observation at truncation.
    Masks describe what happened AFTER action t. A rollout cutoff alone does
    not set either mask: last_gae starts at zero but next_values still bootstrap.
    """
    if rewards.ndim != 2 or min(rewards.shape) < 1:
        raise ValueError("GAE inputs must have nonempty shape [T, N]")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    for tensor in (rewards, values, next_values, terminated, truncated):
        if tensor.shape != rewards.shape or tensor.device != rewards.device:
            raise ValueError("GAE inputs must share shape [T, N] and device")
    for tensor in (rewards, values, next_values):
        if not tensor.is_floating_point() or tensor.dtype != rewards.dtype:
            raise ValueError("rewards and values must share a floating dtype")
        if not torch.isfinite(tensor).all():
            raise ValueError("rewards and values must be finite")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("terminated and truncated must be bool tensors")

    bootstrap_mask = (~terminated).to(rewards.dtype)
    trace_mask = (~(terminated | truncated)).to(rewards.dtype)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])  # [N], independent environment chains
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * bootstrap_mask[t] * next_values[t] - values[t]
        last_gae = delta + gamma * gae_lambda * trace_mask[t] * last_gae
        advantages[t] = last_gae
    returns = advantages + values  # Must precede advantage normalization.
    return advantages, returns


@dataclass(frozen=True)
class PPOBatch:
    obs: Tensor
    actions: Tensor
    old_logprobs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor

    def __post_init__(self):
        if self.obs.ndim != 2 or self.obs.shape[0] < 1:
            raise ValueError("obs must have nonempty shape [B, obs_dim]")
        size = self.obs.shape[0]
        if self.actions.ndim != 2 or self.actions.shape[0] != size:
            raise ValueError("actions must have shape [B, action_dim]")
        for field in fields(self):
            tensor = getattr(self, field.name)
            if field.name not in ("obs", "actions") and tensor.shape != (size,):
                raise ValueError(f"{field.name} must have shape [B]")
            if tensor.requires_grad:
                raise ValueError(f"{field.name} must be detached rollout data")
            if tensor.device != self.obs.device or tensor.dtype != self.obs.dtype:
                raise ValueError("batch fields must share floating dtype and device")
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("batch fields must be finite floating tensors")

    def select(self, indices: Tensor) -> "PPOBatch":
        return PPOBatch(*(getattr(self, f.name)[indices] for f in fields(self)))


def flatten_rollout(obs: Tensor, actions: Tensor, old_logprobs: Tensor,
                    old_values: Tensor, advantages: Tensor, returns: Tensor) -> PPOBatch:
    """Flatten [T, N, ...] AFTER GAE, with a detached copy of all fixed data."""
    if obs.ndim != 3 or actions.ndim != 3 or actions.shape[:2] != obs.shape[:2]:
        raise ValueError("obs/actions must have shape [T, N, feature_dim]")
    scalars = (old_logprobs, old_values, advantages, returns)
    if any(tensor.shape != obs.shape[:2] for tensor in scalars):
        raise ValueError("logprobs, values, advantages and returns must have shape [T, N]")
    tensors = (obs, actions, *scalars)
    return PPOBatch(*(tensor.flatten(0, 1).detach().clone() for tensor in tensors))


@dataclass
class LossTerms:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    ratio: Tensor
    approx_kl: Tensor
    clipfrac: Tensor


def ppo_loss(model: ActorCritic, batch: PPOBatch, *, policy_clip: float = 0.2,
             value_clip: float | None = None, value_coef: float = 0.5,
             entropy_coef: float = 0.0, normalize_advantages: bool = True) -> LossTerms:
    """Reevaluate the sampled actions; old values and targets remain constants.

    value_clip is optional and measured in reward units, independently of
    the dimensionless policy_clip. approx_kl uses mean(ratio - 1 - logratio).
    """
    if not 0 < policy_clip < 1:
        raise ValueError("policy_clip must be in (0, 1)")
    if value_clip is not None and (not math.isfinite(value_clip) or value_clip <= 0):
        raise ValueError("value_clip must be a positive finite number or None")
    if not math.isfinite(value_coef) or value_coef < 0 or not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError("loss coefficients must be finite and nonnegative")
    new_logprobs, entropy, new_values = model.evaluate(batch.obs, batch.actions)
    logratio = new_logprobs - batch.old_logprobs
    ratio = logratio.exp()
    advantages = batch.advantages
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(correction=0) + 1e-8)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - policy_clip, 1 + policy_clip) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    value_error = (new_values - batch.returns).square()
    if value_clip is not None:
        clipped_values = batch.old_values + (new_values - batch.old_values).clamp(-value_clip, value_clip)
        value_error = torch.maximum(value_error, (clipped_values - batch.returns).square())
    value_loss = 0.5 * value_error.mean()
    entropy_mean = entropy.mean()
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    with torch.no_grad():
        approx_kl = (ratio - 1 - logratio).mean()
        clipfrac = ((ratio - 1).abs() > policy_clip).float().mean()
    return LossTerms(total, policy_loss, value_loss, entropy_mean, ratio, approx_kl, clipfrac)


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, batch: PPOBatch,
               *, epochs: int = 4, minibatch_size: int = 16,
               policy_clip: float = 0.2, value_clip: float | None = None,
               value_coef: float = 0.5, entropy_coef: float = 0.0,
               max_grad_norm: float = 0.5) -> dict[str, float | int]:
    """K epochs of shuffled minibatches; report means measured before each step."""
    size = batch.obs.shape[0]
    if epochs < 1 or minibatch_size < 1 or size % minibatch_size:
        raise ValueError("epochs/minibatch_size must be positive; B must divide into full minibatches")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and finite")
    metrics = {key: 0.0 for key in (
        "loss", "policy_loss", "value_loss", "entropy", "approx_kl",
        "clipfrac", "grad_norm_before_clip",
    )}
    updates = 0
    for _ in range(epochs):
        indices = torch.randperm(size, device=batch.obs.device)
        for start in range(0, size, minibatch_size):
            minibatch = batch.select(indices[start:start + minibatch_size])
            terms = ppo_loss(model, minibatch, policy_clip=policy_clip,
                             value_clip=value_clip, value_coef=value_coef,
                             entropy_coef=entropy_coef)
            if not torch.isfinite(terms.total):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, error_if_nonfinite=True,
            )
            optimizer.step()
            observations = (
                terms.total, terms.policy, terms.value, terms.entropy,
                terms.approx_kl, terms.clipfrac, grad_norm,
            )
            for key, value in zip(metrics, observations):
                metrics[key] += value.detach().item()
            updates += 1
    return {**{key: value / updates for key, value in metrics.items()}, "optimizer_steps": updates}


@torch.no_grad()
def synthetic_rollout(model: ActorCritic, *, steps: int = 16, num_envs: int = 4) -> PPOBatch:
    """A small continuing toy process with explicit termination/truncation/reset.

    Collect with frozen parameters, preserve final successor values, calculate
    GAE, then flatten. This provides a reproducible smoke test without Gym.
    """
    if steps < 1 or num_envs < 1 or model.action_dim > model.obs_dim:
        raise ValueError("need T,N >= 1 and action_dim <= obs_dim")
    reference = next(model.parameters())
    obs = torch.randn(num_envs, model.obs_dim, device=reference.device, dtype=reference.dtype)
    ages = torch.zeros(num_envs, device=obs.device, dtype=torch.long)
    env_ids = torch.arange(num_envs, device=obs.device)
    storage: dict[str, list[Tensor]] = {key: [] for key in (
        "obs", "actions", "old_logprobs", "old_values", "rewards",
        "next_values", "terminated", "truncated",
    )}
    for _ in range(steps):
        actions, old_logprobs, old_values = model.sample(obs)
        rewards = -(actions - obs[:, :model.action_dim]).square().sum(-1)
        successor = 0.6 * obs + 0.1 * torch.randn_like(obs)
        successor[:, :model.action_dim] += 0.1 * actions
        ages += 1
        terminated = (env_ids % 2 == 0) & (ages >= 5)
        truncated = (env_ids % 2 == 1) & (ages >= 7)
        # Evaluate BEFORE resetting; a truncation must keep its final-state value.
        next_values = model.value(successor)
        for key, value in (
            ("obs", obs), ("actions", actions), ("old_logprobs", old_logprobs),
            ("old_values", old_values), ("rewards", rewards),
            ("next_values", next_values), ("terminated", terminated), ("truncated", truncated),
        ):
            storage[key].append(value.clone())
        done = terminated | truncated
        obs = successor.clone()
        obs[done] = torch.randn_like(obs[done])
        ages[done] = 0
    data = {key: torch.stack(value) for key, value in storage.items()}
    advantages, returns = compute_gae(
        data["rewards"], data["old_values"], data["next_values"],
        data["terminated"], data["truncated"],
    )
    return flatten_rollout(data["obs"], data["actions"], data["old_logprobs"],
                           data["old_values"], advantages, returns)


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = ActorCritic(obs_dim=4, action_dim=2)  # CPU; shared=True also supported.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    batch = synthetic_rollout(model, steps=16, num_envs=4)
    with torch.no_grad():
        initial_ratio_error = (ppo_loss(model, batch).ratio - 1).abs().max().item()
    metrics = ppo_update(model, optimizer, batch, epochs=4, minibatch_size=16)
    print(json.dumps({"batch_size": batch.obs.shape[0],
                      "initial_ratio_max_error": initial_ratio_error, **metrics}, indent=2))


if __name__ == "__main__":
    main()
