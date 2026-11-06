"""Mathematical and gradient-boundary checks for the published PPO example."""

from dataclasses import fields, replace
import math
import unittest

import torch

from examples.ppo_core import (
    ActorCritic, compute_gae, flatten_rollout,
    ppo_loss, ppo_update, synthetic_rollout,
)


class GaeTests(unittest.TestCase):
    def setUp(self):
        self.rewards = torch.tensor([[1.0], [0.0], [2.0], [1.0]], dtype=torch.float64)
        self.values = torch.tensor([[2.0], [2.5], [2.0], [1.5]], dtype=torch.float64)
        self.next_values = torch.tensor([[2.5], [2.0], [1.5], [1.0]], dtype=torch.float64)
        self.flags = torch.zeros(4, 1, dtype=torch.bool)

    def test_published_hand_calculation(self):
        advantages, returns = compute_gae(
            self.rewards, self.values, self.next_values, self.flags, self.flags,
            gamma=0.9, gae_lambda=0.8,
        )
        torch.testing.assert_close(advantages[:, 0], torch.tensor(
            [1.5951392, 0.47936, 1.638, 0.4], dtype=torch.float64,
        ))
        torch.testing.assert_close(returns[:, 0], torch.tensor(
            [3.5951392, 2.97936, 3.638, 1.9], dtype=torch.float64,
        ))

    def test_lambda_zero_and_one(self):
        _, one_step = compute_gae(
            self.rewards, self.values, self.next_values, self.flags, self.flags,
            gamma=0.9, gae_lambda=0.0,
        )
        torch.testing.assert_close(one_step, self.rewards + 0.9 * self.next_values)
        _, full_return = compute_gae(
            self.rewards, self.values, self.next_values, self.flags, self.flags,
            gamma=0.9, gae_lambda=1.0,
        )
        # At a rollout cutoff, lambda=1 STILL includes the final bootstrap.
        self.assertAlmostEqual(full_return[0, 0].item(), 1 + 0.9**2 * 2 + 0.9**3 + 0.9**4)

    def test_finite_lambda_return_has_last_weight_remainder(self):
        _, returns = compute_gae(
            self.rewards, self.values, self.next_values, self.flags, self.flags,
            gamma=0.9, gae_lambda=0.8,
        )
        for t in range(4):
            horizon = 4 - t
            expected = 0.0
            for n in range(1, horizon + 1):
                n_return = sum(0.9**k * self.rewards[t + k, 0].item() for k in range(n))
                n_return += 0.9**n * self.next_values[t + n - 1, 0].item()
                weight = 0.8**(n - 1) * (0.2 if n < horizon else 1.0)
                expected += weight * n_return
            self.assertAlmostEqual(returns[t, 0].item(), expected)

    def test_terminal_truncation_and_continuation_are_distinct(self):
        # t=1 belongs to a reset episode for env 0/1. Its huge reward must not
        # leak backward. The continuing third environment must retain it.
        rewards = torch.tensor([[1.0, 1.0, 1.0], [1000.0, 1000.0, 1000.0]])
        values = torch.zeros_like(rewards)
        successors = torch.tensor([[10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
        terminated = torch.tensor([[True, False, False], [False, False, False]])
        truncated = torch.tensor([[False, True, False], [False, False, False]])
        advantages, _ = compute_gae(
            rewards, values, successors, terminated, truncated, gamma=0.9, gae_lambda=0.8,
        )
        torch.testing.assert_close(advantages[0], torch.tensor([1.0, 10.0, 730.0]))

    def test_targets_stop_gradient_and_shapes_are_checked(self):
        advantages, returns = compute_gae(
            self.rewards.requires_grad_(), self.values.requires_grad_(),
            self.next_values.requires_grad_(), self.flags, self.flags,
        )
        self.assertFalse(advantages.requires_grad)
        self.assertFalse(returns.requires_grad)
        with self.assertRaises(ValueError):
            compute_gae(self.rewards.flatten(), self.values, self.next_values, self.flags, self.flags)
        with self.assertRaises(ValueError):
            compute_gae(self.rewards, self.values, self.next_values, self.flags.float(), self.flags)


class UpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(11)
        self.model = ActorCritic(4, 2, hidden_size=16)
        self.batch = synthetic_rollout(self.model, steps=8, num_envs=2)

    def test_batch_shape_single_environment_and_initial_ratio(self):
        actions, logprobs, values = self.model.sample(torch.zeros(1, 4))
        self.assertEqual(actions.shape, (1, 2))
        self.assertEqual(logprobs.shape, (1,))
        self.assertEqual(values.shape, (1,))
        torch.testing.assert_close(ppo_loss(self.model, self.batch).ratio, torch.ones(16))
        with self.assertRaises(ValueError):
            self.model.sample(torch.zeros(4))

    def test_gaussian_logprob_is_joint_over_action_dimensions(self):
        obs = self.batch.obs
        dist = self.model.distribution(obs)
        logprobs, _, _ = self.model.evaluate(obs, self.batch.actions)
        expected = dist.base_dist.log_prob(self.batch.actions).sum(-1)
        torch.testing.assert_close(logprobs, expected)

    def test_flatten_preserves_transition_alignment_and_copies_fixed_data(self):
        obs = torch.arange(24.0).reshape(3, 2, 4).requires_grad_()
        actions = torch.arange(12.0).reshape(3, 2, 2)
        scalars = torch.arange(6.0).reshape(3, 2)
        batch = flatten_rollout(obs, actions, scalars, scalars, scalars, scalars)
        torch.testing.assert_close(batch.obs[3], obs.detach()[1, 1])
        self.assertEqual(batch.old_values[3].item(), scalars[1, 1].item())
        scalars.fill_(999)
        self.assertEqual(batch.old_values[3].item(), 3.0)
        self.assertFalse(batch.obs.requires_grad)

    def test_policy_gradient_does_not_reach_independent_critic(self):
        terms = ppo_loss(self.model, self.batch, normalize_advantages=False)
        terms.policy.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.critic.parameters()))
        self.model.zero_grad(set_to_none=True)
        ppo_loss(self.model, self.batch).value.backward()
        self.assertTrue(all(p.grad is None for p in self.model.actor.parameters()))
        self.assertIsNone(self.model.log_std.grad)
        self.assertTrue(any(p.grad is not None for p in self.model.critic.parameters()))

    def test_shared_backbone_receives_both_gradients(self):
        model = ActorCritic(4, 2, hidden_size=16, shared=True)
        batch = synthetic_rollout(model, steps=4, num_envs=2)
        for term in ("policy", "value"):
            model.zero_grad(set_to_none=True)
            getattr(ppo_loss(model, batch, normalize_advantages=False), term).backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters()))

    def test_update_changes_both_networks_and_preserves_rollout(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        actor_before = torch.cat([p.detach().flatten().clone() for p in self.model.actor.parameters()])
        critic_before = torch.cat([p.detach().flatten().clone() for p in self.model.critic.parameters()])
        fixed_before = {f.name: getattr(self.batch, f.name).clone() for f in fields(self.batch)}
        metrics = ppo_update(self.model, optimizer, self.batch, epochs=2, minibatch_size=4)
        self.assertEqual(metrics["optimizer_steps"], 8)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        actor_after = torch.cat([p.detach().flatten() for p in self.model.actor.parameters()])
        critic_after = torch.cat([p.detach().flatten() for p in self.model.critic.parameters()])
        self.assertFalse(torch.equal(actor_before, actor_after))
        self.assertFalse(torch.equal(critic_before, critic_after))
        for name, original in fixed_before.items():
            torch.testing.assert_close(getattr(self.batch, name), original, rtol=0, atol=0)

    def test_clipping_handles_both_advantage_signs_and_bad_directions(self):
        indices = torch.arange(4)
        batch = self.batch.select(indices)
        with torch.no_grad():
            new_logprobs, _, _ = self.model.evaluate(batch.obs, batch.actions)
        desired_ratios = torch.tensor([1.3, 0.7, 0.7, 1.3])
        batch = replace(batch, old_logprobs=new_logprobs - desired_ratios.log(),
                        advantages=torch.tensor([2.0, -2.0, 2.0, -2.0]))
        terms = ppo_loss(self.model, batch, normalize_advantages=False)
        # Favorable excess is capped; unfavorable motion is still penalized.
        expected_objectives = torch.tensor([2.4, -1.6, 1.4, -2.6])
        torch.testing.assert_close(terms.policy, -expected_objectives.mean())
        self.assertEqual(terms.clipfrac.item(), 1.0)  # Only two are on flat branches.

    def test_value_clipping_and_normalization_do_not_change_returns(self):
        with torch.no_grad():
            current_values = self.model.value(self.batch.obs)
        batch = replace(self.batch, old_values=current_values - 1, returns=current_values + 1)
        terms = ppo_loss(self.model, batch, value_clip=0.2)
        self.assertAlmostEqual(terms.value.item(), 0.5 * 1.8**2, places=5)
        fixed_returns = batch.returns.clone()
        # A one-element normalized minibatch is finite (zero advantage), not NaN.
        singleton = batch.select(torch.tensor([0]))
        self.assertTrue(torch.isfinite(ppo_loss(self.model, singleton).total))
        torch.testing.assert_close(batch.returns, fixed_returns, rtol=0, atol=0)

    def test_invalid_batch_and_minibatch_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.batch, returns=self.batch.returns.unsqueeze(-1))
        with self.assertRaises(ValueError):
            replace(self.batch, advantages=self.batch.advantages.clone().requires_grad_())
        with self.assertRaises(ValueError):
            ppo_update(self.model, torch.optim.Adam(self.model.parameters()), self.batch, minibatch_size=3)


if __name__ == "__main__":
    unittest.main()
