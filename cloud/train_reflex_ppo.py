import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cloud.g1_env import G1Env


def mlp_init(key, dims):
    params = {}
    for layer in range(len(dims) - 1):
        key, layer_key = jax.random.split(key)
        params[f'w{layer}'] = (
            jax.random.normal(layer_key, (dims[layer], dims[layer + 1]))
            * jnp.sqrt(2.0 / dims[layer])
        )
        params[f'b{layer}'] = jnp.zeros(dims[layer + 1])
    return params


def mlp_fwd(params, values):
    layers = sum(name.startswith('w') for name in params)
    for layer in range(layers):
        values = values @ params[f'w{layer}'] + params[f'b{layer}']
        if layer < layers - 1:
            values = jnp.tanh(values)
    return values


def adam_state(params):
    zeros_like = lambda value: jnp.zeros_like(value)
    return {
        'm': jax.tree_util.tree_map(zeros_like, params),
        'v': jax.tree_util.tree_map(zeros_like, params),
        't': 0,
    }


def adam_step(params, gradients, state, learning_rate=3e-4):
    step = state['t'] + 1
    moments = jax.tree_util.tree_map(
        lambda moment, gradient: 0.9 * moment + 0.1 * gradient,
        state['m'],
        gradients,
    )
    variances = jax.tree_util.tree_map(
        lambda variance, gradient: 0.999 * variance + 0.001 * gradient**2,
        state['v'],
        gradients,
    )
    corrected_moments = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - 0.9**step),
        moments,
    )
    corrected_variances = jax.tree_util.tree_map(
        lambda variance: variance / (1.0 - 0.999**step),
        variances,
    )
    params = jax.tree_util.tree_map(
        lambda value, moment, variance: value
        - learning_rate * moment / (jnp.sqrt(variance) + 1e-8),
        params,
        corrected_moments,
        corrected_variances,
    )
    return params, {'m': moments, 'v': variances, 't': step}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('stand', 'walk', 'reach'), default='stand')
    parser.add_argument('--envs', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--T', type=int, default=256)
    parser.add_argument('--out', default='data/ckpt')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.envs < 1 or args.steps < 1 or args.T < 1:
        raise ValueError('--envs, --steps, and --T must all be positive')

    env = G1Env(args.task)
    env_count, horizon = args.envs, args.T
    reset_batch = jax.vmap(env.reset)
    step_batch = jax.vmap(env.step)
    observe_batch = jax.vmap(env.obs)
    reward_batch = jax.vmap(env.reward)
    fallen_batch = jax.vmap(env.fallen)

    key = jax.random.PRNGKey(0)
    key, actor_key, critic_key, reset_key = jax.random.split(key, 4)
    params = {
        'pi': mlp_init(actor_key, [env.obs_dim, 64, 64, env.nu]),
        'vf': mlp_init(critic_key, [env.obs_dim, 64, 64, 1]),
        'ls': jnp.full((env.nu,), -1.0),
    }
    optimizer_state = adam_state(params)
    data = reset_batch(jax.random.split(reset_key, env_count))

    def batch_forward(network, observations):
        return jax.vmap(mlp_fwd, in_axes=(None, 0))(network, observations)

    def rollout(current_params, rollout_key, current_data):
        def step(carry, _):
            current_data, step_key = carry
            observations = observe_batch(current_data)
            means = batch_forward(current_params['pi'], observations)
            values = batch_forward(current_params['vf'], observations)[..., 0]
            step_key, action_key, reset_key = jax.random.split(step_key, 3)
            standard_deviation = jnp.exp(current_params['ls'])
            actions = means + standard_deviation * jax.random.normal(
                action_key,
                means.shape,
            )
            log_prob = -0.5 * jnp.sum(
                jnp.square((actions - means) / standard_deviation),
                axis=-1,
            ) - jnp.sum(current_params['ls'])
            next_data = step_batch(current_data, actions)
            rewards = reward_batch(next_data) - 2.0 * fallen_batch(next_data)
            dones = fallen_batch(next_data)
            reset_data = reset_batch(jax.random.split(reset_key, env_count))

            def reset_finished_envs(reset_value, next_value):
                mask = dones.reshape(dones.shape + (1,) * (next_value.ndim - 1))
                return jnp.where(mask, reset_value, next_value)

            next_data = jax.tree_util.tree_map(
                reset_finished_envs,
                reset_data,
                next_data,
            )
            return (next_data, step_key), (
                observations,
                actions,
                rewards,
                dones,
                log_prob,
                values,
            )

        (final_data, final_key), transitions = jax.lax.scan(
            step,
            (current_data, rollout_key),
            None,
            length=horizon,
        )
        final_values = batch_forward(
            current_params['vf'],
            observe_batch(final_data),
        )[..., 0]
        return transitions, final_values, final_data, final_key

    def generalized_advantage(transitions, final_values, gamma=0.97, lam=0.95):
        _, _, rewards, dones, _, values = transitions
        next_values = jnp.concatenate([values[1:], final_values[None, :]], axis=0)

        def scan_step(advantage, transition):
            reward, value, next_value, done = transition
            alive = 1.0 - done.astype(jnp.float32)
            delta = reward + gamma * next_value * alive - value
            advantage = delta + gamma * lam * alive * advantage
            return advantage, advantage

        _, advantages = jax.lax.scan(
            scan_step,
            jnp.zeros_like(final_values),
            (rewards, values, next_values, dones),
            reverse=True,
        )
        return advantages, advantages + values

    def loss(current_params, transitions, advantages, returns):
        observations, actions, _, _, old_log_prob, _ = transitions
        flat_observations = observations.reshape((horizon * env_count, -1))
        means = batch_forward(current_params['pi'], flat_observations)
        means = means.reshape((horizon, env_count, env.nu))
        standard_deviation = jnp.exp(current_params['ls'])
        log_prob = -0.5 * jnp.sum(
            jnp.square((actions - means) / standard_deviation),
            axis=-1,
        ) - jnp.sum(current_params['ls'])
        ratio = jnp.exp(log_prob - old_log_prob)
        normalized_advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)
        clipped_ratio = jnp.clip(ratio, 0.8, 1.2)
        policy_loss = -jnp.mean(
            jnp.minimum(
                ratio * normalized_advantages,
                clipped_ratio * normalized_advantages,
            )
        )

        values = batch_forward(
            current_params['vf'],
            flat_observations,
        )[..., 0].reshape((horizon, env_count))
        value_loss = jnp.mean(jnp.square(values - returns))
        entropy_bonus = 0.01 * jnp.sum(current_params['ls'])
        return policy_loss + 0.5 * value_loss - entropy_bonus

    update = jax.jit(jax.value_and_grad(loss))
    rollout = jax.jit(rollout)
    output_prefix = Path(args.out)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    steps_done = 0

    while steps_done < args.steps:
        key, rollout_key = jax.random.split(key)
        transitions, final_values, data, key = rollout(
            params,
            rollout_key,
            data,
        )
        advantages, returns = generalized_advantage(transitions, final_values)
        loss_value, gradients = update(params, transitions, advantages, returns)
        params, optimizer_state = adam_step(params, gradients, optimizer_state)
        steps_done += env_count * horizon

        print(
            f'[{args.task}] {steps_done} '
            f'r={float(transitions[2].mean()):.2f} '
            f'loss={float(loss_value):.3f} '
            f'{time.time() - start_time:.0f}s',
            flush=True,
        )
        checkpoint_path = Path(f'{output_prefix}_{args.task}.npz')
        np.savez(
            checkpoint_path,
            **{
                f'pi_{name}': np.asarray(value)
                for name, value in params['pi'].items()
            },
            **{
                f'vf_{name}': np.asarray(value)
                for name, value in params['vf'].items()
            },
            ls=np.asarray(params['ls']),
            obs_dim=env.obs_dim,
            nu=env.nu,
        )

    print(f'saved {output_prefix}_{args.task}.npz')


if __name__ == '__main__':
    main()
