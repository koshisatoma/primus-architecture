"""Refine an RL policy using behavior cloning and paired teacher corrections."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cloud.export_policy import rebuild
from cloud.train_reflex_ppo import adam_state, adam_step, mlp_fwd


def _read_corrections(path, observation_dim, action_dim):
    observations, good_actions, bad_actions = [], [], []
    with Path(path).open(encoding='utf-8') as correction_file:
        for line_number, line in enumerate(correction_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'Invalid correction JSON on line {line_number}'
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f'Correction line {line_number} must be a JSON object')
            if not all(key in row for key in ('obs', 'good', 'bad')):
                raise ValueError(f'Correction line {line_number} is missing required fields')
            try:
                observation = np.asarray(row['obs'], dtype=np.float32)
                good = np.asarray(row['good'], dtype=np.float32)
                bad = np.asarray(row['bad'], dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f'Correction line {line_number} contains invalid values'
                ) from error
            if observation.shape != (observation_dim,):
                raise ValueError(f'Correction line {line_number} has invalid observation shape')
            if good.shape != (action_dim,) or bad.shape != (action_dim,):
                raise ValueError(f'Correction line {line_number} has invalid action shape')
            if not (
                np.isfinite(observation).all()
                and np.isfinite(good).all()
                and np.isfinite(bad).all()
            ):
                raise ValueError(f'Correction line {line_number} contains non-finite values')
            observations.append(observation)
            good_actions.append(good)
            bad_actions.append(bad)
    if not observations:
        return None
    return tuple(
        jnp.asarray(values, dtype=jnp.float32)
        for values in (observations, good_actions, bad_actions)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--demos', required=True)
    parser.add_argument('--corrections', default='')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('--epochs must be positive')

    with np.load(args.ckpt, allow_pickle=False) as checkpoint:
        for key in ('ls', 'obs_dim', 'nu'):
            if key not in checkpoint:
                raise ValueError(f'Checkpoint is missing {key!r}')
        observation_dim = int(checkpoint['obs_dim'])
        action_dim = int(checkpoint['nu'])
        log_standard_deviation = np.asarray(checkpoint['ls'], dtype=np.float32)
        if log_standard_deviation.shape != (action_dim,):
            raise ValueError('Checkpoint action-noise vector has an invalid shape')
        pi = {
            key: jnp.asarray(value)
            for key, value in rebuild(checkpoint, 'pi').items()
        }
    expected_input = observation_dim
    for layer in range(len(pi) // 2):
        weights = np.asarray(pi[f'w{layer}'])
        biases = np.asarray(pi[f'b{layer}'])
        if (
            weights.ndim != 2
            or weights.shape[0] != expected_input
            or biases.shape != (weights.shape[1],)
        ):
            raise ValueError(f'Policy layer {layer} has incompatible dimensions')
        expected_input = weights.shape[1]
    if expected_input != action_dim:
        raise ValueError('Policy output layer does not match the checkpoint action size')

    with np.load(args.demos, allow_pickle=False) as demo_data:
        if 'obs' not in demo_data or 'act' not in demo_data:
            raise ValueError('Demonstration archive must contain obs and act arrays')
        observations = np.asarray(demo_data['obs'][:20000], dtype=np.float32)
        actions = np.asarray(demo_data['act'][:20000], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != observation_dim:
        raise ValueError('Demonstration observations do not match the checkpoint')
    if actions.shape != (len(observations), action_dim):
        raise ValueError('Demonstration actions do not match the checkpoint')
    if len(observations) == 0:
        raise ValueError('The demonstration archive contains no teacher frames')
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError('Demonstrations contain non-finite values')
    demo_observations = jnp.asarray(observations)
    demo_actions = jnp.asarray(actions)

    correction_data = (
        _read_corrections(args.corrections, observation_dim, action_dim)
        if args.corrections
        else None
    )

    def loss(policy):
        prediction = jax.vmap(mlp_fwd, in_axes=(None, 0))(
            policy, demo_observations
        )
        behavior_cloning = jnp.mean(jnp.square(prediction - demo_actions))
        if correction_data is None:
            return behavior_cloning

        correction_observations, good_actions, bad_actions = correction_data
        correction_prediction = jax.vmap(mlp_fwd, in_axes=(None, 0))(
            policy, correction_observations
        )
        good_distance = jnp.mean(
            jnp.square(correction_prediction - good_actions), axis=-1
        )
        bad_distance = jnp.mean(
            jnp.square(correction_prediction - bad_actions), axis=-1
        )
        ranking = jnp.mean(jax.nn.softplus(0.1 + good_distance - bad_distance))
        correction_loss = jnp.mean(jnp.square(correction_prediction - good_actions))
        return behavior_cloning + 0.5 * (correction_loss + 0.25 * ranking)

    train_step = jax.jit(jax.value_and_grad(loss))
    state = adam_state(pi)
    for epoch in range(args.epochs):
        loss_value, gradients = train_step(pi)
        pi, state = adam_step(
            pi,
            gradients,
            state,
            learning_rate=1e-3,
        )
        if epoch % 50 == 0:
            print(f'bc epoch {epoch} loss {float(loss_value):.4f}', flush=True)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        **{f'pi_{key}': np.asarray(value) for key, value in pi.items()},
        ls=log_standard_deviation,
        obs_dim=observation_dim,
        nu=action_dim,
    )
    print('saved', output_path)


if __name__ == '__main__':
    main()
