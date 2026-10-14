import argparse
from pathlib import Path

import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from cloud.g1_env import G1Env


def target_features(data, torso_id):
    rotation = data.xmat[torso_id].reshape(3, 3)
    return np.asarray(
        [
            data.xpos[torso_id, 2],
            rotation[2, 2],
            *data.subtree_linvel[1].tolist(),
        ],
        dtype=np.float32,
    )


def output_path(value):
    path = Path(value)
    return path if path.suffix == '.npz' else Path(f'{path}.npz')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=int, default=200)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--out', default='data/episodes')
    args = parser.parse_args()
    if args.eps < 1 or args.T < 1:
        raise ValueError('--eps and --T must both be positive')

    env = G1Env('stand')
    model, mjx_model = env.m, env.mx
    rng = np.random.default_rng(0)
    episodes = []

    for episode_index in range(args.eps):
        cpu_data = mujoco.MjData(model)
        cpu_data.qpos[:] = np.asarray(env.qpos0)
        cpu_data.qpos[7:] += 0.01 * rng.normal(size=env.nu)
        mujoco.mj_forward(model, cpu_data)

        sim_data = mjx.make_data(mjx_model)
        sim_data = mjx.forward(
            mjx_model,
            sim_data.replace(
                qpos=jnp.asarray(cpu_data.qpos),
                qvel=jnp.asarray(cpu_data.qvel),
            ),
        )
        record = {
            name: []
            for name in ('abs', 'act', 'rew', 'sim_next', 'tgt_next', 'obs')
        }

        for _ in range(args.T):
            action = rng.normal(size=model.nu).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            current_features = target_features(cpu_data, env.torso)
            current_observation = np.concatenate(
                [cpu_data.qpos, cpu_data.qvel]
            ).astype(np.float32)
            control = np.asarray(env.qdef) + 0.5 * action
            cpu_data.ctrl[:] = control
            mujoco.mj_step(model, cpu_data)
            sim_data = mjx.step(mjx_model, sim_data.replace(ctrl=jnp.asarray(control)))

            record['abs'].append(current_features)
            record['act'].append(action)
            record['rew'].append(float(env.reward(sim_data)))
            record['sim_next'].append(np.asarray(env.phi(sim_data)))
            record['tgt_next'].append(
                target_features(cpu_data, env.torso)
            )
            record['obs'].append(current_observation)

        episodes.append(
            {
                name: np.asarray(values, dtype=np.float32)
                for name, values in record.items()
            }
        )
        if episode_index % 20 == 0:
            print(f'episode {episode_index + 1}/{args.eps}', flush=True)

    destination = output_path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **{
            f'{name}_{episode_index}': episode[name]
            for episode_index, episode in enumerate(episodes)
            for name in episode
        },
    )
    print(f'saved {destination}')


if __name__ == '__main__':
    main()
