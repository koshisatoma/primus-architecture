"""Collect human demonstrations and policy corrections in a MuJoCo viewer."""

import argparse
import json
from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

from cloud.export_policy import forward, rebuild


GROUPS = {'1': (0, 12), '2': (12, 15), '3': (15, 22), '4': (22, 29)}
KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT, KEY_SPACE = 265, 264, 262, 263, 32


def _load_policy(path, observation_dim, action_dim):
    if not path:
        return None
    with np.load(path, allow_pickle=False) as checkpoint:
        if 'obs_dim' in checkpoint and int(checkpoint['obs_dim']) != observation_dim:
            raise ValueError('Policy observation size does not match the MuJoCo model')
        if 'nu' in checkpoint and int(checkpoint['nu']) != action_dim:
            raise ValueError('Policy action size does not match the MuJoCo model')
        if any(key.startswith('pi_w') for key in checkpoint.files):
            policy = rebuild(checkpoint, 'pi')
        else:
            policy = {
                key: checkpoint[key]
                for key in checkpoint.files
                if key.startswith(('w', 'b'))
            }
    if not policy:
        raise ValueError(f'Policy file has no supported network weights: {path}')
    weight_keys = sorted(
        (key for key in policy if key.startswith('w')),
        key=lambda key: int(key[1:]),
    )
    if not weight_keys:
        raise ValueError(f'Policy file has no weight matrices: {path}')
    expected_input = observation_dim
    for layer, key in enumerate(weight_keys):
        weights = np.asarray(policy[key])
        biases = np.asarray(policy.get(f'b{layer}'))
        if (
            weights.ndim != 2
            or weights.shape[0] != expected_input
            or biases.shape != (weights.shape[1],)
        ):
            raise ValueError(f'Policy layer {layer} has incompatible dimensions')
        expected_input = weights.shape[1]
    if expected_input != action_dim:
        raise ValueError('Policy output layer does not match the MuJoCo actuators')
    return policy


def _bounded_control(model, control):
    limited = model.actuator_ctrllimited.astype(bool)
    control[limited] = np.clip(
        control[limited],
        model.actuator_ctrlrange[limited, 0],
        model.actuator_ctrlrange[limited, 1],
    )
    return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='reach')
    parser.add_argument('--policy', default='')
    parser.add_argument('--out', default='data/demos')
    parser.add_argument('--xml', default='sim/g1/scene_mjx.xml')
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.nq - 7 != model.nu:
        raise ValueError('Expected a free-base G1 with one actuator per joint')
    if model.nu < 29:
        raise ValueError('The configured joint groups require at least 29 actuators')

    observation_dim = model.nq + model.nv
    action_dim = model.nu
    policy = _load_policy(args.policy, observation_dim, action_dim)
    target = np.asarray(model.qpos0[7 : 7 + action_dim], dtype=np.float32)
    control = target.copy()
    selection = ['3']
    takeover = [policy is None]
    stop = [False]
    input_lock = threading.Lock()
    demonstrations = {'obs': [], 'act': []}
    corrections = []

    def on_key(key):
        character = chr(key) if 32 <= key < 128 else ''
        message = None
        with input_lock:
            if character in GROUPS:
                selection[0] = character
                message = f"selected group {character}: {GROUPS[character]}"
            lo, hi = GROUPS[selection[0]]
            if key == KEY_UP:
                control[lo:hi] += 0.05
            elif key == KEY_DOWN:
                control[lo:hi] -= 0.05
            elif key == KEY_RIGHT:
                control[lo:hi] += 0.02
            elif key == KEY_LEFT:
                control[lo:hi] -= 0.02
            elif key == KEY_SPACE and policy is not None:
                takeover[0] = not takeover[0]
                message = 'TEACHER CONTROL' if takeover[0] else 'policy control'
            elif character in ('s', 'S'):
                stop[0] = True
            control[:] = _bounded_control(model, control)
        if message:
            print(message)

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    print('arrows=move selected joints | 1-4=joint group | SPACE=takeover | S=save and quit')
    if policy is None:
        print('No policy loaded; teacher control is active.')

    try:
        while viewer.is_running():
            with input_lock:
                if stop[0]:
                    break
                teacher_control = control.copy()
                teacher_has_control = takeover[0]
            observation = np.concatenate((data.qpos, data.qvel)).astype(np.float32)
            policy_action = None
            if policy is not None:
                policy_action = np.clip(
                    forward(policy, observation, np.tanh), -1.0, 1.0
                ).astype(np.float32)
                if policy_action.shape != (action_dim,) or not np.isfinite(policy_action).all():
                    raise ValueError('Policy produced an invalid action')

            if policy_action is not None and not teacher_has_control:
                data.ctrl[:] = _bounded_control(model, target + 0.5 * policy_action)
            else:
                data.ctrl[:] = _bounded_control(model, teacher_control)
                teacher_action = np.clip((data.ctrl - target) / 0.5, -1.0, 1.0)
                demonstrations['obs'].append(observation.copy())
                demonstrations['act'].append(teacher_action.astype(np.float32))
                if policy_action is not None:
                    corrections.append(
                        {
                            'obs': observation.tolist(),
                            'bad': policy_action.tolist(),
                            'good': teacher_action.tolist(),
                        }
                    )

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)
    finally:
        viewer.close()

    demo_path = Path(f'{args.out}_{args.task}.npz')
    correction_path = Path(f'data/corrections_{args.task}.jsonl')
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    correction_path.parent.mkdir(parents=True, exist_ok=True)
    obs = np.asarray(demonstrations['obs'], dtype=np.float32).reshape(-1, observation_dim)
    act = np.asarray(demonstrations['act'], dtype=np.float32).reshape(-1, action_dim)
    np.savez(demo_path, obs=obs, act=act)
    with correction_path.open('w', encoding='utf-8') as corrections_file:
        for correction in corrections:
            corrections_file.write(json.dumps(correction, allow_nan=False) + '\n')
    print(
        f'saved {len(obs)} teacher frames to {demo_path}; '
        f'{len(corrections)} corrections to {correction_path}'
    )


if __name__ == '__main__':
    main()
