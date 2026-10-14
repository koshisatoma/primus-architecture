import argparse
import json
from pathlib import Path

import numpy as np


def rebuild(checkpoint, prefix):
    weight_keys = sorted(
        (
            key
            for key in checkpoint.files
            if key.startswith(f'{prefix}_w')
        ),
        key=lambda key: int(key.rsplit('w', 1)[1]),
    )
    if not weight_keys:
        raise ValueError(f'Checkpoint is missing {prefix!r} network weights')
    layer_count = len(weight_keys)
    expected = {
        f'{prefix}_{kind}{layer}'
        for layer in range(layer_count)
        for kind in ('w', 'b')
    }
    missing = expected.difference(checkpoint.files)
    if missing:
        raise ValueError(f'Checkpoint is missing network keys: {sorted(missing)}')
    return {
        f'{kind}{layer}': checkpoint[f'{prefix}_{kind}{layer}']
        for layer in range(layer_count)
        for kind in ('w', 'b')
    }


def forward(network, values, tanh):
    layer_count = len(network) // 2
    for layer in range(layer_count):
        values = values @ network[f'w{layer}'] + network[f'b{layer}']
        if layer < layer_count - 1:
            values = tanh(values)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='data/ckpt_stand.npz')
    parser.add_argument('--out', default='data/stand_policy')
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Also export ONNX (requires jax2onnx)',
    )
    args = parser.parse_args()

    with np.load(args.ckpt, allow_pickle=False) as checkpoint:
        for key in ('ls', 'obs_dim', 'nu'):
            if key not in checkpoint:
                raise ValueError(f'Checkpoint is missing {key!r}')
        policy = rebuild(checkpoint, 'pi')
        observation_dim = int(checkpoint['obs_dim'])
        action_dim = int(checkpoint['nu'])
        output_prefix = Path(args.out)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        weights_path = Path(f'{output_prefix}_weights.npz')
        np.savez(
            weights_path,
            **policy,
            ls=checkpoint['ls'],
            obs_dim=observation_dim,
            nu=action_dim,
        )

    print(f'saved {weights_path} (C++ spine loads this directly)')
    cpp_directory = Path(f'{output_prefix}_cpp')
    cpp_directory.mkdir(parents=True, exist_ok=True)
    for name, values in policy.items():
        values.astype('<f4', copy=False).tofile(cpp_directory / f'{name}.bin')
    spec = {
        'layers': len(policy) // 2,
        'obs': observation_dim,
        'act': action_dim,
    }
    with (cpp_directory / 'policy_spec.json').open(
        'w',
        encoding='utf-8',
    ) as spec_file:
        json.dump(spec, spec_file)
    print(f'saved {cpp_directory} (flat bins + spec for C++ spine)')

    if args.onnx:
        try:
            import jax.numpy as jnp
            from jax2onnx import to_onnx
        except ImportError as error:
            raise RuntimeError(
                'ONNX export requires JAX and jax2onnx'
            ) from error

        jax_policy = {key: jnp.asarray(value) for key, value in policy.items()}
        to_onnx(
            lambda values: forward(jax_policy, values, jnp.tanh),
            jnp.zeros(observation_dim),
            f'{output_prefix}.onnx',
        )
        print(f'saved {output_prefix}.onnx')

    action = forward(
        policy,
        np.zeros(observation_dim, dtype=np.float32),
        np.tanh,
    )
    if action.shape != (action_dim,) or not np.isfinite(action).all():
        raise ValueError('Exported policy produced an invalid zero-observation action')
    print('zero-obs action sample:', action[:4])


if __name__ == '__main__':
    main()
