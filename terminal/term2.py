#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import mujoco
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from primus.bus import put
from primus.rigid import Rigid
from primus.scene_builder import load_model


console = Console()
REPORT_INTERVAL = 0.1


def physics_score(metrics):
    tilt_penalty = max(0, 1 - abs(metrics['tilt']) / 30)
    support_score = min(1, metrics['support_contacts'] / 4)
    slip_score = max(0, 1 - metrics['slip'])
    height_score = max(0, min(1, (metrics['torso_z'] - 0.3) / 0.5))
    return 100 * (
        0.35 * tilt_penalty
        + 0.25 * support_score
        + 0.2 * slip_score
        + 0.2 * height_score
    )


def make_table(metrics):
    table = Table(
        title='TERMINAL 2 — G1 RIGID-BODY TELEMETRY',
        expand=True,
        show_lines=True,
    )
    table.add_column('group / metric', style='cyan bold')
    table.add_column('value', style='green bold', justify='right')
    table.add_row(
        '[b]BODY[/]',
        f"DoF {metrics['nv']} · q {metrics['nq']} · "
        f"actuators {metrics['nu']} · bodies {metrics['nbody']}",
    )
    table.add_row(
        'mass / gravity',
        f"{metrics['mass']:.1f} kg / {metrics['grav']:.2f} m/s²",
    )
    table.add_row('[b]KINEMATICS[/]', '')
    table.add_row('torso height', f"{metrics['torso_z']:.3f} m")
    table.add_row('torso tilt', f"{metrics['tilt']:.1f}°")
    table.add_row(
        'CoM',
        f"({metrics['com'][0]:.2f}, {metrics['com'][1]:.2f}, "
        f"{metrics['com'][2]:.2f}) m",
    )
    table.add_row('CoM speed', f"{metrics['com_v']:.2f} m/s")
    table.add_row('qvel rms', f"{metrics['qvel_rms']:.2f} rad/s")
    table.add_row('[b]DYNAMICS[/]', '')
    table.add_row('max actuator effort', f"{metrics['maxtq']:.1f} N·m")
    table.add_row('actuator power', f"{metrics['power']:.1f} W")
    table.add_row('KE / PE', f"{metrics['ke']:.1f} / {metrics['pe']:.1f} J")
    table.add_row('angular momentum', f"{metrics['angmom']:.2f} kg·m²/s")
    table.add_row('[b]CONTACTS[/]', '')
    table.add_row(
        'contacts / ground support',
        f"{metrics['ncon']} / {metrics['support_contacts']}",
    )
    table.add_row(
        'normal force / gravity comp.',
        f"{metrics['fn']:.0f} N / {metrics['gravity_comp']:.2f}×",
    )
    table.add_row(
        'ZMP',
        f"({metrics['zmp'][0]:.2f}, {metrics['zmp'][1]:.2f}) m",
    )
    table.add_row('slip ratio', f"{metrics['slip']:.2f}")

    score = physics_score(metrics)
    color = 'green' if score > 80 else 'yellow' if score > 50 else 'red'
    table.add_row('PHYSICS SCORE', f'[{color}]{score:.1f}/100[/{color}]')
    return Panel(table, border_style=color)


def main():
    model = load_model()
    data = mujoco.MjData(model)
    rigid = Rigid(model, data)
    mujoco.mj_forward(model, data)
    steps_per_report = max(
        1, round(REPORT_INTERVAL / float(model.opt.timestep))
    )

    try:
        with Live(make_table(rigid.snap()), refresh_per_second=10) as live:
            while True:
                started = time.monotonic()
                for _ in range(steps_per_report):
                    mujoco.mj_step(model, data)
                metrics = rigid.snap()
                put(physics_score=physics_score(metrics))
                live.update(make_table(metrics))
                remaining = REPORT_INTERVAL - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        console.print('Physics HUD stopped.')


if __name__ == '__main__':
    main()
