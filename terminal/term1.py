#!/usr/bin/env python3
import sys
import time
from pathlib import Path

from rich.live import Live
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from primus.bus import get


def score(metrics):
    fps = min(metrics.get('fps', 0) / 30, 1)
    stt = max(0, 1 - min(metrics.get('stt_ms', 999) / 300, 1))
    ttfa = max(0, 1 - min(metrics.get('ttfa_ms', 2000) / 800, 1))
    yolo = max(0, 1 - min(metrics.get('yolo_ms', 99) / 20, 1))
    lag = max(0, 1 - min(metrics.get('cam_lag_ms', 999) / 100, 1))
    return 100 * (
        0.25 * fps + 0.25 * stt + 0.15 * ttfa + 0.2 * yolo + 0.15 * lag
    )


def make_table(metrics):
    value = score(metrics)
    color = 'green' if value > 80 else 'yellow' if value > 50 else 'red'
    table = Table(title='TERMINAL 1 — PERFORMANCE SCORE', expand=True)
    table.add_column('metric')
    table.add_column('value', justify='right')
    table.add_row('render fps', f"{metrics.get('fps', 0):.1f}")
    table.add_row('capture fps (C++)', f"{metrics.get('cam_fps', 0)}")
    table.add_row('cam resolution', metrics.get('cam_res', '?'))
    table.add_row('shm lag ms', f"{metrics.get('cam_lag_ms', 0):.1f}")
    table.add_row('yolo ms', f"{metrics.get('yolo_ms', 0):.1f}")
    table.add_row('stt ms', f"{metrics.get('stt_ms', 0):.0f}")
    table.add_row('ttfa ms', f"{metrics.get('ttfa_ms', 0):.0f}")
    table.add_row('SCORE', f'[{color}]{value:.0f}/100[/{color}]')
    return Panel(table, border_style=color)


def main():
    try:
        with Live(refresh_per_second=5) as live:
            while True:
                live.update(make_table(get()))
                time.sleep(0.2)
    except KeyboardInterrupt:
        print('Performance HUD stopped.')


if __name__ == '__main__':
    main()
