#!/usr/bin/env python3
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gpu():
    try:
        output = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits',
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        )
        values = [value.strip() for value in output.strip().split(',')]
        if len(values) == 4:
            return values
    except (OSError, subprocess.SubprocessError):
        pass
    return ['0', '0', '0', '0']


def pod_status():
    url = os.environ.get('PRIMUS_POD', 'http://127.0.0.1:8000/health')
    try:
        response = requests.get(url, timeout=2)
        return f'ONLINE {response.status_code}'
    except requests.RequestException:
        return 'OFFLINE'


def make_panel():
    utilization, memory_used, memory_total, temperature = gpu()
    local = Table(title='LOCAL SYSTEM', expand=True)
    local.add_column('metric')
    local.add_column('value', justify='right')
    local.add_row('cpu %', f'{psutil.cpu_percent():.0f}')
    memory = psutil.virtual_memory()
    local.add_row(
        'ram GB',
        f'{memory.used / 2**30:.1f}/{memory.total / 2**30:.0f}',
    )
    local.add_row('gpu %', utilization)
    local.add_row(
        'vram GB',
        f'{float(memory_used) / 1024:.1f}/{float(memory_total) / 1024:.0f}',
    )
    local.add_row('temp C', temperature)

    pod = Table(title='RUNPOD POD', expand=True)
    pod.add_column('metric')
    pod.add_column('value', justify='right')
    pod.add_row('status', pod_status())
    pod.add_row('role', 'omni cortex (later)')
    return Panel(Columns([local, pod]), border_style='cyan')


def main():
    try:
        with Live(refresh_per_second=1) as live:
            while True:
                live.update(make_panel())
                time.sleep(1)
    except KeyboardInterrupt:
        print('System monitor stopped.')


if __name__ == '__main__':
    main()
