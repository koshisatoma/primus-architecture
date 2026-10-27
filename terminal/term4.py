import subprocess, time
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich import box

HINTS = {'vllm': 'cortex (vLLM)', 'moshi': 'voice (Moshi)', 'svc_percept': 'percept (STT/depth/reID)',
         'train_reflex': 'MJX training', 'train_astra': 'ASTRA training', 'app.py': 'local app',
         'collect_episodes': 'episode collection'}

def gpus():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu'],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return [tuple(x.strip() for x in l.split(',')) for l in out]

def procs():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory'],
                         capture_output=True, text=True).stdout.strip().splitlines()[1:]
    rows = []
    for l in out:
        if not l.strip(): continue
        pid, mem = [x.strip() for x in l.split(',')]
        try: cmd = open(f'/proc/{int(pid)}/cmdline', 'rb').read().decode(errors='ignore').replace('\0', ' ')
        except Exception: cmd = '?'
        name = next((v for k, v in HINTS.items() if k in cmd), cmd[:38])
        rows.append((int(pid), name, int(mem.split()[0])))
    return sorted(rows, key=lambda r: -r[2])

def build():
    tg = Table(box=box.SIMPLE_HEAD, title='GPU MEMORY')
    tg.add_column('GPU'); tg.add_column('Used/Total'); tg.add_column('Util'); tg.add_column('')
    free = 0
    for idx, used, total, util in gpus():
        u, t = int(used.split()[0]), int(total.split()[0]); free += t - u
        f = u / t
        tg.add_row(f'gpu{idx}', f'{u}/{t} MiB', f'{int(util.split()[0])}%',
                   f'[{"red" if f > 0.9 else "green"}]{"█"*int(f*28)}{"░"*(28-int(f*28))}[/]')
    tp = Table(box=box.SIMPLE_HEAD, title='PER-PROCESS VRAM')
    tp.add_column('PID'); tp.add_column('Service'); tp.add_column('MiB')
    for pid, name, mem in procs():
        tp.add_row(str(pid), name, str(mem))
    if not procs(): tp.add_row('-', 'idle', '0')
    return Group(tg, tp, Panel(f'Free VRAM: {free} MiB', style='bold cyan'))

if __name__ == '__main__':
    with Live(build(), refresh_per_second=0.5) as live:
        while True:
            time.sleep(2); live.update(build())