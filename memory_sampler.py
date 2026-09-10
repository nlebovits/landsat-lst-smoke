# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = ["psutil"]
# ///
"""Sample memory for a client process and its worker children.

The number this exists to produce is worker RSS at fleet width. `shard_bytes`
predicts it, and until this module was wired into `shard_lst_p95.py` nothing on
a production run measured it. `FINDINGS.md` carried a worst-shard figure in a
MEASURED table that was `shard_bytes` output, read back as evidence the model
held. A model cannot check itself.

Shared by `profile_lst_p95.py`, which has always sampled, and by
`shard_lst_p95.py`, which now does. Every fleet run writes a `memory.csv` and a
peak beside its summary, so the next instance that dies of memory leaves the
measurement behind rather than a prediction.

The sampler runs in its own process and must not be a thread. The client holds
the GIL for long stretches while it builds the graph, which is exactly when the
sampler must not drift.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import time
from pathlib import Path

#: Bytes per megabyte, the unit every column in the CSV is written in.
MB = 1024.0 * 1024.0


SAMPLER_HEADER = [
    "t_mono",
    "t_wall",
    "n_procs",
    "client_rss_mb",
    "workers_rss_mb",
    "tree_rss_mb",
    "tree_cpu_pct",
    "sys_used_mb",
    "sys_available_mb",
    "swap_used_mb",
    "net_recv_mb",
    "net_sent_mb",
    "disk_read_mb",
    "workers",
]


def _sampler_main(parent_pid: int, out_path: str, interval: float, stop_evt) -> None:
    """Poll the client process and its worker children until told to stop."""
    import psutil

    own_pid = os.getpid()
    try:
        parent = psutil.Process(parent_pid)
    except psutil.Error:
        return

    handles: dict[int, psutil.Process] = {parent_pid: parent}
    parent.cpu_percent(None)  # prime the delta

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SAMPLER_HEADER)

        while not stop_evt.is_set():
            t_mono = time.monotonic()
            t_wall = time.time()
            try:
                children = [
                    c for c in parent.children(recursive=True) if c.pid != own_pid
                ]
            except psutil.Error:
                break

            live = {parent_pid}
            client_rss = 0.0
            worker_rss = 0.0
            cpu = 0.0
            detail = []

            for proc in [parent, *children]:
                pid = proc.pid
                live.add(pid)
                handle = handles.get(pid)
                if handle is None:
                    handle = handles[pid] = proc
                    handle.cpu_percent(None)  # prime, first read is always 0.0
                try:
                    rss = handle.memory_info().rss / MB
                    cpu += handle.cpu_percent(None)
                except psutil.Error:
                    continue
                if pid == parent_pid:
                    client_rss = rss
                else:
                    worker_rss += rss
                    detail.append(f"{pid}:{rss:.0f}")

            for dead in set(handles) - live:
                handles.pop(dead, None)

            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            # Cumulative counters. The interesting number is the delta across
            # a stage, which is what tells us bytes actually pulled off the wire.
            try:
                net = psutil.net_io_counters()
                net_recv, net_sent = net.bytes_recv / MB, net.bytes_sent / MB
            except Exception:
                net_recv = net_sent = 0.0
            try:
                dio = psutil.disk_io_counters()
                disk_read = dio.read_bytes / MB if dio else 0.0
            except Exception:
                disk_read = 0.0
            writer.writerow(
                [
                    f"{t_mono:.4f}",
                    f"{t_wall:.4f}",
                    len(live),
                    f"{client_rss:.2f}",
                    f"{worker_rss:.2f}",
                    f"{client_rss + worker_rss:.2f}",
                    f"{cpu:.1f}",
                    f"{vm.used / MB:.1f}",
                    f"{vm.available / MB:.1f}",
                    f"{sw.used / MB:.1f}",
                    f"{net_recv:.3f}",
                    f"{net_sent:.3f}",
                    f"{disk_read:.3f}",
                    ";".join(detail),
                ]
            )
            fh.flush()
            time.sleep(interval)


class MemorySampler:
    """Start and stop the sampler process, then read its series back."""

    def __init__(self, out_path: Path, interval: float):
        self.out_path = out_path
        self.interval = interval
        self._ctx = mp.get_context("spawn")
        self._stop = self._ctx.Event()
        self._proc = None

    def start(self) -> None:
        self._proc = self._ctx.Process(
            target=_sampler_main,
            args=(os.getpid(), str(self.out_path), self.interval, self._stop),
            daemon=True,
        )
        self._proc.start()
        time.sleep(self.interval * 3)  # let the first samples land

    def stop(self) -> None:
        if self._proc is None:
            return
        self._stop.set()
        self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()

    def series(self) -> list[dict]:
        if not self.out_path.exists():
            return []
        with open(self.out_path, newline="") as fh:
            return list(csv.DictReader(fh))

    def peak_between(self, t0: float, t1: float) -> dict[str, float]:
        """Peak memory and net throughput observed inside a stage window."""
        rows = [r for r in self.series() if t0 <= float(r["t_mono"]) <= t1]
        if not rows:
            return {}
        span = float(rows[-1]["t_mono"]) - float(rows[0]["t_mono"])
        out = {
            "tree_rss_peak_mb": max(float(r["tree_rss_mb"]) for r in rows),
            "client_rss_peak_mb": max(float(r["client_rss_mb"]) for r in rows),
            "workers_rss_peak_mb": max(float(r["workers_rss_mb"]) for r in rows),
            "sys_available_min_mb": min(float(r["sys_available_mb"]) for r in rows),
            "samples": len(rows),
            "window_s": span,
        }
        if "net_recv_mb" in rows[0]:
            recv = float(rows[-1]["net_recv_mb"]) - float(rows[0]["net_recv_mb"])
            sent = float(rows[-1]["net_sent_mb"]) - float(rows[0]["net_sent_mb"])
            disk = float(rows[-1]["disk_read_mb"]) - float(rows[0]["disk_read_mb"])
            out |= {
                "net_recv_mb": recv,
                "net_sent_mb": sent,
                "disk_read_mb": disk,
                "net_recv_mb_s": (recv / span) if span > 0 else 0.0,
                # Peak instantaneous rate, to show whether throughput is flat
                # or bursty across the stage.
                "net_recv_mb_s_peak": max(
                    (
                        (float(b["net_recv_mb"]) - float(a["net_recv_mb"]))
                        / max(float(b["t_mono"]) - float(a["t_mono"]), 1e-6)
                    )
                    for a, b in zip(rows, rows[1:])
                )
                if len(rows) > 1
                else 0.0,
            }
        return out
