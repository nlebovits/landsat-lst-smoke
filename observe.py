"""Frisky observability for a composite run, driver side.

Frisky is a Rust scheduler for dask. Its workers trace themselves, and its
scheduler keeps every span, every event, and a live view of the cluster behind
a REST dashboard. Two things do not happen unless the driver asks for them,
and this module is where the driver asks.

Tracing in the driver process is off until `frisky.enable_tracing` runs
there. `record_span` is a silent no-op before that call, which is how the
prep pass recorded four phase spans for a week and kept none of them.
`enable()` is therefore the first thing a driver does.

A span with a name of the driver's own choosing never reaches the scheduler:
the wire accepts a fixed set of `client.*` names. A named driver phase reaches
the dashboard through `record_event` as a client phase, the way frisky
reports its own graph optimisation. `phase()` records both: the local span,
so the driver's own dump has it, and the event pair, so the live dashboard
shows the phase with an elapsed timer while it runs. That is the instrument
that shows a driver stall as it happens rather than as an absence afterwards.

After the compute, `collect()` pulls what the scheduler holds and what the
driver holds, because they are disjoint, and writes the files `frisky
observe` reads offline once the cluster is gone.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

#: Spans per process. MEASURED in FINDINGS.md: about 5% overhead at 500k
#: spans and 15.8% at 1.8M, so this is a ceiling rather than a target.
DEFAULT_TRACING_CAPACITY = 500_000

_TRACE_ID = 0


def enable(capacity: int = DEFAULT_TRACING_CAPACITY) -> int:
    """Turn tracing on in this process and size the workers' buffers.

    Returns the trace id every phase of this run carries. The environment
    variables are read by the spawned workers and by the scheduler; only
    `FRISKY_*` names cross into a worker process, so nothing else goes here.
    """
    global _TRACE_ID  # noqa: PLW0603
    import frisky

    os.environ.setdefault("FRISKY_TRACING_CAPACITY", str(capacity))
    os.environ.setdefault("FRISKY_EVENT_LOG_CAPACITY", "2000000")
    # The scheduler prints `=== STALL DETECTED ===` into `frisky logs` when
    # nothing moves for this long, which names a driver stall from the
    # cluster's side.
    os.environ.setdefault("FRISKY_STALL_TIMEOUT", "60s")
    frisky.enable_tracing(int(os.environ["FRISKY_TRACING_CAPACITY"]))
    _TRACE_ID = int(frisky.new_trace_id())
    return _TRACE_ID


def trace_id() -> int:
    return _TRACE_ID


def _event(kind: str, name: str, start_ns: int, end_ns: int | None, count) -> None:
    import frisky

    metadata = [
        ("phase", name),
        ("trace_id", str(_TRACE_ID)),
        ("start_ns", str(start_ns)),
    ]
    if end_ns is not None:
        metadata += [("end_ns", str(end_ns)), ("duration_ns", str(end_ns - start_ns))]
    if count is not None:
        metadata.append(("count", str(count)))
    frisky.record_event(kind, metadata=metadata)


@contextlib.contextmanager
def phase(name: str, *, count=None, marks: dict | None = None):
    """One named driver phase: a local span and a client-phase event pair.

    `marks`, when given, receives `<name>_s` with the phase's seconds, so the
    summary carries the same figure the trace does. Instrumentation never
    fails the run: every frisky call is wrapped.
    """
    t0 = time.perf_counter()
    start_ns = _now_ns()
    with contextlib.suppress(Exception):
        _event("client_phase_start", name, start_ns, None, count)
    try:
        yield
    finally:
        end_ns = _now_ns()
        if marks is not None:
            marks[f"{name}_s"] = time.perf_counter() - t0
        with contextlib.suppress(Exception):
            import frisky

            frisky.record_span(
                f"client.phase.{name}",
                start_ns,
                end_ns,
                trace_id=_TRACE_ID,
                count=count,
            )
            _event("client_phase_end", name, start_ns, end_ns, count)


def _now_ns() -> int:
    try:
        import frisky

        return int(frisky.now_ns())
    except Exception:  # noqa: BLE001
        return time.time_ns()


def dashboard_url(cluster) -> str:
    dash = str(cluster.dashboard_address)
    return dash if dash.startswith("http") else f"http://{dash}"


def _observe_json(url: str, *args: str) -> object:
    """One `frisky observe <cmd> --json` call, or the error it raised."""
    # The console script is `frisky`, wired to `frisky.cli:app`. Calling the
    # app through the interpreter running this driver finds the same install
    # whatever the PATH holds.
    cmd = [
        sys.executable,
        "-c",
        "import sys; from frisky.cli import app; sys.argv[0] = 'frisky'; app()",
        "observe",
        *args,
        url,
        "--json",
    ]
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": repr(exc), "command": cmd}
    if done.returncode != 0:
        return {"error": done.stderr[-2000:], "command": cmd}
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"error": "not json", "stdout": done.stdout[-2000:], "command": cmd}


def collect(cluster, out_dir: Path, *, span_limit: int = 2_000_000) -> dict:
    """Everything the run leaves behind for `frisky observe`, before teardown.

    Writes `spans.json` (scheduler spans and driver spans, merged, because
    `query_spans` sees the workers and `get_spans` sees only this process),
    `events.json`, `overview.json`, `prefixes.json`, `stragglers.json`,
    `metrics.json`, and `timeline-component.txt`. Returns the counts and any
    error, for the summary. Must run before `cluster.close()`: the scheduler's
    buffers die with it and nothing here is persisted anywhere else.
    """
    import urllib.request

    import frisky

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = dashboard_url(cluster)
    report: dict = {"dashboard": url}

    scheduler_spans: list = []
    try:
        scheduler_spans = frisky.query_spans(
            limit=span_limit, dashboard_url=url, request_timeout=120
        )
    except Exception as exc:  # noqa: BLE001
        report["span_error"] = repr(exc)
    driver_spans: list = []
    try:
        driver_spans = list(frisky.get_spans())
    except Exception as exc:  # noqa: BLE001
        report["driver_span_error"] = repr(exc)
    spans = scheduler_spans + driver_spans
    (out_dir / "spans.json").write_text(json.dumps(spans, default=str))
    report["n_spans"] = len(spans)
    report["n_driver_spans"] = len(driver_spans)

    # The endpoint serves at most 2,000 events a page, newest last, so the
    # unfiltered log would lose the early driver phases behind the task
    # traffic. The client events are the driver's own record and are fetched
    # on their own; the full log is kept for the counts it reports.
    events: dict = {}
    for name, query in (
        ("client_events", "kind=client_event&limit=2000"),
        ("recent_events", "limit=2000"),
    ):
        try:
            with urllib.request.urlopen(  # noqa: S310
                f"{url}/api/events?{query}", timeout=120
            ) as fh:
                events[name] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            events[name] = {"error": repr(exc)}
    (out_dir / "events.json").write_text(json.dumps(events, default=str))
    client_events = events["client_events"].get("events", [])
    report["n_client_events"] = len(client_events)
    report["n_events"] = events["recent_events"].get("event_log_total", 0)
    phases = {
        str(dict(event.get("metadata", [])).get("phase"))
        for event in client_events
        if event.get("client_event_kind") == "client_phase_start"
        and dict(event.get("metadata", [])).get("phase") is not None
    }
    report["client_phases"] = sorted(phases)
    report["expression_fallbacks"] = [
        dict(event.get("metadata", []))
        for event in client_events
        if event.get("client_event_kind") == "client_phase_end"
        and dict(event.get("metadata", [])).get("phase") == "dask_expression_fallback"
    ]

    for name in ("overview", "prefixes", "stragglers"):
        payload = _observe_json(url, name)
        (out_dir / f"{name}.json").write_text(json.dumps(payload, default=str))
        if isinstance(payload, dict) and "error" in payload:
            report[f"{name}_error"] = payload["error"]

    with contextlib.suppress(Exception):
        (out_dir / "metrics.json").write_text(
            json.dumps(frisky.get_metrics(), indent=2, default=str)
        )
    with contextlib.suppress(Exception):
        (out_dir / "timeline-component.txt").write_text(
            frisky.render_timeline(spans, view="component", width=100)
        )
    with contextlib.suppress(Exception):
        report["span_summary"] = {
            name: row
            for name, row in sorted(
                frisky.summarize_spans(spans).items(),
                key=lambda kv: -kv[1].get("total_ms", 0.0),
            )[:20]
        }
    return report


__all__ = [
    "DEFAULT_TRACING_CAPACITY",
    "collect",
    "dashboard_url",
    "enable",
    "phase",
    "trace_id",
]
