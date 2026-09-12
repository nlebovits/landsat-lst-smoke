"""The whole run on a real frisky cluster, offline, on synthetic scenes.

`tests/test_composite_graph.py` pins the graph under the synchronous
scheduler. This starts the driver the way a fleet instance does: a
two-process `frisky.LocalCluster`, the window writes from the workers into
one staging GeoTIFF, the COG copy, the trace collection, and the summary.
Every socket is blocked for the compute, so a read that reached a bucket
would fail here rather than on a billed instance.

Every run here is a rehearsal. Its output says so on every line.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import composite  # noqa: E402
import shard_lst_p95  # noqa: E402
from cog_catalog import read_cog_encoding  # noqa: E402
from lst_qa import LST_NODATA_DN, LST_OFFSET, LST_SCALE  # noqa: E402

pytestmark = pytest.mark.timeout(600)

TILE = "S30W065"
PPD = 120
CHUNK = 100
N_SCENES = 12


class NetworkBlocked(AssertionError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    """Every outbound connection raises. The frisky cluster listens on unix
    sockets and the dashboard on loopback, which `connect` to 127.0.0.1 still
    has to allow, so only non-loopback destinations are refused."""
    real_connect = socket.socket.connect

    def refuse(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and (host.startswith("127.") or host == "localhost"):
            return real_connect(self, address, *args, **kwargs)
        if isinstance(host, str) and host.startswith("/"):
            return real_connect(self, address, *args, **kwargs)
        raise NetworkBlocked(f"connect to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    yield


def rehearse(out_dir: Path, *extra: str, capsys=None) -> tuple[int, dict]:
    argv = [
        "--tile",
        TILE,
        "--rehearse",
        str(N_SCENES),
        "--pixels-per-degree",
        str(PPD),
        "--chunk",
        str(CHUNK),
        "--workers",
        "2",
        "--threads-per-worker",
        "1",
        "--memory-limit-gib",
        "2",
        "--no-output-mask",
        "--out-dir",
        str(out_dir),
        *extra,
    ]
    code = shard_lst_p95.main(argv)
    summary = json.loads((out_dir / "summary.json").read_text())
    return code, summary


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("run")
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code, summary = rehearse(out, "--no-catalog")
        return code, summary, out, buffer.getvalue()

    def test_it_exits_zero(self, run):
        code, _, _, _ = run
        assert code == 0

    def test_it_is_tainted_as_a_rehearsal(self, run):
        _, summary, _, stdout = run
        assert summary["synthetic"] is True
        lines = [line for line in stdout.splitlines() if line.strip()]
        assert lines, "the run printed nothing"
        assert all(line.startswith(shard_lst_p95.REHEARSAL_TAG) for line in lines), [
            line for line in lines if not line.startswith(shard_lst_p95.REHEARSAL_TAG)
        ][:5]

    def test_both_cogs_are_written_and_verified(self, run):
        _, summary, out, _ = run
        lst = read_cog_encoding(out / "lst_p95.tif")
        qa = read_cog_encoding(out / "qa_count.tif")
        assert lst["dtype"] == "uint16"
        assert lst["nodata"] == LST_NODATA_DN
        assert lst["scale"] == LST_SCALE
        assert lst["offset"] == LST_OFFSET
        assert lst["shape"] == summary["raster"]
        assert lst["block_shape"] == [512, 512]
        assert qa["dtype"] == "uint8"
        assert qa["count"] == 12
        assert qa["nodata"] is None
        assert all("STATISTICS_MEAN" in tags for tags in qa["statistics"])

    def test_the_pixels_are_a_composite_not_fill(self, run):
        _, summary, out, _ = run
        with rasterio.open(out / "lst_p95.tif") as src:
            lst = src.read(1)
        with rasterio.open(out / "qa_count.tif") as src:
            qa = src.read()
        valid = lst != LST_NODATA_DN
        assert 0.1 < valid.mean() < 0.9, valid.mean()
        celsius = lst[valid].astype("float64") * LST_SCALE + LST_OFFSET
        assert 20.0 < celsius.mean() < 70.0
        # A pixel with a temperature has observations, and one without has
        # none: the invariant the writer must not break.
        np.testing.assert_array_equal(valid, qa.sum(axis=0) > 0)
        assert summary["valid_fraction"] == pytest.approx(float(valid.mean()), abs=1e-6)

    def test_nothing_but_the_cogs_and_the_record_is_left(self, run):
        _, _, out, _ = run
        names = sorted(p.name for p in out.iterdir())
        assert not any(name.endswith((".npy", ".npz")) for name in names)
        assert not any(".staging." in name or name.endswith(".lock") for name in names)
        for required in (
            "summary.json",
            "spans.json",
            "events.json",
            "overview.json",
            "prefixes.json",
            "stragglers.json",
            "metrics.json",
            "timeline-component.txt",
            "memory.csv",
        ):
            assert required in names, required

    def test_frisky_traced_the_workers_and_the_driver(self, run):
        _, summary, out, _ = run
        spans = json.loads((out / "spans.json").read_text())
        names = {span["name"] for span in spans}
        assert "worker.exec.call" in names
        assert any(name.startswith("client.phase.") for name in names), sorted(names)[
            :20
        ]
        assert summary["frisky"]["n_driver_spans"] > 0
        # Phases before the cluster exists cannot reach a scheduler that does
        # not exist yet; they live in the driver's own spans. The phases the
        # dashboard can show live are the ones from the cluster on.
        assert "client.phase.stage" in names
        assert "client.phase.graph_build" in names
        # `collect_trace` starts the moment before the events are fetched, so
        # its own start event may not have landed yet; it is not required.
        phases = set(summary["frisky"]["client_phases"])
        for phase in ("graph_build", "compute"):
            assert phase in phases, phases
        # The record of how frisky ran the graph is kept, whatever it says.
        # On this dask the arrays carry no expression, so frisky runs every
        # compute as a stock dask graph and logs that as a fallback.
        assert "expression_fallbacks" in summary["frisky"]
        overview = json.loads((out / "overview.json").read_text())
        assert "perf" in overview, overview

    def test_the_summary_labels_its_figures(self, run):
        _, summary, _, _ = run
        assert summary["n_blocks"] == 36
        assert summary["chunk_px"] == CHUNK
        assert summary["n_scenes"] == N_SCENES
        assert summary["phases"]["compute_s"] > 0
        assert (
            summary["memory_demand_gib_derived"] is None
        )  # a rehearsal skips the guard
        assert summary["workers_rss_peak_gib"] > 0


class TestTheWriter:
    def test_two_processes_write_one_file_pixel_for_pixel(self, tmp_path, no_network):
        """The window writes from separate worker processes, against sync."""
        import dask

        import frisky
        import observe

        bbox = shard_lst_p95.tile_bounds(TILE)
        items, _ = composite.rehearsal_items(bbox, 6, tmp_path / "scenes")
        out = composite.build_graph(
            items, bbox, crs="EPSG:4326", resolution=1 / PPD, chunk=CHUNK
        )
        dims = composite.spatial_dims("EPSG:4326")

        with dask.config.set(scheduler="sync"):
            want = out["lst_p95"].values

        observe.enable(100_000)
        cluster = frisky.LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            memory_limit=2 * 1024**3,
            dashboard_address="127.0.0.1:0",
            silence_summary=True,
        )
        client = cluster.get_client()
        try:
            counts, paths = composite.staging_writes(
                out, tmp_path / "out", crs="EPSG:4326", dims=dims
            )
            flags = composite.compute_all(counts)
        finally:
            client.close()
            cluster.close()
        with rasterio.open(paths["lst_p95"]) as src:
            got = src.read(1)
            assert src.block_shapes[0] == (512, 512)
        np.testing.assert_array_equal(got, want)
        assert flags["valid"] == int((want != LST_NODATA_DN).sum())


class TestTheFusedEngine:
    """`--engine fused` on the same cluster: one task per block, same pixels."""

    @pytest.fixture(scope="class")
    def both(self, tmp_path_factory):
        import contextlib
        import io

        root = tmp_path_factory.mktemp("engines")
        summaries = {}
        for engine in ("graph", "fused"):
            with contextlib.redirect_stdout(io.StringIO()):
                code, summary = rehearse(
                    root / engine, "--no-catalog", "--engine", engine
                )
            assert code == 0, engine
            summaries[engine] = summary
        return root, summaries

    def test_the_two_engines_write_the_same_cogs(self, both):
        root, _ = both
        for name in ("lst_p95.tif", "qa_count.tif"):
            with (
                rasterio.open(root / "graph" / name) as a,
                rasterio.open(root / "fused" / name) as b,
            ):
                np.testing.assert_array_equal(a.read(), b.read(), err_msg=name)

    def test_the_two_engines_report_the_same_figures(self, both):
        _, summaries = both
        assert summaries["fused"]["engine"] == "fused"
        assert summaries["graph"]["engine"] == "graph"
        for key in (
            "valid_fraction",
            "n_pooled_fallback",
            "qa_count_per_month",
            "min_c",
            "mean_c",
            "max_c",
            "n_scenes_rejected",
        ):
            assert summaries["fused"][key] == summaries["graph"][key], key

    def test_the_fused_engine_submits_one_task_per_block(self, both):
        _, summaries = both
        assert summaries["fused"]["n_tasks"] == summaries["fused"]["n_blocks"]
        assert summaries["graph"]["n_tasks"] > summaries["graph"]["n_blocks"]

    def test_it_keeps_the_phases_the_summary_prints(self, both):
        root, summaries = both
        assert summaries["fused"]["phases"]["graph_build_s"] > 0
        assert summaries["fused"]["phases"]["compute_s"] > 0
        phases = set(summaries["fused"]["frisky"]["client_phases"])
        for phase in ("graph_build", "compute"):
            assert phase in phases, phases
        names = {
            span["name"]
            for span in json.loads((root / "fused" / "spans.json").read_text())
        }
        assert "worker.exec.fused_block" in names, sorted(names)[:20]


class TestPooledBaseline:
    def test_emit_pooled_writes_a_third_cog(self, tmp_path):
        out = tmp_path / "run"
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            code, _ = rehearse(out, "--no-catalog", "--emit-pooled")
        assert code == 0
        pooled = read_cog_encoding(out / "lst_p95_pooled.tif")
        assert pooled["nodata"] == LST_NODATA_DN
        assert pooled["scale"] == LST_SCALE
        with (
            rasterio.open(out / "lst_p95_pooled.tif") as a,
            rasterio.open(out / "lst_p95.tif") as b,
        ):
            # No prep artifact in a rehearsal, so the baseline is the product.
            np.testing.assert_array_equal(a.read(1), b.read(1))
