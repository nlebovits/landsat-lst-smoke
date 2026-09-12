"""A loopback S3 that answers `Range`, so ranged staging can be tested.

`FakeS3` in `test_staging.py` stands in for the client. It proves the module
calls `get_object` once per object and counts what it calls, and it cannot
prove anything about bytes, because it never produces a `Content-Range` and
never has two parts of one object in flight at once.

Reassembly is exactly the part a fake client cannot check. `staging._fetch_part`
reads a total out of a header, computes offsets from it, and writes them with
`pwrite` from several threads into one descriptor. An off-by-one in the range
arithmetic, a total taken from the wrong side of the slash, or two parts
written at the same offset all produce a file of plausible length holding the
wrong bytes. Only real ranges over a real socket catch that.

Nothing here reaches the network. The server binds `127.0.0.1` on a port the
kernel picks, and the client is a real boto3 client pointed at it with dummy
credentials, so the request goes through botocore's signing, header and
streaming code rather than around it.

Objects come in two shapes. `add` holds the bytes, for the tests that compare
what landed against what was served. `add_sized` holds a size and generates
the bytes from a fixed 1 MiB block, so an 84 MB object costs a megabyte of
memory per connection in flight rather than 84, and the content is still
reproducible byte for byte.
"""

from __future__ import annotations

import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

#: The block every synthetic object is tiled from. One megabyte of
#: incompressible bytes, drawn once from a seeded generator so that two
#: processes reading the same offsets see the same content.
BLOCK = random.Random(20260912).randbytes(1024 * 1024)

#: Bytes per `wfile.write`. Bounds what one connection holds while it answers,
#: which is what lets 256 in-flight ranges of 16 MiB run in a test process.
CHUNK_BYTES = 1024 * 1024


class Stored:
    """One object: a size, and a way to produce any span of it."""

    def __init__(self, size: int, data: bytes | None = None):
        self.size = size
        self.data = data

    def span(self, start: int, length: int):
        """Yield the bytes from `start` in pieces of at most `CHUNK_BYTES`."""
        end = start + length
        while start < end:
            take = min(CHUNK_BYTES, end - start)
            yield self.at(start, take)
            start += take

    def at(self, start: int, length: int) -> bytes:
        """The object's bytes from `start`, held or generated."""
        if self.data is not None:
            return self.data[start : start + length]
        out = bytearray()
        offset = start
        while len(out) < length:
            i = offset % len(BLOCK)
            piece = BLOCK[i : i + (length - len(out))]
            out += piece
            offset += len(piece)
        return bytes(out)


class LocalS3(ThreadingHTTPServer):
    """The server, its objects, and what it saw.

    `daemon_threads`, so a test that fails partway through does not leave a
    connection thread holding the interpreter open.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *, latency_s: float = 0.0):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.store: dict[str, Stored] = {}
        self.latency_s = latency_s
        self.lock = threading.Lock()
        #: Every GET the server answered, and how many of them carried a
        #: `Range`. `staging.json`'s `get_requests` is checked against the
        #: first of these, which is the only count taken outside the code
        #: under test.
        self.requests = 0
        self.ranged = 0
        self.ranges: list[tuple[str, int, int]] = []
        #: Keys to fail, and how many times each. Counts down.
        self.fail: dict[str, int] = {}
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def add(self, bucket: str, key: str, data: bytes) -> str:
        """Hold an object's bytes. Returns its `s3://` href."""
        self.store[f"{bucket}/{key}"] = Stored(len(data), data)
        return f"s3://{bucket}/{key}"

    def add_sized(self, bucket: str, key: str, size: int) -> str:
        """Hold an object's size, and generate its bytes on demand."""
        self.store[f"{bucket}/{key}"] = Stored(size)
        return f"s3://{bucket}/{key}"

    def __enter__(self):
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        # The names are `BaseServer.__exit__`'s, because a subclass that
        # renames them is a different method to any caller passing keywords.
        del exc_type, exc_val, exc_tb
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)


class _Handler(BaseHTTPRequestHandler):
    """Path-style GET with `Range`. Nothing else, because nothing else is used.

    `HTTP/1.1`, so the connection stays open between requests. Staging's whole
    question is whether connections are reused, and a server that closed one
    per request would answer it wrong before the test began.
    """

    protocol_version = "HTTP/1.1"

    @property
    def s3(self) -> LocalS3:
        """The server, typed. `BaseHTTPRequestHandler.server` is a `BaseServer`."""
        return cast("LocalS3", self.server)

    def log_message(self, format, *args) -> None:  # noqa: A002 - the base names it
        """Silence. The server answers thousands of requests in one test."""

    def do_GET(self) -> None:  # noqa: N802 - the base class names it
        name = self.path.lstrip("/").split("?", 1)[0]
        obj = self.s3.store.get(name)
        if obj is None:
            self.send_error(404, "no such key")
            return
        if self._refused(name):
            return
        if self.s3.latency_s:
            time.sleep(self.s3.latency_s)
        header = self.headers.get("Range")
        start, length = _span(header, obj.size)
        with self.s3.lock:
            self.s3.requests += 1
            self.s3.ranged += header is not None
            self.s3.ranges.append((name, start, length))
        self.send_response(206 if header else 200)
        if header:
            self.send_header(
                "Content-Range", f"bytes {start}-{start + length - 1}/{obj.size}"
            )
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        for chunk in obj.span(start, length):
            self.wfile.write(chunk)

    def _refused(self, name: str) -> bool:
        """Fail this key if the test asked for it, and count the request.

        A failed GET is billed, so it counts. That is the whole reason
        `staging` retries in its own loop rather than botocore's.
        """
        with self.s3.lock:
            left = self.s3.fail.get(name, 0)
            if left <= 0:
                return False
            self.s3.fail[name] = left - 1
            self.s3.requests += 1
        self.send_error(500, "injected")
        return True


def _span(header: str | None, size: int) -> tuple[int, int]:
    """`bytes=a-b` against an object of `size`, as an offset and a length.

    An end past the last byte is clamped, which is what S3 does and what lets
    a fetch ask for a whole part without knowing whether one is left.
    """
    if not header:
        return 0, size
    first, _, last = header.split("=", 1)[1].partition("-")
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    return start, max(0, end - start + 1)


def client_for(server: LocalS3, connections: int = 10):
    """A real boto3 client pointed at the loopback server.

    Path-style addressing, because `bucket.127.0.0.1` does not resolve and is
    not meant to. The retry mode matches `staging._default_client`, so a
    retried GET is staging's retry and not botocore's, and the count the test
    reads is the count the module made.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=server.endpoint,
        aws_access_key_id="staging-test",
        aws_secret_access_key="staging-test",  # noqa: S106 - a loopback server
        region_name="us-east-1",
        config=Config(
            retries={"total_max_attempts": 1, "mode": "standard"},
            max_pool_connections=connections,
            s3={"addressing_style": "path"},
        ),
    )
