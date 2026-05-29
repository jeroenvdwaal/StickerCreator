"""Run a ``QObject`` worker on its own ``QThread``.

Owns the thread lifecycle that was hand-rolled in both the segmenter and the
model-downloader: move the worker onto a fresh thread, start it, and tear it
down with the correct ``quit`` → ``wait`` sequence. Signal forwarding and the
worker's own slots stay with the caller, because they vary per worker — only
the thread plumbing is shared.

Two launch shapes are supported:

* **Persistent** — ``start()`` with no argument. The thread stays alive and the
  caller routes work to the worker via queued-connection signals (the
  segmenter pattern).
* **Run-once** — ``start(on_started=worker.run)``. The worker's entry point
  fires as soon as the thread starts (the downloader pattern).
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QThread


class WorkerThread:
    """Owns a ``QThread`` running *worker*; centralizes start/teardown."""

    def __init__(self, worker: QObject):
        self._worker = worker
        self._thread = QThread()
        worker.moveToThread(self._thread)
        # The worker lives and dies with its thread.
        self._thread.finished.connect(worker.deleteLater)

    @property
    def worker(self) -> QObject:
        return self._worker

    @property
    def is_running(self) -> bool:
        return self._thread.isRunning()

    def start(self, on_started: Callable[[], None] | None = None) -> None:
        """Start the thread. If *on_started* is given, run it on thread start."""
        if on_started is not None:
            self._thread.started.connect(on_started)
        self._thread.start()

    def stop(self, timeout_ms: int = 2000) -> None:
        """Quit the thread and block until it exits (up to *timeout_ms*)."""
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(timeout_ms)
