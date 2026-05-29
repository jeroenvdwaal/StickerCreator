"""Tests for the WorkerThread runner."""

import pytest

pytestmark = pytest.mark.requires_qt

from PySide6.QtCore import QObject, Signal, Slot

from sticker_creator.utils.worker_thread import WorkerThread


class _Worker(QObject):
    done = Signal(object)  # the QThread run() executed on

    @Slot()
    def run(self):
        from PySide6.QtCore import QThread

        self.done.emit(QThread.currentThread())


class TestWorkerThread:
    def test_run_once_fires_entry_point(self, qtbot):
        worker = _Worker()
        runner = WorkerThread(worker)
        with qtbot.waitSignal(worker.done, timeout=2000):
            runner.start(on_started=worker.run)
        assert runner.is_running
        runner.stop()
        assert not runner.is_running

    def test_runs_off_the_caller_thread(self, qtbot):
        from PySide6.QtCore import QThread

        worker = _Worker()
        runner = WorkerThread(worker)
        with qtbot.waitSignal(worker.done, timeout=2000) as blocker:
            runner.start(on_started=worker.run)
        runner.stop()
        # The thread captured inside run() must differ from the test's thread.
        assert blocker.args[0] is not QThread.currentThread()

    def test_stop_is_idempotent(self, qtbot):
        worker = _Worker()
        runner = WorkerThread(worker)
        runner.start()
        runner.stop()
        runner.stop()  # second stop must not raise
        assert not runner.is_running
