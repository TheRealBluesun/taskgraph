"""Signal and process-group plumbing for the lease wrapper (SPEC §5).

``taskgraph lease`` must never die before the command it guards, and must never
release a slot while that command's tree still runs.  The helpers here install
forwarding handlers *before* the slot is acquired, block terminating signals
while waiting, and empty the child's process group after the direct child exits
(SIGTERM, then SIGKILL).  A leased command runs in its own session, so the
wrapper's own process group is never signalled.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

#: Terminating signals the wrapper catches and forwards to the child's group.
TERM_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM)

#: How long a leased command's group gets to exit after SIGTERM before SIGKILL.
GROUP_GRACE = 5.0


def ignore_sigpipe() -> None:
    """Ignore SIGPIPE so a closed pipe in the leased command cannot kill us."""
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        pass


def block_terminating_signals() -> set | None:
    """Block :data:`TERM_SIGNALS` in this thread; return the previous mask.

    ``None`` means the platform has no ``pthread_sigmask`` (the wrapper then
    falls back to handler-only forwarding).  Blocking *before* the slot is
    acquired means no signal can kill the wrapper while it waits for one; the
    pending set is polled so the wait aborts and the mask is restored before the
    child is spawned.
    """
    try:
        return signal.pthread_sigmask(signal.SIG_BLOCK, TERM_SIGNALS)
    except (AttributeError, OSError, ValueError):
        return None


def restore_mask(previous: set | None) -> None:
    """Restore a mask returned by :func:`block_terminating_signals`."""
    if previous is None:
        return
    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    except (AttributeError, OSError, ValueError):
        pass


def pending_terminating_signal() -> int | None:
    """The first blocked :data:`TERM_SIGNALS` that is pending, else ``None``."""
    try:
        pending = signal.sigpending()
    except (AttributeError, OSError):
        return None
    for sig in TERM_SIGNALS:
        if sig in pending:
            return int(sig)
    return None


def swallow_signal(signum: int, previous: set | None) -> None:
    """Discard a pending blocked signal so unblocking cannot kill the wrapper."""
    try:
        signal.signal(signum, signal.SIG_IGN)
    except (OSError, ValueError, RuntimeError):
        pass
    restore_mask(previous)


class SignalForwarder:
    """Catch :data:`TERM_SIGNALS` and forward them to a child's process group.

    Handlers are installed *before* the slot is acquired (SPEC §5), so no signal
    can kill the wrapper between acquiring the slot and spawning the child; a
    signal that arrives before the child exists is queued and delivered by
    :meth:`attach`.  The signal mask is restored before the child is spawned
    because a fork inherits the mask: a child with these signals blocked would
    ignore exactly the signals the wrapper forwards to it.
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.pending: list[int] = []
        self._saved: dict[int, object] = {}

    def install(self) -> None:
        """Install a forwarding handler for every :data:`TERM_SIGNALS` member."""
        for sig in TERM_SIGNALS:
            try:
                self._saved[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError, RuntimeError):
                pass

    def attach(self, proc: subprocess.Popen) -> None:
        """Bind ``proc`` and forward any signal caught before it existed."""
        self.proc = proc
        queued, self.pending = self.pending, []
        for signum in queued:
            self._forward(signum)

    def restore(self) -> None:
        """Put back the handlers that were installed before :meth:`install`."""
        saved, self._saved = self._saved, {}
        for sig, previous in saved.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError, RuntimeError):
                pass

    def _handle(self, signum: int, _frame: object) -> None:
        if self.proc is None:
            self.pending.append(signum)
            return
        if self.proc.poll() is not None:
            return
        self._forward(signum)

    def _forward(self, signum: int) -> None:
        assert self.proc is not None
        try:
            os.killpg(self.proc.pid, signum)
        except OSError:
            try:
                self.proc.send_signal(signum)
            except OSError:
                pass


def group_empty(pgid: int) -> bool:
    """True when no process is left in process group ``pgid``."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return True
    return False


def wait_group_empty(pgid: int, timeout: float) -> bool:
    """Wait up to ``timeout`` for ``pgid`` to empty; True when it is empty."""
    deadline = time.monotonic() + timeout
    while True:
        if group_empty(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def terminate_group(pgid: int, grace: float = GROUP_GRACE) -> None:
    """SIGTERM, then after ``grace`` SIGKILL, whatever remains in ``pgid``.

    Called after the direct child has been reaped: grandchildren can outlive it
    (``sh -c 'build; test'``), and the slot must not be released while they
    still hold the resource (SPEC §5).
    """
    if group_empty(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        pass
    if wait_group_empty(pgid, grace):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        return
    wait_group_empty(pgid, grace)
