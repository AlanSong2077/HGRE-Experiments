"""Background zsh job and sleep/check primitives."""
from __future__ import annotations
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


class BackgroundJob:
    def __init__(self, command, cwd, timeout_s=1200.0, empty_workdir=False):
        self.command = str(command)
        self.cwd = Path(cwd)
        self.timeout_s = float(timeout_s)
        self.started_at = time.time()
        self.ended_at = None
        self.returncode = None
        self.timed_out = False
        self.log_dir = self.cwd.parent / (self.cwd.name + "_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.log_dir / "stdout.log"
        self.stderr_path = self.log_dir / "stderr.log"
        env = dict(os.environ)
        cache = self.cwd / ".cache"
        tmp = self.cwd / ".tmp"
        if not empty_workdir:
            cache.mkdir(parents=True, exist_ok=True)
            tmp.mkdir(parents=True, exist_ok=True)
        env.update({"CI": "1", "PIP_NO_INPUT": "1", "npm_config_yes": "true",
                    "DOCKER_CLI_HINTS": "false", "PYTHONUNBUFFERED": "1",
                    "HOME": str(self.cwd), "XDG_CACHE_HOME": str(cache),
                    "npm_config_cache": str(self.cwd / ".npm-cache"),
                    "PIP_CACHE_DIR": str(self.cwd / ".pip-cache"),
                    "TMPDIR": str(tmp)})
        self._out = open(self.stdout_path, "wb")
        self._err = open(self.stderr_path, "wb")
        self.proc = subprocess.Popen(["zsh", "-lc", self.command],
                                     cwd=str(self.cwd), env=env,
                                     stdout=self._out, stderr=self._err,
                                     start_new_session=True)
        self._monitor = threading.Thread(target=self._watch, daemon=True)
        self._monitor.start()
        self._watchdog = threading.Thread(target=self._watchdog_kill, daemon=True)
        self._watchdog.start()
        self.workdir_bytes_before = dir_bytes(self.cwd)

    def _watchdog_kill(self):
        # Independent of the agent's sleep/check calls.
        time.sleep(self.timeout_s)
        if self.proc.poll() is None:
            self.timed_out = True
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
            if self.ended_at is None:
                self.ended_at = time.time()

    def _watch(self):
        # Record the actual process-exit timestamp independent of the agent's sleep/check.
        while True:
            rc = self.proc.poll()
            if rc is not None:
                if self.ended_at is None:
                    self.ended_at = time.time()
                    self.returncode = rc
                return
            time.sleep(0.005)

    def poll(self):
        rc = self.proc.poll()
        if rc is not None and self.ended_at is None:
            self.ended_at = time.time()
            self.returncode = rc
        return rc

    def workdir_delta_bytes(self):
        return max(0, dir_bytes(self.cwd) - self.workdir_bytes_before)

    def elapsed_ms(self):
        end = self.ended_at or time.time()
        return (end - self.started_at) * 1000.0

    def tail(self, path, limit=800):
        try:
            data = path.read_bytes()
            return data[-limit:].decode("utf-8", "replace")
        except Exception:
            return ""

    def kill(self):
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        if self.ended_at is None:
            self.ended_at = time.time()
            self.returncode = self.proc.returncode

    def close(self):
        try:
            self._out.close(); self._err.close()
        except Exception:
            pass


def dir_bytes(path):
    total = 0
    try:
        for p in Path(path).rglob("*"):
            if p.is_file():
                try: total += p.stat().st_size
                except OSError: pass
    except Exception:
        pass
    return total


def start_job(command, cwd, timeout_s=1200.0, empty_workdir=False):
    return BackgroundJob(command, cwd, timeout_s=timeout_s, empty_workdir=empty_workdir)


def sleep_seconds(seconds):
    seconds = max(0.0, float(seconds))
    time.sleep(seconds)
