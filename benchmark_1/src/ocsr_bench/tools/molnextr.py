"""MolNexTR, run in its own Python environment through a persistent worker process."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils import resolve_device
from .base import OCSRTool, RawPrediction, ToolSetupError, register_tool
from .common import HF_TOKEN_ENV, clone_repo, git_head

logger = logging.getLogger("ocsr_bench")

# Worker script executed *inside the isolated environment*. It must not use triple double quotes.
MOLNEXTR_WORKER_SOURCE = r"""
import argparse
import json
import sys
import traceback


def numbers(value):
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in numbers(v)]
    return []


def find_confidence(out):
    # Best effort: numeric values stored under any key containing 'conf'; top level first, else per atom/bond.
    top = [n for k, v in out.items() if "conf" in str(k).lower() for n in numbers(v)]
    if top:
        return sum(top) / len(top), "top-level"
    nested = []
    for key in ("atom_sets", "bonds_sets"):
        for item in out.get(key) or []:
            if isinstance(item, dict):
                for k, v in item.items():
                    if "conf" in str(k).lower():
                        nested.extend(numbers(v))
    return (sum(nested) / len(nested), "per-atom/bond mean") if nested else (None, None)


def summarize(out):
    if not isinstance(out, dict):
        return {"error": "unexpected output type " + type(out).__name__}
    confidence, source = find_confidence(out)
    return {"smiles": out.get("predicted_smiles"), "confidence": confidence,
            "extra": {"output_keys": sorted(str(k) for k in out), "confidence_source": source}}


def find_model_class(package):
    # The README and prediction.py call molnextr(checkpoint, device).predict_final_results(...), but in the installed
    # package the name `molnextr` can be bound to the submodule of that name. So find the class by what it can do.
    import inspect
    modules = [package] + [m for n, m in list(sys.modules.items()) if n.startswith("MolNexTR.") and m is not None]
    found = []
    for module in modules:
        for obj in list(vars(module).values()):
            if inspect.isclass(obj) and hasattr(obj, "predict_final_results") and obj not in found:
                found.append(obj)
    for preferred in ("molnextr", "MolNexTR"):
        for cls in found:
            if cls.__name__ == preferred:
                return cls
    return found[0] if found else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr  # stray library prints must not corrupt the protocol channel

    def send(obj):
        protocol.write(json.dumps(obj) + "\n")
        protocol.flush()

    try:
        sys.path.insert(0, args.repo)
        import torch
        import MolNexTR as package
        factory = find_model_class(package)
        if factory is None:
            raise RuntimeError("Could not find a class with a predict_final_results method in the MolNexTR package. "
                               "Package names: " + ", ".join(sorted(n for n in dir(package) if not n.startswith("_")))
                               + ". Loaded modules: " + ", ".join(sorted(n for n in sys.modules if n.startswith("MolNexTR"))))
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        model = factory(args.checkpoint, device)
    except BaseException:
        send({"ready": False, "error": traceback.format_exc()})
        return 1
    send({"ready": True, "torch": torch.__version__, "device": str(device), "python": sys.version.split()[0],
          "class": factory.__module__ + "." + factory.__name__})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("op") == "quit":
            break
        results = []
        for path in request["paths"]:
            try:
                out = model.predict_final_results(path, return_atoms_bonds=True, return_confidence=True)
                results.append(summarize(out))
            except Exception:
                results.append({"error": traceback.format_exc(limit=4)})
        peak = None
        try:
            if torch.cuda.is_available():
                peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        except Exception:
            pass
        send({"results": results, "peak_gpu_mb": peak})
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


class WorkerProcess:
    """A long-lived child process that exchanges one JSON object per line over stdin / stdout."""

    def __init__(self, command: List[str], env: Dict[str, str], log_path: Path) -> None:
        self.command, self.env, self.log_path = command, env, log_path
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._log: Any = None

    def log_tail(self, n_chars: int = 2500) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-n_chars:]
        except OSError:
            return ""

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in iter(self._proc.stdout.readline, ""):
                self._lines.put(line)
        except (ValueError, OSError):  # the pipe was closed by stop()
            pass
        self._lines.put(None)  # end of stream

    def _read(self, timeout: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self._lines.get(timeout=max(deadline - time.monotonic(), 0.01))
            except queue.Empty:
                raise ToolSetupError(f"Worker gave no reply within {timeout:.0f}s. Worker log tail:\n{self.log_tail()}")
            if line is None:
                raise ToolSetupError(f"Worker process exited unexpectedly. Worker log tail:\n{self.log_tail()}")
            try:
                message = json.loads(line)
            except ValueError:
                continue  # stray non-protocol output
            if isinstance(message, dict):
                return message

    def start(self, ready_timeout: float) -> Dict[str, Any]:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                      text=True, bufsize=1, env=self.env)
        threading.Thread(target=self._pump, daemon=True).start()
        return self._read(ready_timeout)

    def request(self, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ToolSetupError(f"Worker is not accepting requests ({exc}). Log tail:\n{self.log_tail()}") from exc
        return self._read(timeout)

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        """Ask the worker to quit, kill it if it does not, and release every pipe and the log file."""
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin:
                    proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=15)
            except Exception:
                pass
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass
        if self._log:
            self._log.close()
            self._log = None


MOLNEXTR_DEFAULTS: Dict[str, Any] = {
    "checkpoint_repo": "CYF200127/MolNexTR",
    "checkpoint_repo_type": "dataset",
    "checkpoint_filename": "molnextr_best.pth",
    "checkpoint_path": None,
    "repo_url": "https://github.com/CYF2000127/MolNexTR.git",
    "repo_ref": None,
    "repo_path": None,
    "python_version": "3.8",
    "torch_spec": "torch",
    "extra_requirements": [],
    "python_executable": None,   # use an existing interpreter instead of building an environment
    "venv_dir": None,
    "worker_env": {},            # extra environment variables for the worker process
    "ready_timeout_s": 900,
    "request_timeout_s": 900,
}


@register_tool("molnextr")
class MolNexTRTool(OCSRTool):
    """MolNexTR run in its own virtual environment through a persistent worker process."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.params = {**MOLNEXTR_DEFAULTS, **self.spec.params}
        self._worker: Optional[WorkerProcess] = None
        self._command: List[str] = []
        self._repo: Optional[Path] = None
        self._ready: Dict[str, Any] = {}
        self._peak_gpu_mb: Optional[float] = None
        self._env_freeze: List[str] = []

    @property
    def _work_dir(self) -> Path:
        return self.data_dir / "_envs"

    # ---- weights and code -------------------------------------------------------------------
    def _fetch_checkpoint(self) -> Path:
        local = self.params["checkpoint_path"]
        if local:
            path = Path(str(local)).expanduser()
            if not path.is_file():
                raise ToolSetupError(f"checkpoint_path does not exist: {path}")
            return path
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ToolSetupError("huggingface_hub is required to download the checkpoint.") from exc
        try:
            return Path(hf_hub_download(repo_id=self.params["checkpoint_repo"], filename=self.params["checkpoint_filename"],
                                        repo_type=self.params["checkpoint_repo_type"],
                                        token=os.environ.get(HF_TOKEN_ENV) or None))
        except Exception as exc:
            raise ToolSetupError(f"Could not download {self.params['checkpoint_filename']} from "
                                 f"{self.params['checkpoint_repo']}: {type(exc).__name__}: {exc}") from exc

    def _ensure_repo(self) -> Path:
        given = self.params["repo_path"]
        if given:
            repo = Path(str(given)).expanduser().resolve()
            if not repo.is_dir():
                raise ToolSetupError(f"repo_path is not a directory: {repo}")
        else:
            repo = clone_repo(self.params["repo_url"], self.data_dir / "_third_party" / "MolNexTR", self.params["repo_ref"])
        if not (repo / "MolNexTR" / "__init__.py").is_file():
            raise ToolSetupError(f"{repo} does not contain MolNexTR/__init__.py; the repository layout may have "
                                 "changed. Pin repo_ref to a known commit.")
        return repo

    # ---- isolated environment ---------------------------------------------------------------
    def _uv(self, args: List[str], log: Path, what: str) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as sink:
            sink.write(f"\n$ uv {' '.join(args)}\n")
            sink.flush()
            result = subprocess.run([sys.executable, "-m", "uv", *args], stdout=sink, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
            raise ToolSetupError(f"Building MolNexTR's environment failed at step '{what}'. Last log lines "
                                 f"({log}):\n{tail}\nAdd or loosen packages via the tool's 'extra_requirements' / "
                                 "'torch_spec', or point 'python_executable' at an interpreter that already works."
                                 + (" If uv cannot provide this Python version, try python_version '3.9' or '3.10'."
                                    if what == "create venv" else ""))

    def ensure_python(self, repo: Path) -> Path:
        """Interpreter that has MolNexTR's dependencies: an explicit one, or a cached `uv` virtual environment."""
        explicit = self.params["python_executable"]
        if explicit:
            if not Path(str(explicit)).exists():
                raise ToolSetupError(f"python_executable does not exist: {explicit}. On the cluster, create the "
                                     "MolNexTR environment first (slurm/setup_envs.sbatch).")
            return Path(str(explicit))
        env_dir = Path(str(self.params["venv_dir"])) if self.params["venv_dir"] else self._work_dir / "molnextr"
        python, marker = env_dir / "bin" / "python", env_dir / ".env_ok"
        req_file = repo / "requirements.txt"
        requirements = req_file.read_text(encoding="utf-8") if req_file.is_file() else ""
        fingerprint = hashlib.sha256(json.dumps([self.params["python_version"], self.params["torch_spec"],
                                                 requirements, self.params["extra_requirements"]]).encode()).hexdigest()[:16]
        if python.exists() and marker.exists() and marker.read_text() == fingerprint:
            return python
        log = self._work_dir / "molnextr_env_install.log"
        if subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True).returncode != 0:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=False)
        logger.info("Building MolNexTR's isolated environment (Python %s). First time only; this takes several minutes.",
                    self.params["python_version"])
        shutil.rmtree(env_dir, ignore_errors=True)
        self._uv(["venv", str(env_dir), "--python", str(self.params["python_version"])], log, "create venv")
        self._uv(["pip", "install", "--python", str(python), str(self.params["torch_spec"])], log, "install torch")
        extra = [str(x) for x in self.params["extra_requirements"]]
        self._uv(["pip", "install", "--python", str(python), *(["-r", str(req_file)] if requirements else []), *extra],
                 log, "install requirements")
        marker.write_text(fingerprint)
        return python

    def _freeze(self, python: Path) -> List[str]:
        """Package versions installed in the worker's environment (stored in the run metadata)."""
        for command in ([sys.executable, "-m", "uv", "pip", "freeze", "--python", str(python)],
                        [str(python), "-m", "pip", "freeze"]):
            try:
                out = subprocess.run(command, capture_output=True, text=True, timeout=120)
            except (OSError, subprocess.SubprocessError):
                continue
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()
        return []

    # ---- worker lifecycle --------------------------------------------------------------------
    def _start_worker(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update({"PYTHONUNBUFFERED": "1", **{str(k): str(v) for k, v in self.params["worker_env"].items()}})
        if self._worker is not None:  # restarting after a crash: release the dead worker's pipes first
            self._worker.stop()
        self._worker = WorkerProcess(self._command, env, self._work_dir / "molnextr_worker.log")
        ready = self._worker.start(float(self.params["ready_timeout_s"]))
        if not ready.get("ready"):
            self._worker.stop()
            raise ToolSetupError("MolNexTR failed to start inside its environment:\n" + str(ready.get("error", ready)) +
                                 "\nIf a package is missing, add it to 'extra_requirements'.")
        self._ready = ready
        logger.info("MolNexTR worker ready (Python %s, torch %s, device %s, model class %s)", ready.get("python"),
                    ready.get("torch"), ready.get("device"), ready.get("class"))

    def _prepare_command(self, device: str) -> None:
        """Fetch the checkpoint and code, make sure the interpreter exists, and build the worker command."""
        checkpoint = self._fetch_checkpoint()
        self._repo = self._ensure_repo()
        python = self.ensure_python(self._repo)
        self._env_freeze = self._freeze(python)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        script = self._work_dir / "molnextr_worker.py"
        script.write_text(MOLNEXTR_WORKER_SOURCE, encoding="utf-8")
        self._command = [str(python), str(script), "--repo", str(self._repo), "--checkpoint", str(checkpoint),
                         "--device", device]

    def load(self) -> None:
        self._prepare_command(resolve_device(self.run["device"]))
        self._start_worker()

    def predict_batch(self, image_paths: Sequence[Path]) -> List[RawPrediction]:
        if self._worker is None:
            raise ToolSetupError("predict_batch() called before load().")
        if not self._worker.alive():
            logger.warning("MolNexTR worker died; restarting it.")
            self._start_worker()
        reply = self._worker.request({"paths": [str(p) for p in image_paths]}, float(self.params["request_timeout_s"]))
        results = reply.get("results", [])
        if reply.get("peak_gpu_mb") is not None:
            self._peak_gpu_mb = reply["peak_gpu_mb"]
        if len(results) != len(image_paths):
            raise RuntimeError(f"worker returned {len(results)} results for {len(image_paths)} images")
        failed = [r["error"] for r in results if r.get("error")]
        if failed:  # make the pipeline retry image by image so each failure is attributed to its image
            raise RuntimeError(str(failed[0]).strip().splitlines()[-1][:300])
        return [RawPrediction(smiles=r.get("smiles"), confidence=r.get("confidence"), extra=r.get("extra") or {})
                for r in results]

    def prefetch(self) -> None:
        """Download the checkpoint and code, then start the worker once on the CPU.

        That proves, before any GPU is requested, that MolNexTR imports in its environment and that the
        checkpoint loads.
        """
        self._prepare_command("cpu")
        self._start_worker()
        self.close()

    def resource_summary(self) -> Dict[str, Any]:
        return {"molnextr_commit": git_head(self._repo) if self._repo else None, "worker_python": self._ready.get("python"),
                "worker_torch": self._ready.get("torch"), "worker_class": self._ready.get("class"),
                "peak_gpu_memory_mb": self._peak_gpu_mb,
                "environment_freeze": self._env_freeze}

    def close(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
