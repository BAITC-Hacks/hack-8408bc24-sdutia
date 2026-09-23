"""Black-box CLI harness: real child processes, isolated writes, no network."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli(tmp_path):
    outputs = tmp_path / "outputs"
    guard = tmp_path / "network_guard"
    guard.mkdir()
    network_log = tmp_path / "network_attempts.log"
    # Only block external I/O; do not replace application functions or responses.
    # This catches commands that ignore WINDAGENT_OFFLINE without reaching an API.
    (guard / "sitecustomize.py").write_text(
        "import os, sys\n"
        "def block_network(event, args):\n"
        "    if event in {'socket.connect', 'socket.getaddrinfo'}:\n"
        "        with open(os.environ['QA_NETWORK_LOG'], 'a', encoding='utf-8') as log:\n"
        "            log.write(event + '\\n')\n"
        "        raise RuntimeError('QA network access disabled')\n"
        "sys.addaudithook(block_network)\n",
        encoding="utf-8",
    )

    def run(*args, env=None):
        child_env = os.environ.copy()
        child_env.update({
            "WINDAGENT_DATA_DIR": str(REPO / "data"),
            "WINDAGENT_MODELS_DIR": str(REPO / "models"),
            "WINDAGENT_CONFIG": str(REPO / "config" / "sites.yaml"),
            "WINDAGENT_OUTPUTS_DIR": str(outputs),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        })
        if env:
            child_env.update({key: str(value) for key, value in env.items()})
        # These constraints cannot be overridden by a test or a local .env file.
        child_env.update({
            "WINDAGENT_OFFLINE": "1",
            "LLM_PROVIDER": "none",
            "OPENAI_API_KEY": "",
            "NVIDIA_API_KEY": "",
            "QA_NETWORK_LOG": str(network_log),
            "PYTHONPATH": os.pathsep.join((str(guard), str(REPO / "src"))),
        })
        return subprocess.run(
            [sys.executable, "-m", "windagent", *map(str, args)],
            cwd=REPO,
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )

    run.outputs = outputs
    run.repo = REPO
    run.network_log = network_log
    return run
