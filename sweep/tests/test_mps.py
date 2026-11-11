import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sweep import mps  # noqa: E402


def test_env_uses_private_dirs_under_root(tmp_path):
    e = mps.env(tmp_path)
    assert e["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "mps/pipe") == e["CLEF_MPS_DIR"]
    assert e["CUDA_MPS_LOG_DIRECTORY"] == str(tmp_path / "mps/log")
    assert "/tmp/nvidia-mps" not in "".join(e.values())   # never the default pipe dir other users' processes would join


def test_docker_flags_only_mount_when_enabled(tmp_path, monkeypatch):
    import subprocess
    script = Path(__file__).resolve().parents[2] / "serve/docker_env.sh"
    def flags(env):
        out = subprocess.run(["bash", "-c", f"source {script}; clef_docker_flags"], capture_output=True, text=True,
                             env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), **env}).stdout
        return out
    assert "CUDA_MPS_PIPE_DIRECTORY" not in flags({})
    on = flags({"CLEF_MPS_DIR": str(tmp_path / "pipe")})
    assert f"-v {tmp_path}/pipe:/mps" in on and "CUDA_MPS_PIPE_DIRECTORY=/mps" in on
