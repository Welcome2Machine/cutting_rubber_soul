# 运行环境

默认使用项目 `.venv/`，不修改全局 Python。约定见 [AGENT_CONTRACT.md](AGENT_CONTRACT.md)。本机已登记例外：`D:/Doctoral Research/AcademicForge/softcutting0731/.venv/Scripts/python.exe`，Python 3.11.9、PyTorch 2.11.0+cu128。

需要 Python 3.11、FFmpeg、FFprobe，以及 `requirements-core.txt` 和 PyTorch。旧 SAM、标注、Notebook 安装分支已移除。`requirements.lock.txt` 是此前科学计算环境的版本记录，不表示每个包仍被当前代码使用。

```powershell
# 仅需新建环境时运行；安装依赖但不下载模型
powershell -NoProfile -ExecutionPolicy Bypass -File environment/setup.ps1 -Cuda

# 只读验证
.venv/Scripts/python.exe environment/verify_environment.py
.venv/Scripts/python.exe -m pytest -q
```

不加 `-Cuda` 使用 CPU PyTorch。分割权重由本地保留目录提供，安装脚本不会训练或下载权重。计算证据应记录实际解释器、版本、输入、时间映射、参数和模型 SHA-256。
