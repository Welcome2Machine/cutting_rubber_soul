# Agent 运行环境契约

此文件只规定运行环境，不记录研究历史。能力与版本以实际验证为准，不以旧文档或旧报告为准。

- 默认解释器：`.venv\Scripts\python.exe`。
- 验证命令：`.\.venv\Scripts\python.exe .\environment\verify_environment.py`。
- 依赖基线：`environment\requirements.lock.txt`。
- 禁止静默使用全局 Python 或 Conda 根环境；若确需例外，必须在会话证据卡中说明。
- 新依赖、模型权重、外部 API、数据上传或高资源计算须在研究决策卡中获授权。
- 运行结束时，保留的正式资产必须在证据卡中列出；临时文件不得作为后续科学结论的唯一依据。
