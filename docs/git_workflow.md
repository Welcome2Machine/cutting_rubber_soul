# Git 协作与发布

2026-11-27。当前科研任务只见 [唯一任务书](../researchtopic/research_prototype_task.md)。

## 1. 唯一活动分支与写入者

本轮使用 `codex/deformation-sliding-baseline`。服务器 agent 是数值执行者，本地 agent 暂停。负责人可维护交接与文档；修改共享代码前确认没有另一执行者同时写入或运行它。

`main`、`codex/server-fracture-20261029`、`codex/intact-baseline-upgrade-20261031` 等保留历史，不自动整分支合并或重启旧任务。FE 与视频子目录仍是普通目录，不建立嵌套 Git。新工作根 `<USER_WORK_ROOT>` 不要求复制旧虚拟环境。

## 2. 接手与源码身份

先核对远端仓库、HEAD、分支、index、未提交项、Slurm 作业及其实际源码路径。已有脏工作区原状保留；运行中的源码不 pull、切分支或改核心。需要新版本时使用 Git worktree，不复制冻结源码树。

```bash
git fetch origin codex/deformation-sliding-baseline
task_sha=$(git rev-parse FETCH_HEAD)
git show "$task_sha":researchtopic/research_prototype_task.md
```

保存固定 SHA 后再做其他 fetch。仅在本地分支正确、工作区干净且未被作业使用时正常快进。新机器 clone 时显式选活动分支，默认 `main` 不是当前任务入口。

交接写明分支、提交、实际解释器、数值源码与输入 SHA、结果和审查身份。Release 的历史 patch 仅用于来源复核；当前分支已经纳入的改动不得重复 apply。源码更新不自动授权旧状态跨签名恢复；用已有身份规则核对，必要时从一致初态重算。不同来源物性和内部变量不可混用。

## 3. 提交与增量发布

只暂存本次负责路径；不使用 `git add .` 收走他人修改。保持普通前进提交，不 force push、不 reset/clean/stash 他人工作，不改写共享历史。远端前进时先重新审阅；共享分支不自动覆盖。

提交前运行根 `python -m pytest -q`、直接相关 FE 回归、`git diff --check`、`python tools/check_publish.py --staged`。文档更新不触发研究长算例。`.githooks` 仍正常启用，不能使用 `--no-verify` 或换 API 绕过失败。

发布检查的职责：

- `--staged`：检查本次暂存的新增/修改文件。
- `--ref HEAD`：严格审计全部可达历史，供历史债务盘点。
- `--pre-push`：对 Git 提供的每条真实远端更新，先严格检查当前树，再核对远端 commit 存在且为祖先，检查全部新增历史中的新 `(路径, blob)`。远端已公开的完全相同历史对象不重复当作新增泄漏。新分支没有基线时仍全历史严格审查。

增量规则不会放过“先提交数据再删除”、改名后的数据、修改过的旧数据或其他 ref 借来的豁免。缺远端对象或分叉时拒绝并要求 fetch/审阅。允许删除当前树中的历史误入文件，同时保留共享历史。

整理前唯一已知发布债务为 `ec1f327` 历史中的 114 字节状态 CSV。其摘要已并入现有研究状态、当前树移除；`--ref` 仍会如实报告历史对象，不能声称历史已清空。它不是凭据或原始实验数据，不为消除此记录改写历史。

## 4. 数据与交接

Git 只保存代码、必要文字、审阅配置。NPZ/CSV、原实验、视频、图片、日志、运行库、机器设置、patch/包留在忽略目录。结果通过现有 GitHub Release 发布，并匿名下载验证字节和 SHA；统一回执在 Issue #1。仅给服务器路径不算已上传。

清理前检查引用、活动作业和唯一副本。不得批量删除旧数值模块、原始输入、完整父史和失败证据。已不再执行的旧文档使用固定 Git 链接保留；当前任务、状态和角色入口原位更新，不累积 round/v2/final 副本。
