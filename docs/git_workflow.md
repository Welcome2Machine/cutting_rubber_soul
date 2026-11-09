# Git 协作与仓库维护

更新：2026-11-09。当前研究任务见[唯一任务书](../researchtopic/research_prototype_task.md)。
本页规定协作方法，不另行复制参数、运行预算或科研验收。

## 1. 一个仓库，两端分工

仓库为 `Welcome2Machine/cutting_rubber_soul`。`projects/softmodeling` 与
`projects/rec_inc_soft_cut` 是普通子目录，不建立嵌套 Git 或 submodule。

| 维护者 | 工作分支 | 主要写入责任 |
|---|---|---|
| 服务器 agent | `codex/server-fracture-20261029` | 共同数值核心、物理配置接通、基线桥接与断裂实现；唯一数值集成者 |
| 本地 agent | `codex/intact-baseline-upgrade-20261031` | 已有参考/交接配置、实验和拟合依据、本地结果摘要 |
| 项目负责人 | 服务器分支上的小型文档提交 | 现行任务、两个短提示词、协作规则；遵守同样的正常前进规则 |

数值文件包括 FE 顶层数值 Python 文件、`prototype/`、`mechanics/`、`tools/`、`workbench/` 及共同适配器。
本地发现缺陷先提供最小复现/补丁建议；确需直接改共享文件时，双方先明确唯一临时编辑者与修改窗口，
服务器暂停该文件的并行编辑并负责最终审阅集成。未约定前本地只交建议，不双方默默修改核心。现有本地未提交成果保留并登记，不把它们视为可清理垃圾。

服务器的 `projects/softmodeling/docs/research_status.md` 由服务器更新实际结果；负责人只调整当前任务入口。
本地优先原位更新其已有 `docs/intact_baseline_upgrade_task_20261031.md` 和
`configs/intact_baseline_upgrade_20261031.json` 的交接段。不要在两分支各造一套共同任务书。

## 2. 开工与交接

先执行 `git status --short`、`git branch --show-current`、`git rev-parse HEAD`，检查暂存区、
活动作业及其实际导入的源码位置，核对 origin 的仓库身份，再 fetch。不在运行中的源码目录切分支、pull 或修改核心。
已有工作区脏时保留原状；需要新版代码则在隔离 worktree 中使用指定提交，不覆盖活动拟合。

两端均先从远端固定提交读取新任务；不假设单分支 clone 已有完整 tracking refs：

```bash
git fetch origin codex/server-fracture-20261029
task_sha=$(git rev-parse FETCH_HEAD)
git show "$task_sha":researchtopic/research_prototype_task.md
git show "$task_sha":docs/git_workflow.md
git show "$task_sha":AGENTS.md
git show "$task_sha":projects/softmodeling/AGENTS.md
```

fetch 不更新工作区文件。服务器确认当前分支正确、工作区干净且无作业使用这套源码后，
才能正常快进到已核对的 task_sha；否则从该提交建立隔离 worktree。本地只读任务无需合并代码。
后续 fetch 会改变 FETCH_HEAD，因此始终保留刚才固定的 task_sha。

交接给出：**分支、提交 SHA、用途、变更文件、验证、实际结果包位置/哈希**。服务器先看
`git show --stat <SHA>` 和具体差异/前置依赖，再选择独立且需要的提交，通常使用
`git cherry-pick -x <SHA>`。目前两分支已经分叉；不能为取得数据说明而整分支合并。
交接 SHA 在本机缺对象时，先 fetch 其来源分支并验证所交 SHA 可读；仍不可读就报告缺口，
不擅自改用该分支最新 HEAD，也不盲目挑不存在的对象。
本地修改某文件的提交可能依赖服务器没有的创建提交；纯文档/数据说明可直接读取固定 SHA 的文件，
将必要信息归入现有入口，不必导入整套旧任务。冲突按物理含义逐处处理，不统一选 ours/theirs。

服务器交付可复用核心时给出已验证 SHA、配置与命令；本地通过独立 worktree 消费它。
`main`、`feature/czm-cutting` 和早期整合分支不在本轮自动更新范围。
无需新增锁文件、任务注册器、协作机器人或每轮远程分支。

## 3. 提交与推送

一个提交只解决一个逻辑问题；数值实现、配置和必要验证可同提交，纯任务改动单独提交。
只暂存自己明确负责的路径，逐项检查 `git diff --cached`。脏工作区不使用 `git add .`、
`git add -A` 或 `git commit -am` 收走无关修改。已有他人暂存时，普通 git commit 也会带走它们；
应在独立 worktree 提交，原 index 不动，并核对整个暂存清单。不以 reset、clean、stash 他人修改换取空白状态。

按根 AGENTS 执行 `python -m pytest -q` 和 `python tools/check_publish.py --staged`；
涉及子项目数值变化时增加直接覆盖该风险的已有回归。不要为文档任务启动 FE，也不扩建一套验收框架。
提交前检查 `git diff --check`、暂存文件与数据排除；按子项目/钩子要求检查可达发布内容。

推送时显式指定自己负责的目标分支，仅正常推送。远端前进或推送被拒时，重新 fetch 并审查新提交，在干净的隔离工作区接上已审阅的改动；
不强推、不改写共享历史、不覆盖负责人或另一 agent 的提交。必要回退用 `git revert`。
工作完成时报告自己的提交、未提交文件归属和活动作业；“干净”不等于删除用户原有工作。

## 4. 代码、数据与检查点

Git 仅保存代码、文本说明、审阅配置和小型来源清单。原始实验、视频、权重、CSV/NPZ 状态、
图、日志、结果包、机器设置、运行库留在既有忽略目录；不以 `git add -f` 绕过规则。
数据另行备份/传输，以相对路径和 SHA 交接；未实收不得标记完成。Git LFS 未启用，clone 不含结果库。

刀形、材料、摩擦时间律、网格、裂面模型等改变后，从该设置的自身初态运行；不得移植旧内部历史。
仅文档改变导致 Git SHA 改变，不自动否定数值源码和输入身份相同的历史；沿用已有身份检查，
不改哈希规则绕过保护。记录实际解释器、导入路径、数值源码/输入与必要检查点，不复制冻结源码树。

## 5. 简洁是交付条件

- 当前任务只有一份，两个提示词只做入口；当前状态在既有状态页顶部原位更新，旧全文留 Git 历史。
- 修改现有入口和函数。只有明确的新物理职责才新增模块；说明替代关系，不再套一层继承、动态换类或包装 runner。
- 不新增 roundXX/v2/final/new 同职责文件、逐轮总报告、控制器/证书/同步框架。一次性诊断留忽略的运行目录。
- 清理先查引用、检查点来源集合和重放依赖；不批量删除旧核心、失败证据或用户数据。逻辑停用不等于物理删库。
- 提交说明交代研究用途、实际修改和验证。验收看物理问题是否得到回答，不按文件数、脚本数或运行数算进展。

## 6. 初始导入与钩子（历史/环境）

`codex/integrated-solution-blade-profile` 是早期整合起点，原始独立仓库与 bundle/patch 留在本地来源库；
它不再是本轮协作默认目标。不要再整目录覆盖项目副本或复制旧 `.git`。
新机器按需执行 `git config core.hooksPath .githooks`；pre-commit/pre-push 会核查发布内容，
但不能替代测试和审阅，也不等于已启用服务器分支保护。环境和原数据布局见[环境记录](environment.md)。
