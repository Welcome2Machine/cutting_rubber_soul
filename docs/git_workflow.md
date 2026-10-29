# Git 管理方法

## 单一总仓库

当前目录是整合 monorepo；协作起点为 `codex/integrated-solution-blade-profile`。
远端 `main` 仍是旧独立 FE 工程，不作为整合开发起点。`projects/softmodeling` 与
`projects/rec_inc_soft_cut` 是普通受控子目录，没有嵌套 `.git`，不使用 submodule。
这样一次提交能够原子修改两个接口与整合层，且保持一套可复现的版本。

初始导入保留当前源工作区，包含未提交修改与未跟踪源码。原 Git 历史保存在本地 `provenance/simulation-original.bundle`、`video-original.bundle`，未提交跟踪文件差异保存在相应 patch；未跟踪源码本身已纳入导入提交。bundle 不嵌入新仓库历史，避免带入旧数据与二进制。原项目仓库保留不动。

## 代码与数据分离

Git 管理源码、测试、方法文档、审阅后的配置、输入格式和小型文本来源清单。视频、权重、实验原始数据、FE 状态、报告、运行库、日志和机器设置留在本地，`.gitignore` 与发布检查双重约束。不要用 `git add -f` 绕过规则；新文件类型需要先更新明确的发布策略。

实验数据另做至少两份独立备份，保存原文件名与 SHA256 清单。结果通过唯一运行目录、输入哈希和 Git commit 关联；从 Git clone 只能恢复代码。Git LFS 并未启用，不能假设历史状态或视频已备份到远端。

## 日常分支与提交

```powershell
git switch codex/integrated-solution-blade-profile
git switch -c codex/point-observables
# 修改、检查diff、运行对应测试
python -m pytest -q
git add softcut_solution tests docs configs
python tools/check_publish.py --staged
git diff --cached --stat
git commit -m "feat(compare): add regional displacement observable"
```

分支采用 `feat/<主题>`、`fix/<问题>`、`research/<假设>`、`docs/<主题>`。一个提交解决一个逻辑问题；提交前缀采用 `feat/fix/refactor/test/docs/chore`，scope 用 `integration/video/fe/compare`。研究分支将参数、假设和判据提交，不提交算例二进制。生成结果的模型修改和界面修改尽量分开，以便判断数值变化来源。

`main` 只接收通过对应回归和发布检查的改动。单人本地可完成检查后 `git merge --ff-only <分支>`；多人协作通过 PR 审阅。合并接口改动时同步 schema 文档和跨项目契约测试。禁止强推 main，禁止修改已发布研究标签。

## 检查与钩子

本机已配置 `core.hooksPath=.githooks`。新 clone 需要执行一次：

```powershell
git config core.hooksPath .githooks
python tools/check_publish.py --staged
python tools/check_publish.py --ref HEAD
```

pre-commit 检查暂存文件；pre-push 根据 Git 传入的实际待推送分支或标签检查全部可达历史的 blob，拦截原数据、媒体、运行库、机器设置及大于 2 MiB 的文件。钩子不会替代测试，也不是服务器端保护；有意绕过钩子仍需团队审查。对子工程数值或视频算法改动，还要分别运行其相关回归。

## 发布、回退与上游同步

验收后创建语义版本标签，如 `v0.1.0`；研究参数版本另用 `study/<主题>-YYYYMMDD`。标签注释写清功能、验证和物理范围。用 `git revert <commit>` 撤销已共享改动；不以 reset 改写共享历史。

远端使用用户指定的 `https://github.com/Welcome2Machine/cutting_rubber_soul`。
该仓库的远端 `main` 目前仍是旧独立 FE 项目，与本地整合工程历史不同。
本次整合发布分支为 `codex/integrated-solution-blade-profile`，包含两个复制子工程与整合层；
不强推或覆盖远端 `main`、`feature/czm-cutting`，也不自动合并无关历史。
新协作者应显式 clone 这个分支，默认 clone main 不会获得整合工作台：

```powershell
git clone --branch codex/integrated-solution-blade-profile https://github.com/Welcome2Machine/cutting_rubber_soul.git
```

后续整合开发从此分支建立 `codex/<主题>` 分支；完成检查后通过审阅合入整合分支。
未来若迁移远端默认分支，应作为单独的明确仓库管理操作。源码 ZIP 可从 GitHub
该分支下载；原视频、权重、原始显微 PPT、运行结果与本地环境不在 ZIP 中。
所需环境与本地数据布局见 `docs/environment.md`。尚未配置服务器端分支保护，
本地钩子不能替代它。公开仓库不得提交机器设置和凭据。

原工程有新改动时，在独立 `chore/import-<日期>` 分支按文件 diff 审阅同步代码，并更新导入来源；不要再次整目录覆盖当前副本，也不要把旧 `.git` 复制进来。数据更新单独同步并记录来源哈希。原始历史可在另一个空目录 `git clone <bundle> <directory>` 恢复查看。
