# Git 大文件误提交清理手册

适用场景：本地把大文件（如 `.pth`、模型权重、数据集）`git add`/`commit` 了，`git push` 到 GitHub 报错（常见是超过 `100MB`）。

---

## 1. 先判断你处于哪种情况

### 情况 A：只 `git add` 了，还没 `commit`

直接从暂存区移除即可（文件保留在本地磁盘）：

```bash
git restore --staged path/to/large_file.pth
```

如果是整类文件：

```bash
git restore --staged "*.pth"
```

然后继续正常提交并推送。

---

### 情况 B：已经 `commit`，但还没 `push`（最常见）

这是最推荐、最安全的处理方式：**重写本地未推送提交**。

#### Step 1) 回到远程基线（保留工作区改动）

```bash
git fetch origin
git reset --mixed origin/main
```

> 如果你的分支不是 `main`，把 `main` 换成当前目标分支。

#### Step 2) 忽略大文件（防止再次 add）

在对应项目的 `.gitignore` 增加规则，例如：

```gitignore
# RQ 产物模型
output/rq/**/*.pth
```

#### Step 3) 重新 add（这次不会再包含被忽略的大文件）

```bash
git add -A
git status
```

检查确认没有大文件后再提交：

```bash
git commit -m "remove large artifacts from commits"
git push origin main
```

---

### 情况 C：已经有多个本地提交都带了大文件（但还没 push）

同样用情况 B 的方式最省事：`reset --mixed origin/main` 后统一重新提交一次。

---

## 2. 如何确认“将要推送”的内容里没有大文件

先看相对远程的改动文件：

```bash
git diff --name-only origin/main..HEAD
```

再看其中 `.pth` 文件大小（示例脚本）：

```bash
python - <<'PY'
import os, subprocess
files = subprocess.check_output(
    ['git', 'diff', '--name-only', 'origin/main..HEAD'],
    text=True
).splitlines()
for f in files:
    if f.endswith('.pth') and os.path.exists(f):
        print(f"{f}\t{os.path.getsize(f)/1024/1024:.2f} MB")
PY
```

---

## 3. 你这次问题的最小命令模板

把 `output/rq` 下的大 `.pth` 去掉，但保留其他改动并推送：

```bash
git fetch origin
git reset --mixed origin/main
echo "output/rq/**/*.pth" >> MiniOneRec/.gitignore
git add -A
git commit -m "remove oversized rq checkpoints from push history"
git push origin main
```

---

## 4. 常见误区

- `git rm --cached` 只移除当前索引/提交，不会自动清理“之前提交历史”中的大文件。
- 只改 `.gitignore` 不会影响已经被跟踪的文件，需要先从索引或历史中移除。
- GitHub 报 `GH001` 时，通常是**历史里仍有超限对象**，不是当前目录有没有这个文件。

---

## 5. 如果“大文件已经 push 到远程”怎么办

这时需要改写远程历史（如 `git filter-repo` 或 BFG）并强推，影响协作分支。建议先和团队确认窗口再操作。

---

## 6. 团队协作版（推荐流程）

适用：仓库多人协作、有分支保护、CI 校验、PR 审核。

### 6.1 处理原则（先沟通再改历史）

- 不要直接在 `main` 上改历史，先开修复分支。
- 如果已推到远程，先在群里同步“需要改写历史”的影响范围。
- 明确冻结窗口：暂停他人向受影响分支提交，避免更多分叉。

建议沟通模板：

```text
[通知] 仓库误提交大文件，GitHub 拒绝推送/需要清理历史
影响分支：main（或 xxx）
处理窗口：今天 20:00-20:30（期间请勿 push）
处理方式：filter-repo 清理 *.pth + force push
后续动作：请所有同学按文档执行一次 fetch + reset/rebase
```

### 6.2 分支保护下的安全操作

如果仓库开启了分支保护（禁止 force push）：

1. 方案一（推荐）：临时放开管理员 force push 权限，仅在窗口内操作，完成后立即恢复保护规则。  
2. 方案二：新建“干净分支”并发 PR（仅适用于不需要改写 `main` 历史的场景）。  
3. 方案三：如果合规要求严格，走管理员审批 + 变更单（记录命令与时间点）。

### 6.3 已推远程后的标准清理流程（`git filter-repo`）

> 在镜像仓库或本地备份上先演练，再对正式仓库执行。

#### Step 0) 创建备份（强烈建议）

```bash
git clone --mirror git@github.com:ORG/REPO.git REPO.backup.git
```

#### Step 1) 在工作仓库执行历史清理

示例：删除所有历史中的 `.pth` 文件

```bash
git filter-repo --path-glob "*.pth" --invert-paths
```

如果只清理某目录：

```bash
git filter-repo --path "MiniOneRec/output/rq/" --invert-paths
```

#### Step 2) 增加忽略规则，防止复发

```gitignore
output/rq/**/*.pth
```

#### Step 3) 强推（在冻结窗口内）

```bash
git push origin --force --all
git push origin --force --tags
```

### 6.4 团队成员后续同步指令

历史改写后，所有协作者都要同步，否则会出现大量冲突或“幽灵提交”。

#### 对未提交本地改动的成员（最简）

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

#### 对有本地开发分支的成员

```bash
git fetch origin
git checkout your-feature-branch
git rebase --rebase-merges origin/main
```

如果 rebase 成本太高，可选择新建分支并 `cherry-pick` 需要的提交。

### 6.5 回滚预案（失败时如何快速恢复）

- 保留 `mirror backup`，可用于完整恢复仓库。
- 记录强推前的关键提交 SHA（`main`、`release`、`tags`）。
- 若清理误删关键文件：从备份仓库按路径 `checkout` 回来，重新提交。
- 若远程状态异常：先锁分支，使用备份仓库重新推送恢复。

### 6.6 推荐的“防再犯”基线

- 在仓库根目录补全 `.gitignore`（模型、日志、数据集、checkpoint）。
- 使用 pre-commit 钩子阻止超大文件进入提交，例如限制 `>50MB`。
- 在 CI 加一步检查（扫描新增文件大小和后缀）。
- 模型/数据使用对象存储或 Git LFS，不直接进 Git 历史。

---

## 7. 团队常用 `.gitignore` 示例（可按需合并）

```gitignore
# Python cache
__pycache__/
*.pyc

# Model artifacts
*.pth
*.pt
*.ckpt
*.bin

# Training outputs
output/
runs/
checkpoints/
wandb/
tensorboard/

# Dataset files (按项目实际保留白名单)
data/raw/
data/intermediate/
*.parquet
*.feather
*.h5
```
