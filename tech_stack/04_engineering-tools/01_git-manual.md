# Git 常用指令手册

## 1. 概述（Overview）

> 待补充。

## 2. 基础与定义（Foundations）

> 待补充。

## 3. 核心原理（Core Principles）

> 待补充。

## 4. 实现与应用（Implementation & Applications）

### 一、Git 常用指令

#### 1. `git clone <仓库地址>`

用途：从远程仓库克隆项目到本地。

```bash
git clone https://github.com/xxx/project.git
```

---

#### 2. `git status`

用途：查看当前仓库状态，包括修改、暂存、未跟踪文件。

```bash
git status
```

---

#### 3. `git branch`

用途：查看本地分支。

```bash
git branch
```

---

#### 4. `git branch -r`

用途：查看远程分支。

```bash
git branch -r
```

---

#### 5. `git branch -a`

用途：查看所有分支，包括本地和远程。

```bash
git branch -a
```

---

#### 6. `git branch --show-current`

用途：查看当前所在分支。

```bash
git branch --show-current
```

---

#### 7. `git fetch origin`

用途：获取远程仓库最新提交和分支信息，但不修改当前代码。

```bash
git fetch origin
```

---

#### 8. `git fetch origin --prune`

用途：更新远程分支信息，并删除已经失效的远程分支引用。

```bash
git fetch origin --prune
```

---

#### 9. `git fetch --all --prune`

用途：更新所有远程仓库的信息。

```bash
git fetch --all --prune
```

---

#### 10. `git switch <分支名>`

用途：切换到已有本地分支。

```bash
git switch main
```

---

#### 11. `git switch -c <新分支名>`

用途：从当前代码创建并切换到新的本地分支。

```bash
git switch -c feat/test
```

---

#### 12. `git switch -c <新分支> origin/<远程分支>`

用途：基于某个远程分支创建自己的本地分支。

```bash
git switch -c feat/minco-hjr origin/feat/minco
```

---

#### 13. `git branch <新分支名>`

用途：创建新分支，但不切换过去。

```bash
git branch feat/test
```

---

#### 14. `git branch -d <分支名>`

用途：删除已经合并的本地分支。

```bash
git branch -d feat/test
```

---

#### 15. `git branch -D <分支名>`

用途：强制删除本地分支。

```bash
git branch -D feat/test
```

---

#### 16. `git add <文件>`

用途：把指定文件加入暂存区。

```bash
git add README.md
```

---

#### 17. `git add .`

用途：把当前目录下所有修改加入暂存区。

```bash
git add .
```

---

#### 18. `git restore <文件>`

用途：撤销尚未加入暂存区的文件修改。

```bash
git restore README.md
```

---

#### 19. `git restore --staged <文件>`

用途：把文件从暂存区移出，但保留文件修改。

```bash
git restore --staged README.md
```

---

#### 20. `git diff`

用途：查看尚未执行 `git add` 的代码修改。

```bash
git diff
```

---

#### 21. `git diff --staged`

用途：查看已经加入暂存区、准备提交的修改。

```bash
git diff --staged
```

---

#### 22. `git diff <分支1>..<分支2>`

用途：比较两个分支之间的代码差异。

```bash
git diff main..feat/minco-hjr
```

---

#### 23. `git diff --stat <分支1>..<分支2>`

用途：查看两个分支修改了哪些文件以及修改量。

```bash
git diff --stat origin/feat/minco..feat/minco-hjr
```

---

#### 24. `git commit -m "<提交说明>"`

用途：提交暂存区中的修改。

```bash
git commit -m "Add dynamic obstacle support"
```

---

#### 25. `git log`

用途：查看当前分支的完整提交历史。

```bash
git log
```

---

#### 26. `git log --oneline`

用途：简洁查看提交历史。

```bash
git log --oneline
```

---

#### 27. `git log <分支名> --oneline`

用途：查看指定分支的提交历史。

```bash
git log origin/feat/minco --oneline
```

---

#### 28. `git log --graph --decorate --oneline --all`

用途：查看所有分支的提交、分叉和合并关系。

```bash
git log --graph --decorate --oneline --all
```

---

#### 29. `git show <commit>`

用途：查看某一次提交具体修改了什么。

```bash
git show a1b2c3d
```

---

#### 30. `git pull`

用途：获取当前远程分支最新代码并合并到本地。

```bash
git pull
```

---

#### 31. `git pull --rebase`

用途：获取远程更新，并把自己的提交接到最新远程提交之后。

```bash
git pull --rebase
```

---

#### 32. `git push`

用途：把当前本地分支的提交推送到对应远程分支。

```bash
git push
```

---

#### 33. `git push -u origin <分支名>`

用途：第一次把本地新分支推送到远程，并建立跟踪关系。

```bash
git push -u origin feat/minco-hjr
```

之后只需要：

```bash
git push
```

---

#### 34. `git push origin --delete <分支名>`

用途：删除远程分支。

```bash
git push origin --delete feat/test
```

---

#### 35. `git remote -v`

用途：查看远程仓库名称和地址。

```bash
git remote -v
```

---

#### 36. `git remote show origin`

用途：查看 `origin` 的详细信息以及本地、远程分支跟踪关系。

```bash
git remote show origin
```

---

#### 37. `git stash`

用途：临时保存当前未提交修改。

```bash
git stash
```

---

#### 38. `git stash list`

用途：查看已经保存的 stash。

```bash
git stash list
```

---

#### 39. `git stash pop`

用途：恢复最近一次 stash，并删除该 stash。

```bash
git stash pop
```

---

#### 40. `git merge <分支名>`

用途：把指定分支合并到当前分支。

```bash
git merge feat/minco-hjr
```

---

#### 41. `git rebase <分支名>`

用途：把当前分支的提交重新接到另一个分支最新提交之后。

```bash
git rebase main
```

---

#### 42. `git reset --soft HEAD~1`

用途：撤销最近一次 commit，但保留修改，并保持在暂存区。

```bash
git reset --soft HEAD~1
```

---

#### 43. `git reset HEAD~1`

用途：撤销最近一次 commit，保留代码修改，但取消暂存。

```bash
git reset HEAD~1
```

---

#### 44. `git reset --hard HEAD~1`

用途：彻底删除最近一次 commit 和对应代码修改。

```bash
git reset --hard HEAD~1
```

> 注意：该命令会直接丢失代码修改。

---

#### 45. `git revert <commit>`

用途：通过创建新的 commit 撤销指定历史提交。

```bash
git revert a1b2c3d
```

适合撤销已经推送到远程的提交。

---

#### 46. `git tag`

用途：查看所有标签。

```bash
git tag
```

---

#### 47. `git tag <标签名>`

用途：给当前提交创建标签。

```bash
git tag v1.0
```

---

#### 48. `git push origin <标签名>`

用途：把标签上传到远程。

```bash
git push origin v1.0
```

---

### 二、从 Git Clone 到最终提交的完整工作流程

假设远程仓库为：

```text
https://github.com/xxx/YOPO-swarm.git
```

远程已有基础分支：

```text
origin/feat/minco
```

希望创建自己的开发分支：

```text
feat/minco-hjr
```

---

#### 第 1 步：克隆仓库

```bash
git clone https://github.com/xxx/YOPO-swarm.git
cd YOPO-swarm
```

---

#### 第 2 步：检查远程仓库

```bash
git remote -v
```

---

#### 第 3 步：同步远程分支

```bash
git fetch origin --prune
```

查看全部分支：

```bash
git branch -a
```

可能看到：

```text
* main
  remotes/origin/main
  remotes/origin/feat/minco
```

---

#### 第 4 步：基于远程分支创建自己的分支

```bash
git switch -c feat/minco-hjr origin/feat/minco
```

确认当前分支：

```bash
git branch --show-current
```

应该显示：

```text
feat/minco-hjr
```

---

#### 第 5 步：第一次上传自己的分支

```bash
git push -u origin feat/minco-hjr
```

此时分支关系：

```text
origin/feat/minco
        │
        │ 基于该分支创建
        ▼
feat/minco-hjr
        │
        │ git push
        ▼
origin/feat/minco-hjr
```

---

#### 第 6 步：进行代码开发

修改代码后检查状态：

```bash
git status
```

查看具体修改：

```bash
git diff
```

---

#### 第 7 步：加入暂存区

全部加入：

```bash
git add .
```

或者只加入指定文件：

```bash
git add src/example.py
```

再次确认：

```bash
git status
```

---

#### 第 8 步：提交修改

```bash
git commit -m "Add MINCO trajectory optimization"
```

查看提交：

```bash
git log --oneline
```

---

#### 第 9 步：推送到自己的远程分支

```bash
git push

或者

git push -u origin HEAD
```

---

### 三、日常开发推荐流程

每次开始开发：

```bash
git status
git pull --rebase
```

修改代码后：

```bash
git diff
git add .
git commit -m "修改说明"
git push
```

完整流程：

```text
检查仓库状态
    ↓
同步远程代码
    ↓
修改代码
    ↓
检查代码差异
    ↓
加入暂存区
    ↓
创建 Commit
    ↓
推送到远程
```

推荐日常使用：

```bash
git status

git pull --rebase

# 修改代码

git diff

git add .

git commit -m "Update xxx"

git push
```

---

## 5. 分析与总结（Analysis & Summary）

### 四、YOPO 项目推荐分支结构

```text
origin/feat/minco
        │
        │ 原始基础分支
        ▼
feat/minco-hjr
        │
        │ 本地开发
        ▼
origin/feat/minco-hjr
```

原则：

* `origin/feat/minco`：保留原始代码。
* `feat/minco-hjr`：自己的本地开发分支。
* `origin/feat/minco-hjr`：自己的远程开发分支。
* 日常修改尽量只在 `feat/minco-hjr` 上进行。

## 6. 关联与参考（Related Topics & References）

> 待补充。
