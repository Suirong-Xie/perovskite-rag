# windows_PC — 论文 PDF 下载 & 整理工具集

本目录包含在 **Windows 校内机器** 上完成"下载论文 PDF → 归类 → 校验 → 修复"全流程的工具。

所有脚本需从**项目根目录**运行（因为路径 `journals_pdf/`、`missing_non_wiley.jsonl` 等都在根目录）：

```batch
cd D:\perovskite-rag
python pipeline\windows_PC\<脚本名>.py [参数]
```

---

## 📦 工具概览

| 工具 | 用途 | 什么时候用 |
|------|------|-----------|
| `batch_open.py` | **主力**: 分批打开 DOI → 手动点下载 → 自动归类 | 日常下载 |
| `fix_doi_mismatch.py` | 从 PDF **内容**提取真实 DOI，与文件名比对 | 下载完一批后做质检 |
| `audit_classification.py` | 检查所有 PDF 是否归入正确的期刊目录 | 整理完后验证 |
| `merge_unmatched.py` | 将 `unmatched/` 中修正过的 PDF 合并回 `journals_pdf/` | 手动修正后 |
| `journal_classifier.py` | 共享库: 期刊名 → 安全目录名 | 被其他脚本引用 |
| ~~`campus_download.py`~~ | ~~Playwright 全自动下载 (已被 `batch_open.py` 取代)~~ | 不再使用 |

---

## 🔧 详细说明

### 1. `batch_open.py` — 分批下载 + 自动归类（主力工具）

**设计思路**: 不依赖下载顺序。你只需在浏览器里点 PDF 下载按钮，脚本从**文件名中的 DOI** 自动匹配到正确的论文和期刊目录。

**工作流**:
```
循环: [清空临时目录] → [打开 20 个 DOI 标签] 
    → [你在浏览器中逐个点击 PDF 下载] 
    → [按 Enter] → [3 级匹配] → [归类到 journals_pdf/<期刊>/]
```

**3 级匹配策略**:
1. **批次内匹配**: PDF 文件名含 DOI → 匹配本批 20 篇
2. **全局匹配** (未命中时): 遍历全量论文索引 (18000+ 篇)，检查文件名
3. **内容提取** (仍未命中): 打开 PDF，从元数据/正文/末页提取真实 DOI → 匹配全量索引
4. **兜底**: 连内容都提不到 DOI → 移入 `unmatched/_unknown_batch/`

**用法**:
```batch
# 默认: 每次 20 篇, 按出版者分组
python pipeline\windows_PC\batch_open.py

# 每次 30 篇
python pipeline\windows_PC\batch_open.py --batch 30

# 只处理 ACS
python pipeline\windows_PC\batch_open.py --publisher acs

# 断点续传 (从上次中断处继续)
python pipeline\windows_PC\batch_open.py --resume
```

**依赖**: 无第三方库 (pypdf 可选，用于内容提取)；纯 Python 标准库 + webbrowser

**注意事项**:
- Edge 下载路径需设为 `D:\Edge浏览器下载\batch_tmp`
- Elsevier 下载的文件名是 `1-s2.0-Sxxx-main.pdf` 格式，不含 DOI，会触发内容提取或进入 unmatched
- `unmatched/` 中的文件可以后续用 `fix_doi_mismatch.py` 重新检查

---

### 2. `fix_doi_mismatch.py` — PDF 内容 DOI 校验

**用途**: 扫描 `journals_pdf/` 下所有 PDF，从 PDF **内容**中提取真实 DOI，与文件名中的 DOI 比对。不匹配的移到 `unmatched/`。

**什么时候用**: 下载了几批之后，做一次全面质检。特别是：
- 用 `batch_open.py` 下载的，文件名可能含错误 DOI
- Elsevier 类的文件（文件名不含 DOI）经过内容提取后归了类，想复查一遍

**工作流**:
```
扫描 journals_pdf/ 所有 PDF 
  → 从 PDF 元数据提取 DOI (dc:identifier, prism:doi...)
  → 从前 2 页正文提取 DOI
  → 从末页提取 DOI (Science 的 DOI 在末页)
  → 全页兜底提取
  → 与文件名 DOI 比对
  → 匹配 ✓ → 保留原位
  → 不匹配 ✗ → 用真实 DOI 重命名 → 移入 unmatched/<期刊>/
  → 无法提取 ❓ → unmatched/_no_doi_found/
```

**用法**:
```batch
# 仅检查, 不动文件
python pipeline\windows_PC\fix_doi_mismatch.py --dry-run

# 检查 + 整理
python pipeline\windows_PC\fix_doi_mismatch.py

# 指定 PDF 目录
python pipeline\windows_PC\fix_doi_mismatch.py --pdf-dir ./journals_pdf
```

**依赖**: `pypdf` (纯 Python, `pip install pypdf`)；可选 `pdftotext` (poppler)

---

### 3. `audit_classification.py` — 期刊归类审计

**用途**: 检查 `journals_pdf/` 下每个 PDF 是否归入了正确的期刊目录。可选自动整理。

**用法**:
```batch
# 仅检查 (打印报告)
python pipeline\windows_PC\audit_classification.py

# 检查 + 自动移动错放文件
python pipeline\windows_PC\audit_classification.py --fix

# 整理后清理空目录
python pipeline\windows_PC\audit_classification.py --fix --cleanup-empty-dirs

# 指定目录
python pipeline\windows_PC\audit_classification.py --pdf-dir ./journals_pdf
```

**典型场景**: 一批下载完成后，运行 `--fix` 把可能归类错误的 PDF 移到正确目录。

**依赖**: `journal_classifier.py` (同目录)

---

### 4. `merge_unmatched.py` — 合并修正后的文件

**用途**: 把 `unmatched/` 里的 PDF 合并回 `journals_pdf/`。

当你手动修正了 unmatched 里的文件（或 `fix_doi_mismatch.py` 已重新分类），运行此脚本把它们移回去。

**策略**:
- 目标已存在 & 大小相同 → 删除 unmatched 副本（重复）
- 目标已存在 & 大小不同 → 旧文件加 `_OLD` 后缀，unmatched 覆盖过去
- 目标不存在 → 直接移过去

**用法**:
```batch
python pipeline\windows_PC\merge_unmatched.py
```

---

### 5. `journal_classifier.py` — 共享分类器

**用途**: 期刊名 → 安全目录名的统一映射。被 `batch_open.py`、`fix_doi_mismatch.py`、`audit_classification.py` 等共用。

内含 200+ 期刊的精确映射表（Nature 家族、ACS、RSC、Wiley、Elsevier 等），支持：
- 精确匹配
- 缩写匹配 (`Nat. Energy` → `NatEnergy`)
- HTML 实体解码
- 无匹配时安全化期刊名

**用法**（在代码中）:
```python
from journal_classifier import classify_venue
dirname = classify_venue("ACS Energy Letters")  # → "ACS_Energy_Letters"
```

---

## 🗺️ 推荐工作流

### 日常下载
```
batch_open.py          ← 主力, 分批下载 + 自动归类
    ↓ (下载了 100+ 篇之后)
audit_classification.py --fix     ← 检查归类, 修正错放
```

### 质检修复
```
fix_doi_mismatch.py    ← 从内容提取真实 DOI 做全面比对
    ↓ (检查报告)
merge_unmatched.py     ← 把修正后的文件合并回去
audit_classification.py --fix --cleanup-empty-dirs  ← 最终整理
```

### 完整流程 (推荐每 500 篇执行一次)
```
1. batch_open.py → 日常下载 (多批)
2. fix_doi_mismatch.py --dry-run → 先看看有哪些问题
3. fix_doi_mismatch.py             → 修复
4. merge_unmatched.py              → 合并回来
5. audit_classification.py --fix --cleanup-empty-dirs  → 最终整理
```

---

## ⚠️ 注意事项

- **所有脚本需从项目根目录运行** (`D:\perovskite-rag`)
- `batch_open.py` 依赖 Edge 浏览器下载到特定目录 (`D:\Edge浏览器下载\batch_tmp`)
- `fix_doi_mismatch.py` 内容提取较慢 (每篇 1-3 秒)，建议后台运行
- `unmatched/` 目录中的文件是"待人工处理"的，不要直接删除
- `missing_non_wiley.jsonl` 是全量论文元数据，所有脚本都依赖它
