#!/usr/bin/env python3
"""
merge_unmatched.py — 把 unmatched/ 里修正过的 PDF 合并回 journals_pdf/

策略:
  目标已存在，大小相同 → 删掉 unmatched 的副本
  目标已存在，大小不同 → journals_pdf 的加 _DUP 后缀，unmatched 的移过去
  目标不存在 → 直接移过去
"""

import os
import sys
import shutil

SRC = "./unmatched"
DST = "./journals_pdf"

moved = 0
skipped_dup = 0
conflict = 0

for root, dirs, files in os.walk(SRC):
    for fname in files:
        if not fname.lower().endswith(".pdf"):
            continue
        src_path = os.path.join(root, fname)

        # 目标路径: unmatched/Nature/xxx.pdf → journals_pdf/Nature/xxx.pdf
        rel_dir = os.path.relpath(root, SRC)
        if rel_dir == ".":
            rel_dir = ""
        dst_dir = os.path.join(DST, rel_dir)
        dst_path = os.path.join(dst_dir, fname)

        if os.path.exists(dst_path):
            src_size = os.path.getsize(src_path)
            dst_size = os.path.getsize(dst_path)
            if src_size == dst_size:
                os.remove(src_path)
                skipped_dup += 1
            else:
                # 大小不同, 保留 unmatched 版本, 旧的改名
                base, ext = os.path.splitext(fname)
                backup = os.path.join(dst_dir, f"{base}_OLD{ext}")
                os.rename(dst_path, backup)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src_path, dst_path)
                conflict += 1
                print(f"  ⚡ Conflict: {fname[:50]} (old→{base}_OLD)")
        else:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src_path, dst_path)
            moved += 1

# 清理 unmatched 下的空目录
for root, dirs, files in os.walk(SRC, topdown=False):
    for d in dirs:
        dpath = os.path.join(root, d)
        try:
            if not os.listdir(dpath):
                os.rmdir(dpath)
        except OSError:
            pass

# 如果 unmatched 根目录空了就删掉
try:
    if not os.listdir(SRC):
        os.rmdir(SRC)
except OSError:
    pass

print(f"✅ Merged: {moved}  |  Duplicates skipped: {skipped_dup}  |  Conflicts resolved: {conflict}")
