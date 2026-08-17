"""
PerovskiteGPT v1.5 — Gaussian 16 DFT 计算服务

在 perovskite-node 上提交 Gaussian 作业并监控。
"""
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

GAUSSIAN_ROOT = "/data1/gaussian"
GAUSSIAN_BIN = f"{GAUSSIAN_ROOT}/g16/g16"
GAUSSIAN_PROFILE = f"{GAUSSIAN_ROOT}/g16/bsd/g16.profile"
SCRATCH_DIR = Path("/data1/gaussian/scratch")
CALC_DIR = Path("/data1/gaussiancalc")
DEFAULT_CORES = 16
DEFAULT_MEM = "32GB"

# 正在运行的作业
_active_jobs: dict[str, dict] = {}


def _env():
    return f"export g16root={GAUSSIAN_ROOT} && source $g16root/g16/bsd/g16.profile"


def _generate_gjf(molecule_name: str, charge: int, multiplicity: int,
                  coordinates: str, method: str = "B3LYP",
                  basis: str = "6-31G(d)", job_type: str = "opt freq",
                  extra_keywords: str = "", nproc: int = DEFAULT_CORES,
                  mem: str = DEFAULT_MEM) -> str:
    """生成 Gaussian .gjf 输入文件内容。"""
    chk = SCRATCH_DIR / f"{molecule_name}_{uuid.uuid4().hex[:8]}.chk"
    header = f"%chk={chk}\n%nprocshared={nproc}\n%mem={mem}"
    route = f"#p {method}/{basis} {job_type} {extra_keywords}".strip()
    return (
        f"{header}\n"
        f"{route}\n\n"
        f"{molecule_name}\n\n"
        f"{charge} {multiplicity}\n"
        f"{coordinates}\n\n"
    )


def submit_job(molecule_name: str, charge: int, multiplicity: int,
               coordinates: str, method: str = "B3LYP",
               basis: str = "6-31G(d)", job_type: str = "opt freq",
               extra_keywords: str = "", nproc: int = DEFAULT_CORES,
               mem: str = DEFAULT_MEM) -> dict:
    """提交 Gaussian 计算作业。

    Returns:
        {"job_id": "...", "molecule": "...", "status": "submitted", "log_path": "..."}
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = CALC_DIR / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    gjf_content = _generate_gjf(
        molecule_name, charge, multiplicity, coordinates,
        method, basis, job_type, extra_keywords, nproc, mem
    )
    gjf_path = job_dir / f"{molecule_name}.gjf"
    log_path = job_dir / f"{molecule_name}.log"

    with open(gjf_path, "w") as f:
        f.write(gjf_content)

    cmd = (
        f"{_env()} && cd {job_dir} && "
        f"nohup g16 < {gjf_path} > {log_path} 2>&1 &"
    )
    subprocess.run(cmd, shell=True, executable="/bin/bash")

    job_info = {
        "job_id": job_id,
        "molecule": molecule_name,
        "status": "submitted",
        "gjf_path": str(gjf_path),
        "log_path": str(log_path),
        "method": method,
        "basis": basis,
        "job_type": job_type,
        "submitted_at": time.time(),
    }
    _active_jobs[job_id] = job_info
    return job_info


def check_job(job_id: str) -> dict:
    """检查 Gaussian 作业状态。

    Returns:
        {"job_id": "...", "status": "running|done|failed|not_found",
         "energy": None|float, "dipole": None|float, "termination": "..."}
    """
    if job_id not in _active_jobs:
        # 尝试从磁盘恢复
        job_dir = CALC_DIR / f"job_{job_id}"
        if not job_dir.exists():
            return {"job_id": job_id, "status": "not_found"}
        log_files = list(job_dir.glob("*.log"))
        if not log_files:
            return {"job_id": job_id, "status": "not_found"}
        log_path = log_files[0]
    else:
        log_path = Path(_active_jobs[job_id]["log_path"])

    if not log_path.exists():
        return {"job_id": job_id, "status": "submitted"}

    log_text = log_path.read_text(errors="replace")

    # 检查是否完成
    if "Normal termination" in log_text:
        status = "done"
    elif "Error termination" in log_text:
        status = "failed"
    else:
        # 检查进程是否还在运行
        result = subprocess.run(
            "ps aux | grep -v grep | grep g16 | wc -l",
            shell=True, capture_output=True, text=True
        )
        running = int(result.stdout.strip() or "0")
        status = "running" if running > 0 else "failed"

    # 提取能量
    energy = None
    scf_matches = re.findall(r'SCF Done:\s+E\(\w+\)\s+=\s+([-\d.]+)', log_text)
    if scf_matches:
        energy = float(scf_matches[-1])

    # 提取偶极矩
    dipole = None
    dipole_match = re.search(r'Tot=\s+([-\d.]+)', log_text)
    if dipole_match:
        dipole = float(dipole_match.group(1))

    # 提取计算时间
    wall_time = None
    time_match = re.search(r'Job cpu time:.*wall time.*', log_text)
    if time_match:
        wall_time = time_match.group(0).strip()

    result = {
        "job_id": job_id,
        "status": status,
        "energy_hartree": energy,
        "dipole_debye": dipole,
        "wall_time": wall_time,
    }

    if status == "done":
        result["termination"] = "normal"
    elif status == "failed" and "Error termination" in log_text:
        # 提取错误信息
        err_lines = [l for l in log_text.split("\n") if "Error termination" in l]
        result["error"] = err_lines[-1].strip() if err_lines else "Unknown error"

    return result


def list_jobs() -> list[dict]:
    """列出所有作业。"""
    jobs = []
    for job_id, info in _active_jobs.items():
        jobs.append({"job_id": job_id, "molecule": info["molecule"],
                      "status": info["status"]})
    return jobs
