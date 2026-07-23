"""run_gaussian + check_gaussian — DFT 计算工具。"""

from ...core.schemas import ToolCall, ToolResult
from ..gaussian_service import submit_job, check_job

# ── run_gaussian ──

GAUSSIAN_SCHEMA = {
    "name": "run_gaussian",
    "description": (
        "Submit a DFT calculation using Gaussian 16 on the local cluster "
        "(384 cores, 377GB RAM). Supports geometry optimization, single point "
        "energy, dipole moment, frequency analysis. Use for computing molecular "
        "properties of SAM molecules, perovskite fragments, or interface models."
    ),
    "parameters": {
        "molecule_name": "Short identifier (e.g., 'MeO-2PACz')",
        "charge": "Net charge (integer, e.g., 0 for neutral)",
        "multiplicity": "Spin multiplicity (1=singlet, 2=doublet, 3=triplet)",
        "coordinates": "XYZ format coordinates (element symbols + x y z per line)",
        "method": "DFT functional (default: B3LYP, options: PBE0, M06-2X, wB97XD)",
        "basis": "Basis set (default: 6-31G(d), options: 6-31+G(d,p), def2SVP, LANL2DZ)",
        "job_type": "Calculation type (default: 'opt freq', options: 'sp' for single point)",
    },
}


def execute_gaussian(arguments: dict) -> tuple:
    try:
        result = submit_job(
            molecule_name=arguments.get("molecule_name", "molecule"),
            charge=int(arguments.get("charge", 0)),
            multiplicity=int(arguments.get("multiplicity", 1)),
            coordinates=arguments.get("coordinates", ""),
            method=arguments.get("method", "B3LYP"),
            basis=arguments.get("basis", "6-31G(d)"),
            job_type=arguments.get("job_type", "opt freq"),
        )
        output = (
            f"Gaussian job submitted.\n"
            f"  Job ID: {result['job_id']}\n"
            f"  Molecule: {result['molecule']}\n"
            f"  Method: {result['method']}/{result['basis']}\n"
            f"  Type: {result['job_type']}\n"
            f"Use check_gaussian with this job_id to check status and get results."
        )
        return (ToolResult(ToolCall("run_gaussian", arguments), output), None)
    except Exception as e:
        return (ToolResult(ToolCall("run_gaussian", arguments), "", error=str(e)), None)


# ── check_gaussian ──

CHECK_SCHEMA = {
    "name": "check_gaussian",
    "description": (
        "Check the status of a submitted Gaussian calculation. Returns whether "
        "the job is running, done, or failed, along with extracted results "
        "(energy in Hartree, dipole moment in Debye) if available."
    ),
    "parameters": {
        "job_id": "The job ID returned by run_gaussian",
    },
}


def execute_check(arguments: dict) -> tuple:
    job_id = arguments.get("job_id", "")
    if not job_id:
        return (ToolResult(ToolCall("check_gaussian", arguments), "", error="job_id is required"), None)

    result = check_job(job_id)
    lines = [f"Job {job_id}: {result['status']}"]
    if result.get("energy_hartree"):
        lines.append(f"  Energy: {result['energy_hartree']:.6f} Hartree")
    if result.get("dipole_debye") is not None:
        lines.append(f"  Dipole: {result['dipole_debye']:.4f} Debye")
    if result.get("wall_time"):
        lines.append(f"  Time: {result['wall_time']}")
    if result.get("error"):
        lines.append(f"  Error: {result['error']}")

    return (ToolResult(ToolCall("check_gaussian", arguments), "\n".join(lines)), None)


SCHEMAS = [GAUSSIAN_SCHEMA, CHECK_SCHEMA]
EXECUTOR_MAP = {
    "run_gaussian": execute_gaussian,
    "check_gaussian": execute_check,
}
