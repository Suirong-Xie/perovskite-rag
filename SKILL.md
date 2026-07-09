---
name: gaussian-calc
description: Run Gaussian 16 calculations on perovskite-node (10.28.0.147). Use when the user needs DFT/ab initio calculations (geometry optimization, single point energy, dipole moment, transition state search, NMR, etc.). Covers: job file generation, submission, monitoring, troubleshooting.
---

# Gaussian 16 — DFT Calculations

## Environment

```
Gaussian: /data1/gaussian/g16/g16
Scratch:  /data1/gaussian/scratch/
Setup:    export g16root=/data1/gaussian && source $g16root/g16/bsd/g16.profile
```

## Usage

### 1. Create .gjf input

Standard template for opt + dipole:

```
%chk=/data1/gaussian/scratch/molecule.chk
%nprocshared=32
%mem=64GB
#p B3LYP/6-31G(d) opt freq

Title

0 1
[coordinates in XYZ or internal format]

```

### 2. Submit

```bash
export g16root=/data1/gaussian
source $g16root/g16/bsd/g16.profile
cd /data1/gaussiancalc/sam_dipole/
nohup g16 < molecule.gjf > molecule.log 2>&1 &
```

### 3. Monitor

```bash
# Check if running
ps aux | grep g16

# Check .rwf file size (real progress indicator, more reliable than .log)
ls -lh /data1/gaussian/scratch/*.rwf

# Check log tail
tail -20 molecule.log

# Check termination
grep "Normal termination" molecule.log
```

### 4. Extract dipole moment

From .log file, grep for:

```
grep -A5 "Dipole moment" molecule.log
```

The `Tot=` line gives total dipole in Debye.

## Known Issues

- **def2SVP may segfault** on high-charge systems → switch to LANL2DZ or 6-31G(d)
- **scf=xqc** may cause infinite loop (l508) → remove if stuck
- **.log file has buffer delay** → check .rwf file size for real progress
- **Same-name .log conflicts** → always use unique job names per molecule
- **OSTE** (Open Shell Transition Energy) not needed for closed-shell SAM molecules
