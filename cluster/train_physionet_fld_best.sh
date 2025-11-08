#!/bin/bash
#SBATCH --job-name=physionet_fld_full
#SBATCH --output=logs/physionet_fld_full_%j.out
#SBATCH --error=logs/physionet_fld_full_%j.err
#SBATCH --mail-user=asadi@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

set -euo pipefail

eval "$(conda shell.bash hook)"
export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-LP64}
conda activate thesis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  cd "${SCRIPT_DIR}"
fi

mkdir -p logs

REPO_ROOT=""
for candidate in "$(pwd)" "$(cd .. 2>/dev/null && pwd)" "$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd)"; do
  if [[ -n "${candidate}" ]] && [[ -d "${candidate}/FLD" ]] && [[ -d "${candidate}/FLD_ICC" ]]; then
    REPO_ROOT="${candidate}"
    break
  fi
done
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(pwd)"
fi

if [[ -d "${REPO_ROOT}/cluster" ]]; then
  CLUSTER_DIR="${REPO_ROOT}/cluster"
else
  CLUSTER_DIR="$(pwd)"
fi

JSON_PATH="${1:-${CLUSTER_DIR}/physionet_FLD.json}"
LOG_ROOT="${2:-${REPO_ROOT}/runs/physionet_fld_full}"

if [[ ! -f "${JSON_PATH}" ]]; then
  echo "[error] JSON file '${JSON_PATH}' not found." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

python3 - <<'PYCODE' "${JSON_PATH}" "${REPO_ROOT}" "${LOG_ROOT}"
import json
import shlex
import subprocess
import sys
from pathlib import Path

json_path = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()
log_root = Path(sys.argv[3]).resolve()

data = json.loads(json_path.read_text())
functions = data.get("functions") or {}
dataset = data.get("dataset", "physionet")

train_script = repo_root / "FLD" / "train_FLD.py"
if not train_script.exists():
    raise SystemExit(f"[error] Expected trainer at {train_script} not found.")

def maybe(params: dict, *keys, default=None):
    for key in keys:
        if key in params and params[key] is not None:
            return params[key]
    return default

def compute_embedding_dim(params: dict):
    val = maybe(params, "embed_dim", "embedding_dim")
    if val is not None:
        return val
    per_head = maybe(params, "embed_per_head")
    heads = maybe(params, "num_heads")
    if per_head is not None and heads is not None:
        try:
            return int(per_head) * int(heads)
        except Exception:
            return None
    return None

for fn_name, meta in sorted(functions.items()):
    if not isinstance(meta, dict):
        continue
    params = meta.get("params")
    if not isinstance(params, dict):
        print(f"[warn] Missing params for function '{fn_name}'. Skipping.")
        continue

    run_dataset = str(maybe(params, "dataset", default=dataset) or dataset)

    cmd = [
        sys.executable,
        str(train_script),
        "-d",
        run_dataset,
        "-fn",
        fn_name,
    ]

    def add_arg(flag: str, key, required: bool = False):
        keys = (key,) if isinstance(key, str) else key
        val = maybe(params, *keys)
        if val is None:
            if required:
                names = ", ".join(keys)
                raise ValueError(f"Missing required param(s) {names!r} for function '{fn_name}'.")
            return
        cmd.extend([flag, str(val)])

    embed_dim = compute_embedding_dim(params)
    if embed_dim is not None:
        cmd.extend(["-ed", str(embed_dim)])

    add_arg("-nh", "num_heads")
    add_arg("-dp", "depth")
    add_arg("-bs", "batch_size")
    add_arg("-e", "epochs")
    add_arg("-es", ("early_stop", "early_stop_patience"))
    add_arg("-lr", ("lr", "learn_rate"))
    add_arg("-wd", ("wd", "weight_decay"))
    add_arg("-s", "seed")
    add_arg("-ot", ("history", "observation_time"))

    logdir = log_root / f"{run_dataset}_{fn_name}"
    logdir.mkdir(parents=True, exist_ok=True)

    cmd.extend(["--tbon", "--logdir", str(logdir)])

    display_cmd = " ".join(shlex.quote(part) for part in cmd)
    print(f"[physionet_fld] {run_dataset}/{fn_name} :: {display_cmd}")
    sys.stdout.flush()

    subprocess.check_call(cmd, cwd=str(train_script.parent))

PYCODE
