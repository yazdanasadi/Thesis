#!/usr/bin/env bash
set -euo pipefail

BEST_PARAMS_PATH="${1:-best_hparams.json}"

if [[ ! -f "$BEST_PARAMS_PATH" ]]; then
  echo "[error] Best params file '$BEST_PARAMS_PATH' not found." >&2
  exit 1
fi

python3 - <<'PYCODE' "$BEST_PARAMS_PATH"
import json
import sys
from pathlib import Path

best_path = Path(sys.argv[1])
data = json.loads(best_path.read_text())

for dataset, functions in data.items():
    if not isinstance(functions, dict):
        continue
    for fn, entry in functions.items():
        if not entry or not isinstance(entry, dict):
            continue
        best_params = entry.get("best_params")
        trainer = entry.get("trainer", "icfld")
        if not isinstance(best_params, dict):
            print(f"[warn] Missing or malformed best_params for {dataset}/{fn}; skipping.")
            continue

        if dataset.lower() == "ushcn" and trainer == "icfld":
            script = Path("FLD_ICC/train_FLD_ICC_ushcn.py")
            cmd = [
                "python", script.name,
                "-d", dataset,
                "-fn", fn,
            ]
            latent_dim = best_params.get("latent_dim")
            embed_dim = best_params.get("embed_dim")
            if latent_dim is not None:
                cmd += ["--latent-dim", str(latent_dim)]
            if embed_dim is not None:
                cmd += ["--embedding-dim", str(embed_dim)]
        elif trainer == "fld":
            script = Path("FLD/train_FLD.py")
            cmd = [
                "python", script.name,
                "-d", dataset,
                "-fn", fn,
            ]
            embed_dim = best_params.get("embed_dim")
            if embed_dim is not None:
                cmd += ["-ed", str(embed_dim)]
        else:
            script = Path("FLD_ICC/train_FLD_ICC.py")
            cmd = [
                "python", script.name,
                "-d", dataset,
                "-fn", fn,
            ]
            latent_dim = best_params.get("latent_dim")
            embed_dim = best_params.get("embed_dim")
            if latent_dim is not None:
                cmd += ["--latent-dim", str(latent_dim)]
            if embed_dim is not None:
                cmd += ["--embedding-dim", str(embed_dim)]

        def add_arg(flag, key):
            val = best_params.get(key)
            if val is not None:
                cmd.extend([flag, str(val)])

        add_arg("--num-heads", "num_heads")
        add_arg("--depth", "depth")
        add_arg("--batch-size", "batch_size")
        add_arg("--epochs", "epochs")
        add_arg("--early-stop", "early_stop")
        if trainer == "fld":
            val = best_params.get("lr") or best_params.get("learn_rate")
            if val is not None:
                cmd.extend(["-lr", str(val)])
            val = best_params.get("wd") or best_params.get("weight_decay")
            if val is not None:
                cmd.extend(["-wd", str(val)])
        else:
            add_arg("--lr", "lr")
            add_arg("--lr", "learn_rate")
            add_arg("--wd", "wd")
            add_arg("--wd", "weight_decay")
        add_arg("--observation-time", "history")
        logdir = f"runs/{dataset}_{fn}_{trainer}"
        cmd.extend(["--tbon", "--logdir", logdir])

        print(f"[run] {dataset}/{fn} via {trainer}")
        print(" ".join(cmd))
        sys.stdout.flush()

        try:
            import subprocess
            script_dir = script.parent if 'script' in locals() else Path('.')
            subprocess.check_call(cmd, cwd=str(script_dir))
        except subprocess.CalledProcessError as exc:
            print(f"[error] Command failed for {dataset}/{fn}: {exc}")

PYCODE
