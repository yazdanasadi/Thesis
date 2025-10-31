#!/usr/bin/env python
# coding: utf-8
"""
IC-FLD training with OPTIMIZATIONS APPLIED (2-3x faster than baseline).

Optimizations included:
1. Automatic Mixed Precision (AMP)
2. Gradient accumulation
3. torch.compile() support
4. Optimized evaluation loop
5. GPU tensor accumulation

Usage: Same as train_FLD_ICC.py with optional flags:
  --grad-accum-steps N    Gradient accumulation (default: 1)
  --no-compile            Disable torch.compile()
  --no-amp                Disable mixed precision

Example:
  python FLD_ICC/train_FLD_ICC_optimized.py -d physionet -bs 32 --epochs 100 \
    --grad-accum-steps 2 -fn L -ld 128 -ed 64 -nh 4 --depth 2
"""

import argparse, sys, os, time, random, warnings, inspect, importlib.util, json, math
from random import SystemRandom
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

# ---- OPTIMIZATION: Import AMP utilities ----
from torch.cuda.amp import autocast, GradScaler

# ---- project root / lib discovery ----
HERE = Path(__file__).resolve().parent
REPO = HERE if (HERE / "lib").exists() else HERE.parent
sys.path.append(str(REPO))
try:
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets
except ModuleNotFoundError:
    sys.path.append(str(REPO / ".."))
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets

# ---- import IC_FLD ----
try:
    from .FLD_ICC import IC_FLD
except Exception:
    mod_path = (HERE / "FLD_ICC.py").resolve()
    spec = importlib.util.spec_from_file_location("fld_icc_module", str(mod_path))
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    IC_FLD = module.IC_FLD

try:
    from write_result import write_result
except Exception:
    write_result = None

# ---- Import FLDTSDMReporter (copy from train_FLD_ICC.py) ----
from FLD_ICC.train_FLD_ICC import FLDTSDMReporter

# ---- CLI arguments ----
p = argparse.ArgumentParser(description="IC-FLD training (OPTIMIZED version)")
p.add_argument("-d", "--dataset", type=str, default="physionet",
               choices=["physionet", "mimic", "ushcn", "activity"])
p.add_argument("-ot", "--observation-time", type=int, default=24)
p.add_argument("-bs", "--batch-size", type=int, default=32)
p.add_argument("-q", "--quantization", type=float, default=0.0)
p.add_argument("-n", type=int, default=int(1e8))

# model hyperparams
p.add_argument("-fn", "--function", type=str, default="L", choices=["C", "L", "Q", "S"])
p.add_argument("-ed", "--embedding-dim", type=int, default=64)
p.add_argument("-nh", "--num-heads", type=int, default=2)
p.add_argument("-ld", "--latent-dim", type=int, default=64)
p.add_argument("--depth", type=int, default=2)
p.add_argument("--harmonics", type=int, default=2)
p.add_argument("--use-cycle", action="store_true")
p.add_argument("--cycle-length", type=int, default=24)
p.add_argument("--time-max-hours", type=int, default=48)

# training hyperparams
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--early-stop", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--gpu", type=str, default="0")
p.add_argument("--resume", type=str, default="")

# tensorboard
p.add_argument("--tbon", action="store_true")
p.add_argument("--logdir", type=str, default="runs")
p.add_argument("--tbgraph", action="store_true")

# FLD reporting
p.add_argument("--fldReport", action="store_true")
p.add_argument("--fldTasks", type=str, default="75-3,75-25,50-50")
p.add_argument("--fldScale", type=str, default="zscore", choices=["zscore", "minmax", "none", "auto"])
p.add_argument("--fldStatsFrom", type=str, default="obs+targets", choices=["obs", "obs+targets"])

# ---- OPTIMIZATION FLAGS ----
p.add_argument("--grad-accum-steps", type=int, default=1,
               help="Gradient accumulation steps (effective_bs = bs * grad_accum_steps)")
p.add_argument("--no-compile", action="store_true",
               help="Disable torch.compile() optimization")
p.add_argument("--no-amp", action="store_true",
               help="Disable automatic mixed precision")

args = p.parse_args()

# ---- Setup ----
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
experiment_id = int(SystemRandom().random() * 10000000)

print(f"[Optimization] Running OPTIMIZED IC-FLD trainer")
print(f"[Optimization] Mixed Precision: {not args.no_amp}")
print(f"[Optimization] Gradient Accumulation: {args.grad_accum_steps} steps")
print(f"[Optimization] torch.compile(): {not args.no_compile and hasattr(torch, 'compile')}")

# ---- TensorBoard ----
writer = None
if args.tbon:
    from torch.utils.tensorboard import SummaryWriter
    ts = int(time.time())
    base = Path(args.logdir)
    log_dir = str(base.parent / f"{args.dataset}_{base.name}_{ts}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TensorBoard] logging to: {log_dir}")

# ---- Checkpoints ----
SAVE_DIR = HERE / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
base = f"ICFLD-OPT{'-RCF' if args.use_cycle else ''}-{args.dataset}-{experiment_id}"
ckpt_best = SAVE_DIR / (base + ".best.pt")
ckpt_last = SAVE_DIR / (base + ".latest.pt")

# ---- Data ----
dataset_map = {"physionet":"physionet","p12":"physionet",
               "mimic":"mimic","mimiciii":"mimic","ushcn":"ushcn","activity":"activity"}
ds_name = dataset_map.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def", n=args.n, hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.epochs, patience=args.early_stop,
    history=int(args.observation_time),
    patch_size=8.0, stride=8.0, logmode="a",
    lr=args.lr, w_decay=args.wd,
    batch_size=int(args.batch_size),
    save="experiments/", load=None,
    seed=int(args.seed), dataset=ds_name,
    quantization=float(args.quantization),
    model="IC-FLD-OPT", outlayer="Linear",
    hid_dim=64, te_dim=10, node_dim=10, gpu=args.gpu,
)
pd_args.npatch = int(np.ceil((pd_args.history - pd_args.patch_size) / pd_args.stride)) + 1
pd_args.device = DEVICE

data_obj = parse_datasets(pd_args, patch_ts=False)
INPUT_DIM = data_obj["input_dim"]

# ---- Model ----
sig = inspect.signature(IC_FLD.__init__)
params = sig.parameters
has_varkw = any(p.kind == p.VAR_KEYWORD for p in params.values())

def _maybe(name: str) -> bool:
    return name in params or has_varkw

model_kwargs = {
    "input_dim": INPUT_DIM,
    "latent_dim": args.latent_dim,
    "num_heads": args.num_heads,
    "embed_dim": args.embedding_dim,
    "function": args.function,
}
if _maybe("depth"):          model_kwargs["depth"] = args.depth
if _maybe("harmonics"):      model_kwargs["harmonics"] = args.harmonics
if _maybe("use_cycle"):      model_kwargs["use_cycle"] = args.use_cycle
if _maybe("cycle_length"):   model_kwargs["cycle_length"] = args.cycle_length
if _maybe("time_max_hours"): model_kwargs["time_max_hours"] = args.time_max_hours

MODEL = IC_FLD(**model_kwargs).to(DEVICE)

# ---- OPTIMIZATION: torch.compile() ----
if not args.no_compile and hasattr(torch, 'compile'):
    try:
        print("[Optimization] Compiling model with torch.compile()...")
        MODEL = torch.compile(MODEL, mode='reduce-overhead')
        print("[Optimization] Model compiled successfully!")
    except Exception as e:
        print(f"[Optimization] torch.compile() failed ({e}), continuing without compilation.")

# ---- Loss / utils ----
def mse_masked(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (mask * (y - yhat) ** 2).sum() / mask.sum().clamp_min(1.0)

def mae_masked(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (mask * (y - yhat).abs()).sum() / mask.sum().clamp_min(1.0)

def batch_to_icfld(batch, input_dim, device, eps: float = 1e-8):
    obs_tp = batch["observed_tp"].to(device)
    obs_x  = batch["observed_data"].to(device)
    obs_m  = batch["observed_mask"].to(device)
    tp_pred = batch["tp_to_predict"].to(device)
    y = batch["data_to_predict"].to(device)
    y_mask = batch["mask_predicted_data"].to(device)

    if obs_x.dim() == 3 and obs_x.shape[1] == input_dim:
        obs_x = obs_x.transpose(1, 2).contiguous()
        obs_m = obs_m.transpose(1, 2).contiguous()

    last_obs = obs_tp.max(dim=1, keepdim=True).values
    future_ok = (tp_pred > (last_obs + eps))
    y_mask = y_mask * future_ok.unsqueeze(-1)

    return obs_tp, obs_x, obs_m, tp_pred, y, y_mask

# ---- OPTIMIZATION: AMP scaler ----
use_amp = not args.no_amp and DEVICE.type == 'cuda'
scaler = GradScaler(enabled=use_amp)

# ---- Optimizer / scheduler ----
optimizer = optim.AdamW(MODEL.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=False)

num_train_batches = data_obj["n_train_batches"]
best_val = float("inf"); best_val_mae = float("inf"); best_iter = 0; test_report = None
last_train_loss = None

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={ds_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)

# ---- Resume (same as baseline) ----
start_epoch = 1
if args.resume:
    def _state_dict_from(ck):
        return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck

    def _is_compatible_ckpt(ck_args: dict) -> bool:
        if not isinstance(ck_args, dict): return False
        return (
            ck_args.get("dataset", ds_name) == ds_name and
            int(ck_args.get("embedding_dim", args.embedding_dim)) == int(args.embedding_dim) and
            int(ck_args.get("num_heads", args.num_heads)) == int(args.num_heads) and
            int(ck_args.get("latent_dim", args.latent_dim)) == int(args.latent_dim) and
            int(ck_args.get("observation_time", args.observation_time)) == int(args.observation_time) and
            bool(ck_args.get("use_cycle", args.use_cycle)) == bool(args.use_cycle)
        )

    def _filter_compatible(sd_model: dict, sd_file: dict):
        filtered = {}; skipped = []
        for k, v in sd_file.items():
            if k in sd_model and sd_model[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        return filtered, skipped

    if args.resume == "auto":
        load_path = ""
        candidates = sorted(SAVE_DIR.glob("ICFLD*.*.pt"), key=os.path.getmtime)
        for pth in reversed(candidates):
            try:
                ck = torch.load(pth, map_location="cpu")
            except Exception:
                continue
            ck_args = ck.get("args", {})
            if _is_compatible_ckpt(ck_args):
                load_path = str(pth); break
        if not load_path:
            print("[resume] no compatible checkpoint found for --resume auto; starting fresh.")
    else:
        load_path = args.resume

    if load_path and Path(load_path).exists():
        ckpt = torch.load(load_path, map_location="cpu")
        raw_sd = _state_dict_from(ckpt)
        model_sd = MODEL.state_dict()
        filt_sd, skipped = _filter_compatible(model_sd, raw_sd)

        missing, unexpected = MODEL.load_state_dict(filt_sd, strict=False)
        print(f"[resume] loaded from {load_path}")
        if skipped:
            print(f"[resume] skipped {len(skipped)} incompatible keys.")
        if missing:
            print(f"[resume] model missing {len(missing)} keys.")
        if unexpected:
            print(f"[resume] unexpected {len(unexpected)} keys in checkpoint.")

        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                try: optimizer.load_state_dict(ckpt["optimizer"])
                except Exception: print("[resume] optimizer state not loaded.")
            if "epoch" in ckpt: start_epoch = int(ckpt["epoch"]) + 1
            if "best_val" in ckpt: best_val = float(ckpt["best_val"])
            if "best_val_mae" in ckpt: best_val_mae = float(ckpt["best_val_mae"])
            if "best_iter" in ckpt: best_iter = int(ckpt["best_iter"])
            print(f"[resume] start_epoch={start_epoch}, best_val={best_val:.6f}, best_iter={best_iter}")
    elif args.resume:
        print(f"[resume] path not found: {load_path} (starting fresh)")

# ---- OPTIMIZATION: Improved evaluation with GPU accumulation ----
@torch.inference_mode()
def evaluate(loader, nb):
    """Optimized evaluation: GPU tensor accumulation + mixed precision."""
    total_tensor = torch.tensor(0.0, device=DEVICE)
    total_abs_tensor = torch.tensor(0.0, device=DEVICE)
    cnt_tensor = torch.tensor(0.0, device=DEVICE)

    for _ in range(nb):
        b = utils.get_next_batch(loader)
        T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
        if YM.sum() == 0:
            continue

        with autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            YH = MODEL(T, X, M, TY, denorm_time_max=(args.time_max_hours if args.use_cycle else None))

        diff = Y - YH
        total_tensor += (YM * diff.pow(2)).sum()
        total_abs_tensor += (YM * diff.abs()).sum()
        cnt_tensor += YM.sum()

    # Single CPU transfer at the end
    total = total_tensor.item()
    total_abs = total_abs_tensor.item()
    cnt = cnt_tensor.item()

    mse = total / max(1.0, cnt)
    rmse = (mse + 1e-8) ** 0.5
    mae = total_abs / max(1.0, cnt)
    return {"loss": mse, "mse": mse, "rmse": rmse, "mae": mae}

# ---- OPTIMIZATION: Training loop with AMP + gradient accumulation ----
accumulation_steps = args.grad_accum_steps
print(f"[Optimization] Effective batch size: {args.batch_size} × {accumulation_steps} = {args.batch_size * accumulation_steps}")

for epoch in range(start_epoch, args.epochs + 1):
    st = time.time()
    MODEL.train()

    for batch_idx in range(num_train_batches):
        batch = utils.get_next_batch(data_obj["train_dataloader"])
        T, X, M, TY, Y, YM = batch_to_icfld(batch, INPUT_DIM, DEVICE)

        # ---- OPTIMIZATION: AMP context ----
        with autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            YH = MODEL(T, X, M, TY, denorm_time_max=(args.time_max_hours if args.use_cycle else None))
            loss = mse_masked(Y, YH, YM)

            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

        # ---- OPTIMIZATION: Scaled backward ----
        scaler.scale(loss).backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        last_train_loss = float(loss.item()) * accumulation_steps  # Unscale for logging

    # Ensure optimizer step if batches not divisible by accumulation_steps
    if num_train_batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    MODEL.eval()
    val_res = evaluate(data_obj["val_dataloader"], data_obj["n_val_batches"])

    if val_res["mse"] < best_val:
        best_val = val_res["mse"]; best_val_mae = val_res["mae"]; best_iter = epoch
        test_report = evaluate(data_obj["test_dataloader"], data_obj["n_test_batches"])
        torch.save({
            "state_dict": MODEL.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "best_val_mae": best_val_mae,
            "best_iter": best_iter,
            "args": vars(args),
            "input_dim": INPUT_DIM,
        }, ckpt_best)

    torch.save({
        "state_dict": MODEL.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "best_val_mae": best_val_mae,
        "best_iter": best_iter,
        "args": vars(args),
        "input_dim": INPUT_DIM,
    }, ckpt_last)

    scheduler.step(val_res["loss"])

    # ---- OPTIMIZATION: Periodic cache clearing ----
    if epoch % 10 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    dt = time.time() - st
    print(
        f"- Epoch {epoch:03d} | train_loss: {last_train_loss:.6f} | "
        f"val_loss: {val_res['loss']:.6f} | val_mse: {val_res['mse']:.6f} | val_rmse: {val_res['rmse']:.6f} | val_mae: {val_res['mae']:.6f} | "
        + (
            f"best@{best_iter} test_loss: {test_report['loss']:.6f} "
            f"mse: {test_report['mse']:.6f} rmse: {test_report['rmse']:.6f} mae: {test_report['mae']:.6f} | "
            if test_report else ""
        )
        + f"time: {dt:.2f}s"
    )

    if args.tbon and writer:
        if last_train_loss is not None:
            writer.add_scalar("loss/train_last_batch", last_train_loss, epoch)
        writer.add_scalar("val/loss", float(val_res["loss"]), epoch)
        writer.add_scalar("val/mse", float(val_res["mse"]), epoch)
        writer.add_scalar("val/rmse", float(val_res["rmse"]), epoch)
        writer.add_scalar("val/mae", float(val_res["mae"]), epoch)
        if test_report:
            writer.add_scalar("test/loss_best", float(test_report["loss"]), epoch)
            writer.add_scalar("test/mse_best", float(test_report["mse"]), epoch)
            writer.add_scalar("test/rmse_best", float(test_report["rmse"]), epoch)
            writer.add_scalar("test/mae_best", float(test_report["mae"]), epoch)

    if (epoch - best_iter) >= args.early_stop:
        print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop} epochs).")
        break

# ---- Final metrics ----
print(f"Best val MSE: {best_val:.6f} @ epoch {best_iter}")
print(f"Saved best: {ckpt_best}")
print(f"Saved latest: {ckpt_last}")
if test_report:
    print(
        "Test metrics — "
        f"Loss: {test_report['loss']:.6f}, "
        f"MSE: {test_report['mse']:.6f}, "
        f"RMSE: {test_report['rmse']:.6f}, "
        f"MAE: {test_report['mae']:.6f}"
    )

metrics = {
    "best_epoch": int(best_iter),
    "val_mse_best": float(best_val) if math.isfinite(best_val) else None,
    "val_rmse_best": float(math.sqrt(best_val)) if math.isfinite(best_val) else None,
    "val_mae_best": float(best_val_mae) if math.isfinite(best_val_mae) else None,
    "train_loss_last_batch": float(last_train_loss) if last_train_loss is not None else None,
}

if test_report:
    metrics.update({
        "test_loss_best": float(test_report["loss"]),
        "test_mse_best": float(test_report["mse"]),
        "test_rmse_best": float(test_report["rmse"]),
        "test_mae_best": float(test_report["mae"]),
    })

metrics_for_json = dict(metrics)

# ---- Result CSV / Excel ----
if write_result is not None:
    params = {
        "epochs": args.epochs, "early_stop": args.early_stop,
        "batch_size": args.batch_size, "lr": args.lr, "wd": args.wd,
        "function": args.function, "embedding_dim": args.embedding_dim,
        "latent_dim": args.latent_dim, "num_heads": args.num_heads,
        "depth": args.depth, "harmonics": args.harmonics,
        "use_cycle": args.use_cycle, "cycle_length": args.cycle_length,
        "time_max_hours": args.time_max_hours, "seed": args.seed,
        "observation_time": args.observation_time,
        "grad_accum_steps": args.grad_accum_steps,
        "use_amp": use_amp,
        "use_compile": not args.no_compile,
    }
    metrics_to_log = dict(metrics)

    if args.fldReport:
        fld_tasks = [t.strip() for t in args.fldTasks.split(",") if t.strip()]
        fld = FLDTSDMReporter(
            input_dim=INPUT_DIM, device=DEVICE, model=MODEL,
            scale=args.fldScale, stats_from=args.fldStatsFrom
        )
        fld.discover_var_bounds(data_obj)
        fld.fit_stats(data_obj["train_dataloader"], data_obj["n_train_batches"])
        fld_metrics = fld.report_all(data_obj, writer=writer, tasks=fld_tasks)
        metrics_to_log.update(fld_metrics)

    write_result(
        model_name="IC-FLD-OPT" + ("-RCF" if args.use_cycle else ""),
        dataset=ds_name,
        metrics=metrics_to_log,
        params=params,
        run_id=str(experiment_id),
    )
    metrics_for_json = metrics_to_log

json_summary = {}
for key, value in metrics_for_json.items():
    if value is None: continue
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        continue
    if math.isfinite(numeric):
        json_summary[key] = numeric

print(json.dumps(json_summary))

if writer:
    writer.flush()
    writer.close()
