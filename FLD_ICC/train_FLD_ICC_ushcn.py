#!/usr/bin/env python
# coding: utf-8
"""
IC-FLD (USHCN) trainer aligned with train_FLD.py CLI & preprocessing.
- Same flags: -d/-ot/-bs/-e/-es/-lr/-wd/-s/-fn/-ed/-nh/-dp/--gpu/--resume/--tbon/--logdir
- No cycle options here!!!!!!!!
- USHCN safety: NaN-safe masked MSE + ensure >=1 observed entry per sample. ><>>>><><><><><>
"""

import argparse, sys, os, time, random, warnings, inspect, glob
from random import SystemRandom
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# ---- project root / lib discovery ----
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent  # repo root (same as train_FLD.py)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets
except ModuleNotFoundError:
    sys.path.append(str(REPO_ROOT))
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets

# IC-FLD core
from FLD_ICC import IC_FLD

# optional shared CSV writer
try:
    from write_result import write_result
except Exception:
    write_result = None

# ---------------- CLI (mirrors train_FLD.py) ----------------
p = argparse.ArgumentParser(description="IC-FLD (USHCN) training with t-PatchGNN preprocessing (no patches)")
p.add_argument("-r", "--run_id", default=None, type=str)
p.add_argument("-e", "--epochs", default=300, type=int)
p.add_argument("-es", "--early-stop", default=30, type=int)
p.add_argument("-bs", "--batch-size", default=64, type=int)
p.add_argument("-lr", "--lr", default=1e-3, dest="lr", type=float, help="learning rate")
p.add_argument("-wd", "--wd", default=0.0, dest="wd", type=float, help="weight decay")
p.add_argument("-s", "--seed", default=0, type=int)

p.add_argument("-d", "--dataset", default="ushcn", type=str,
               help="physionet | mimic | ushcn | activity (use ushcn here)")
p.add_argument("-ot", "--observation-time", default=24, type=int, help="history window length")

# model hyperparams (keep parity with FLD flags)
p.add_argument("-fn", "--function", default="L", choices=("L", "S", "C", "Q"))
p.add_argument("-ed", "--embedding-dim", default=64, type=int)    # total embedding dim
p.add_argument("-nh", "--num-heads", default=2, type=int)
p.add_argument("-ld", "--latent-dim", default=64, type=int)
p.add_argument("-dp", "--depth", default=2, type=int)

p.add_argument("--gpu", default="0", type=str)
p.add_argument("--resume", default="", type=str, help="'auto' or path to a .pt checkpoint")

# TensorBoard
p.add_argument("--tbon", action="store_true", help="Enable TensorBoard logging")
p.add_argument("--logdir", type=str, default="runs", help="TensorBoard log root")
p.add_argument("--fldReport", action="store_true",
               help="Report FLD-style TSDM metrics (75-3, 75-25, 50-50) at the end.")
args = p.parse_args()
class FLDTSDMReporter:
    """
    FLD paper-style TSDM evaluator (75-3, 75-25, 50-50).
    Usage:
        rep = FLDTSDMReporter(input_dim=INPUT_DIM, device=DEVICE, model=MODEL)
        out = rep.report_all(data_obj, writer=writer)  # prints + returns dict of metrics
    """

    def __init__(self, input_dim: int, device: torch.device, model):
        self.D = input_dim
        self.device = device
        self.model = model

    # ----- internal helpers -----
    @staticmethod
    def _tsdm_spec(task: str):
        """Return (obs_ratio, K, forecast_all)."""
        if task == "75-3":   return 0.75, 3, False
        if task == "75-25":  return 0.75, 25, False
        if task == "50-50":  return 0.50, None, True   # forecast ALL remaining
        raise ValueError(f"Unknown TSDM task: {task}")

    def _ensure_feat_last(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")
        if x.shape[-1] == self.D: return x
        if x.shape[1]  == self.D: return x.transpose(1, 2).contiguous()
        raise ValueError(f"Cannot orient tensor of shape {tuple(x.shape)} with D={self.D}")

    def rebucket_batch(self, b: dict, task: str):
        """
        Re-slice a batch to match a TSDM task WITHOUT changing tensor shapes.
        We keep the original [B, L, D] / [B, Ly, D] shapes and only modify masks:
        - Observations: zero out M where time > cutoff (ensure at least 1 kept)
        - Targets: mark exactly K earliest future points (> cutoff), or ALL for 50-50
        """
        ratio, K, forecast_all = self._tsdm_spec(task)

        # Original tensors (shapes preserved)
        Tobs = b["observed_tp"].to(self.device)                              # [B, Lo]
        X    = self._ensure_feat_last(b["observed_data"].to(self.device))    # [B, Lo, D]
        M    = self._ensure_feat_last(b["observed_mask"].to(self.device))    # [B, Lo, D]

        TY   = b["tp_to_predict"].to(self.device)                            # [B, Ly]
        Y    = self._ensure_feat_last(b["data_to_predict"].to(self.device))  # [B, Ly, D]
        YM   = self._ensure_feat_last(b["mask_predicted_data"].to(self.device))  # [B, Ly, D]

        # Time cutoff over the whole episode span
        Tall   = torch.cat([Tobs, TY], dim=1)                                # [B, Lo+Ly]
        t0     = Tall.min(dim=1, keepdim=True).values
        t1     = Tall.max(dim=1, keepdim=True).values
        cutoff = t0 + ratio * (t1 - t0)                                      # [B,1]

        # ---- Observations: keep <= cutoff (ensure at least one) ----
        keep_obs = (Tobs <= cutoff)                                          # [B, Lo]
        no_obs   = ~keep_obs.any(dim=1)
        if no_obs.any():
            keep_obs[no_obs, 0] = True                                       # force first obs

        # Apply mask to observation mask M (don’t slice tensors)
        M2 = M * keep_obs.unsqueeze(-1).float()                               # [B, Lo, D]

        # ---- Targets: strictly future (> cutoff) ----
        fut_mask = (TY > cutoff)                                              # [B, Ly]
        if forecast_all:
            sel_mask = fut_mask.clone()
        else:
            sel_mask = torch.zeros_like(fut_mask)
            for i in range(TY.size(0)):
                idx = torch.nonzero(fut_mask[i], as_tuple=False).flatten()
                if idx.numel() > 0:
                    sel_mask[i, idx[:K]] = True                               # earliest K

        # Apply selection to target mask (don’t slice tensors)
        YM2 = YM * sel_mask.unsqueeze(-1).float()                              # [B, Ly, D]

        # Return original tensors with adjusted masks
        return Tobs, X, M2, TY, Y, YM2


    @torch.no_grad()
    def evaluate_task(self, loader, nbatches: int, task: str):
        """Compute masked MSE/RMSE for a TSDM task on a loader."""
        total, cnt = 0.0, 0.0
        for _ in range(nbatches):
            b = utils.get_next_batch(loader)
            T, X, M, TY, Y, YM = self.rebucket_batch(b, task)
            if YM.sum() == 0:
                continue
            YH = self.model(T, X, M, TY)
            total += float(((Y - YH) ** 2 * YM).sum().item())
            cnt   += float(YM.sum().item())
        mse  = total / max(1.0, cnt)
        rmse = (mse + 1e-8) ** 0.5
        return {"mse": mse, "rmse": rmse, "targets_scored": int(cnt)}

    def report_all(self, data_obj: dict, writer=None, tasks=("75-3","75-25","50-50")):
        """
        Evaluate on Val/Test for all tasks; print and return a flat dict merge
        into CSV writer (write_result).
        """
        print("\n=== FLD-style TSDM report (Val/Test) ===")
        out = {}
        for task in tasks:
            v = self.evaluate_task(data_obj["val_dataloader"],  data_obj["n_val_batches"],  task)
            t = self.evaluate_task(data_obj["test_dataloader"], data_obj["n_test_batches"], task)
            print(f"  Task {task:>5} | "
                  f"val_mse={v['mse']:.6f} val_rmse={v['rmse']:.6f} (targets={v['targets_scored']}) | "
                  f"test_mse={t['mse']:.6f} test_rmse={t['rmse']:.6f} (targets={t['targets_scored']})")

            key = task.replace("-", "_")
            out[f"fld_val_mse_{key}"]   = float(v["mse"])
            out[f"fld_val_rmse_{key}"]  = float(v["rmse"])
            out[f"fld_test_mse_{key}"]  = float(t["mse"])
            out[f"fld_test_rmse_{key}"] = float(t["rmse"])

            if writer:
                writer.add_scalar(f"fld/val_mse_{key}",  v["mse"])
                writer.add_scalar(f"fld/val_rmse_{key}", v["rmse"])
                writer.add_scalar(f"fld/test_mse_{key}",  t["mse"])
                writer.add_scalar(f"fld/test_rmse_{key}", t["rmse"])
        return out
# ---------- end FLD/TSDM Reporter ----------
# ---------------- Setup ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

experiment_id = int(SystemRandom().random() * 10000000)
SAVE_DIR = THIS_DIR / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
ckpt_best  = SAVE_DIR / f"ICFLD-ushcn-{experiment_id}.best.pt"
ckpt_last  = SAVE_DIR / f"ICFLD-ushcn-{experiment_id}.latest.pt"

# ---------------- Data via tPatchGNN (no patches) ----------------
dataset_map = {"physionet":"physionet","p12":"physionet",
               "mimic":"mimic","mimiciii":"mimic","ushcn":"ushcn","activity":"activity"}
dataset_name = dataset_map.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def",
    n=int(1e8),
    hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.epochs, patience=args.early_stop,
    history=int(args.observation_time),
    patch_size=8.0, stride=8.0,
    logmode="a",
    lr=float(args.lr), w_decay=float(args.wd),
    batch_size=int(args.batch_size),
    save="experiments/", load=None,
    seed=int(args.seed), dataset=dataset_name,
    quantization=0.0,
    model="IC-FLD", outlayer="Linear",
    hid_dim=64, te_dim=10, node_dim=10,
    gpu=args.gpu,
)
pd_args.npatch = int(np.ceil((pd_args.history - pd_args.patch_size) / pd_args.stride)) + 1
pd_args.device = DEVICE

data_obj = parse_datasets(pd_args, patch_ts=False)
INPUT_DIM = data_obj["input_dim"]
num_train_batches = data_obj["n_train_batches"]

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={dataset_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)

# ---------------- Model ----------------
# Build kwargs that IC_FLD actually accepts
sig = inspect.signature(IC_FLD.__init__)
accepts = set(sig.parameters.keys())

kw = dict(
    input_dim=INPUT_DIM,
    latent_dim=args.latent_dim,
    num_heads=args.num_heads,
    embed_dim=args.embedding_dim,  # IC_FLD expects total embedding dim
    function=args.function,
)
if "depth" in accepts:  kw["depth"] = args.depth

MODEL = IC_FLD(**kw).to(DEVICE)

# ---------------- Helpers ----------------
def _orient_time_last(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.dim() != 3: raise ValueError(f"Expected 3D tensor, got {x.shape}")
    if x.shape[-1] == input_dim: return x
    if x.shape[1] == input_dim:  return x.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot infer feature axis from {x.shape} with D={input_dim}")

def mse_masked(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NaN-robust masked MSE."""
    y    = torch.nan_to_num(y,    nan=0.0)
    yhat = torch.nan_to_num(yhat, nan=0.0)
    m = mask.float()
    num = (m * (y - yhat) ** 2).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def _ushcn_safe_context(T: torch.Tensor, X: torch.Tensor, M: torch.Tensor):
    """Ensure each sample has at least one observed entry (dummy (t=0, ch=0)=0)."""
    B, L, D = X.shape
    no_ctx = (M.view(B, -1).sum(dim=1) == 0)
    if no_ctx.any():
        idx = torch.nonzero(no_ctx).squeeze(-1)
        X[idx, 0, 0] = 0.0
        M[idx, 0, 0] = 1.0
        if L > 0: T[idx, 0] = 0.0
    return T, X, M

def batch_to_icfld(batch: dict, input_dim: int, device: torch.device):
    T  = batch["observed_tp"].to(device)                                   # [B,L]
    X  = _orient_time_last(batch["observed_data"].to(device), input_dim)   # [B,L,D]
    M  = _orient_time_last(batch["observed_mask"].to(device), input_dim)   # [B,L,D]
    TY = batch["tp_to_predict"].to(device)                                 # [B,Ty]
    Y  = _orient_time_last(batch["data_to_predict"].to(device), input_dim) # [B,Ty,D]
    YM = _orient_time_last(batch["mask_predicted_data"].to(device), input_dim) # [B,Ty,D]

    # sanitize NaNs and enforce at least one observed entry
    X  = torch.nan_to_num(X, nan=0.0); M = torch.where(torch.isnan(X), torch.zeros_like(M), M)
    T, X, M = _ushcn_safe_context(T, X, M)
    return T, X, M, TY, Y, YM

@torch.no_grad()
def evaluate(loader, nbatches):
    total = 0.0; cnt = 0.0
    for _ in range(nbatches):
        b = utils.get_next_batch(loader)
        T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
        YH = MODEL(T, X, M, TY)
        total += float(((Y - YH) ** 2 * YM).sum().item())
        cnt   += float(YM.sum().item())
    mse = total / max(1.0, cnt)
    rmse = (mse + 1e-8) ** 0.5
    return {"loss": mse, "mse": mse, "rmse": rmse}

# ---------------- Optim / sched / TB ----------------
optimizer = optim.AdamW(MODEL.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=False)

writer = None
if args.tbon:
    from torch.utils.tensorboard import SummaryWriter
    run_name = f"ICFLD_USHCN_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))
    try:
        b0 = utils.get_next_batch(data_obj["train_dataloader"])
        T0, X0, M0, TY0, _, _ = batch_to_icfld(b0, INPUT_DIM, DEVICE)
        writer.add_graph(MODEL, (T0, X0, M0, TY0))
    except Exception as e:
        print(f"[tbgraph] skipped: {e}")

# ---------------- Resume (optional) ----------------
start_epoch = 1
best_val = float("inf"); best_iter = 0; test_report = None
if args.resume:
    if args.resume == "auto":
        pattern = str(SAVE_DIR / ("ICFLD-ushcn-*.latest.pt"))
        ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
        load_path = ckpts[-1] if ckpts else ""
    else:
        load_path = args.resume
    if load_path and Path(load_path).exists():
        ckpt = torch.load(load_path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        MODEL.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                try: optimizer.load_state_dict(ckpt["optimizer"])
                except: pass
            if "epoch" in ckpt:     start_epoch = int(ckpt["epoch"]) + 1
            if "best_val" in ckpt:  best_val    = float(ckpt["best_val"])
            if "best_iter" in ckpt: best_iter   = int(ckpt["best_iter"])
        print(f"[resume] loaded {load_path} (start_epoch={start_epoch}, best_val={best_val:.6f})")

# ---------------- Train ----------------
print("Starting training…")
for epoch in range(start_epoch, args.epochs + 1):
    t0 = time.time()
    MODEL.train()
    last_train_loss = None

    for _ in range(num_train_batches):
        optimizer.zero_grad(set_to_none=True)
        b = utils.get_next_batch(data_obj["train_dataloader"])
        T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
        YH = MODEL(T, X, M, TY)
        loss = mse_masked(Y, YH, YM)
        loss.backward(); optimizer.step()
        last_train_loss = float(loss.item())

    MODEL.eval()
    val_res = evaluate(data_obj["val_dataloader"], data_obj["n_val_batches"])

    if val_res["mse"] < best_val:
        best_val = val_res["mse"]; best_iter = epoch
        test_report = evaluate(data_obj["test_dataloader"], data_obj["n_test_batches"])
        torch.save({
            "state_dict": MODEL.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "best_iter": best_iter,
            "args": vars(args),
            "input_dim": INPUT_DIM,
        }, ckpt_best)

    torch.save({
        "state_dict": MODEL.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "best_iter": best_iter,
        "args": vars(args),
        "input_dim": INPUT_DIM,
    }, ckpt_last)

    scheduler.step(val_res["loss"])

    dt = time.time() - t0
    print(f"- Epoch {epoch:03d} | train_loss(one-batch): {last_train_loss:.6f} | "
          f"val_mse: {val_res['mse']:.6f} | val_rmse: {val_res['rmse']:.6f} | "
          + (f"best@{best_iter} test_mse: {test_report['mse']:.6f} rmse: {test_report['rmse']:.6f} | " if test_report else "")
          + f"time: {dt:.2f}s")

    if writer:
        writer.add_scalar("train/loss_one_batch", last_train_loss, epoch)
        writer.add_scalar("val/mse",  float(val_res["mse"]),  epoch)
        writer.add_scalar("val/rmse", float(val_res["rmse"]), epoch)
        if test_report:
            writer.add_scalar("test/mse_best",  float(test_report["mse"]),  epoch)
            writer.add_scalar("test/rmse_best", float(test_report["rmse"]), epoch)

    if (epoch - best_iter) >= args.early_stop:
        print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop}).")
        break

print(f"Best val MSE: {best_val:.6f} @ epoch {best_iter}")
print(f"Saved best:   {ckpt_best}")
print(f"Saved latest: {ckpt_last}")

# ---- write shared results row (if available) ----
if write_result is not None:
    params = {
        "epochs": args.epochs,
        "early_stop": args.early_stop,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "function": args.function,
        "embedding_dim": args.embedding_dim,
        "latent_dim": args.latent_dim,
        "num_heads": args.num_heads,
        "depth": args.depth,
        "seed": args.seed,
        "observation_time": args.observation_time,
    }

    metrics = {
        "best_epoch": best_iter,
        "val_mse_best": float(best_val),
        "val_rmse_best": float((best_val + 1e-8) ** 0.5),
        "train_loss_last_batch": last_train_loss,
        "test_mse_best": (float(test_report["mse"]) if test_report else None),
        "test_rmse_best": (float(test_report["rmse"]) if test_report else None),
    }
    if args.fldReport:
        fld = FLDTSDMReporter(input_dim=INPUT_DIM, device=DEVICE, model=MODEL)
        fld_metrics = fld.report_all(data_obj, writer=writer)
        if write_result is not None:
            try:
                metrics.update(fld_metrics)
            except NameError:
                pass  
    write_result(
        model_name="IC-FLD",
        dataset=dataset_name,
        metrics=metrics,
        params=params,
        run_id=str(experiment_id),
    )

if writer:
    writer.flush(); writer.close()
