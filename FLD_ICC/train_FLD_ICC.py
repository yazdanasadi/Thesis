#!/usr/bin/env python
# coding: utf-8
"""
IC-FLD training with t-PatchGNN preprocessing (no patches).
- Supports: -d / -ot / -bs / --epochs / --early-stop / --lr / --wd / --gpu / --resume / --tbon / --logdir / --tbgraph
- Model flags: -fn / -ed / -ld / -nh / --depth / --harmonics / --use-cycle / --cycle-length / --time-max-hours
"""

import argparse, sys, os, time, random, warnings, inspect, importlib.util, json, math, logging
from random import SystemRandom
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim

# ===== TIMING & LOGGING SETUP =====
SCRIPT_START_TIME = time.time()
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("IC-FLD TRAINING STARTED")
logger.info(f"Script: {__file__}")
logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)

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

# ---- import IC_FLD without triggering circulars ----
try:
    # When run as a module: python -m FLD_ICC.train_FLD_ICC
    from .FLD_ICC import IC_FLD
except Exception:
    # When run as a plain script: python FLD_ICC/train_FLD_ICC.py
    mod_path = (HERE / "FLD_ICC.py").resolve()
    spec = importlib.util.spec_from_file_location("fld_icc_module", str(mod_path))
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    IC_FLD = module.IC_FLD

# optional shared result writer
try:
    from write_result import write_result
except Exception:
    write_result = None

# ---------- FLD/TSDM Reporter with proper scaling ----------
import torch

class FLDTSDMReporter:
    """
    FLD paper-style TSDM evaluator with real scaling:
      - Discovers dataset min/max to denormalize batches back to RAW.
      - Fits per-channel train stats (mean/std, min/max) on RAW.
      - Scores 75-3 / 75-25 / 50-50 in requested scale (zscore|minmax|none|auto).
    """
    def __init__(self, input_dim: int, device: torch.device, model,
                 scale: str = "zscore", stats_from: str = "obs+targets"):
        self.D = input_dim
        self.device = device
        self.model = model
        self.scale = scale
        self.stats_from = stats_from  # "obs" or "obs+targets"

        # dataset-wide bounds to denormalize normalized batches -> raw
        self.var_min = None  # [D]
        self.var_max = None  # [D]

        # train statistics on raw scale (for z-score/minmax)
        self.train_mean = None  # [D]
        self.train_std  = None  # [D]
        self.train_min  = None  # [D]
        self.train_max  = None  # [D]

    # ----- helpers -----------------------------------------------------------------------------
    @staticmethod
    def _tsdm_spec(task: str):
        if task == "75-3":   return 0.75, 3, False
        if task == "75-25":  return 0.75, 25, False
        if task == "50-50":  return 0.50, None, True
        raise ValueError(f"Unknown TSDM task: {task}")

    def _ensure_feat_last(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {tuple(x.shape)}")
        if x.shape[-1] == self.D: return x
        if x.shape[1]  == self.D: return x.transpose(1, 2).contiguous()
        raise ValueError(f"Cannot orient tensor of shape {tuple(x.shape)} with D={self.D}")

    def _to_raw(self, z: torch.Tensor) -> torch.Tensor:
        """Denormalize from t-PatchGNN normalized scale back to RAW using dataset min/max."""
        if self.var_min is None or self.var_max is None:
            # Fallback: treat current values as raw if we couldn't discover bounds
            return z
        rng = (self.var_max - self.var_min).clamp_min(1e-12)
        return self.var_min + z * rng

    def _apply_scale(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Convert RAW -> requested report scale:
          - zscore: (x - mean) / std
          - minmax: (x - train_min) / (train_max - train_min)
          - none:   return raw
          - auto:   prefer zscore if mean/std are available else minmax else none
        """
        mode = self.scale
        if mode == "auto":
            if self.train_mean is not None and self.train_std is not None:
                mode = "zscore"
            elif self.train_min is not None and self.train_max is not None:
                mode = "minmax"
            else:
                mode = "none"

        if mode == "zscore":
            if self.train_mean is None or self.train_std is None:
                return raw  # graceful fallback
            std = self.train_std.clamp_min(1e-8)
            return (raw - self.train_mean) / std

        if mode == "minmax":
            if self.train_min is None or self.train_max is None:
                return raw
            rng = (self.train_max - self.train_min).clamp_min(1e-12)
            return (raw - self.train_min) / rng

        return raw  # "none"

    # ----- data plumbing -----------------------------------------------------------------------
    def discover_var_bounds(self, data_obj: dict):
        """Try to locate dataset-wide var_min/var_max to denormalize normalized data back to raw."""
        dev = self.device
        candidates = []

        # common places (dataset objects often store these)
        try: candidates.append(getattr(data_obj["train_dataloader"].dataset, "data_min", None))
        except: pass
        try: candidates.append(getattr(data_obj["train_dataloader"].dataset, "data_max", None))
        except: pass
        try: candidates.append(data_obj.get("data_min", None))
        except: pass
        try: candidates.append(data_obj.get("data_max", None))
        except: pass

        mins = [c for c in candidates if c is not None and hasattr(c, "shape")]
        # crude pairing if both present
        if len(mins) >= 2:
            a, b = mins[0], mins[1]
            if a.max() >= b.max():
                _min, _max = b, a
            else:
                _min, _max = a, b
            self.var_min = torch.as_tensor(_min, device=dev).view(-1)[:self.D]
            self.var_max = torch.as_tensor(_max, device=dev).view(-1)[:self.D]
        else:
            # explicit attributes on dataset
            ds = getattr(data_obj.get("train_dataloader", None), "dataset", None)
            if ds is not None:
                vmin = getattr(ds, "data_min", None)
                vmax = getattr(ds, "data_max", None)
                if vmin is not None and vmax is not None:
                    self.var_min = torch.as_tensor(vmin, device=dev).view(-1)[:self.D]
                    self.var_max = torch.as_tensor(vmax, device=dev).view(-1)[:self.D]

    @torch.no_grad()
    def fit_stats(self, train_loader, n_batches: int):
        """
        Fit channel-wise train statistics on RAW scale using masked data.
        By default uses observations+targets ("obs+targets") so evaluation matches FLD/TSDM.
        """
        dev = self.device
        sum_, sumsq_, cnt_ = (torch.zeros(self.D, device=dev),
                              torch.zeros(self.D, device=dev),
                              torch.zeros(self.D, device=dev))
        min_ = torch.full((self.D,), float("inf"), device=dev)
        max_ = torch.full((self.D,), float("-inf"), device=dev)

        for _ in range(n_batches):
            b = utils.get_next_batch(train_loader)
            Xn = self._ensure_feat_last(b["observed_data"].to(dev))      # normalized
            Mn = self._ensure_feat_last(b["observed_mask"].to(dev)).float()
            Yn  = self._ensure_feat_last(b["data_to_predict"].to(dev))
            YMn = self._ensure_feat_last(b["mask_predicted_data"].to(dev)).float()

            Xr = self._to_raw(Xn)
            Yr = self._to_raw(Yn)

            if self.stats_from == "obs":
                raw = Xr; m = Mn
            else:
                raw = torch.cat([Xr, Yr], dim=1)
                m   = torch.cat([Mn, YMn], dim=1)

            sum_   += (raw * m).sum(dim=(0, 1))
            sumsq_ += (raw.pow(2) * m).sum(dim=(0, 1))
            cnt_   += m.sum(dim=(0, 1))

            tmp = raw.clone()
            tmp[m == 0] = float("inf")
            min_ = torch.minimum(min_, tmp.amin(dim=(0, 1)))
            tmp = raw.clone()
            tmp[m == 0] = float("-inf")
            max_ = torch.maximum(max_, tmp.amax(dim=(0, 1)))

        cnt_safe = cnt_.clamp_min(1.0)
        mean = sum_ / cnt_safe
        var  = (sumsq_ / cnt_safe) - mean.pow(2)
        std  = var.clamp_min(0.0).sqrt().clamp_min(1e-8)

        self.train_mean = mean
        self.train_std  = std
        self.train_min  = min_
        self.train_max  = max_

    # ----- TSDM rebucketing (ragged-safe via masks) -------------------------------------------
    def rebucket_batch(self, b: dict, task: str):
        """
        Re-slice a batch to a TSDM task WITHOUT changing tensor shapes.
        Keep original shapes and gate with masks only.
        """
        ratio, K, forecast_all = self._tsdm_spec(task)

        Tobs = b["observed_tp"].to(self.device)                              # [B, Lo]
        Xn   = self._ensure_feat_last(b["observed_data"].to(self.device))    # [B, Lo, D]
        Mn   = self._ensure_feat_last(b["observed_mask"].to(self.device)).float()

        TY   = b["tp_to_predict"].to(self.device)                            # [B, Ly]
        Yn   = self._ensure_feat_last(b["data_to_predict"].to(self.device))  # [B, Ly, D]
        YM   = self._ensure_feat_last(b["mask_predicted_data"].to(self.device)).float()

        # cutoff over whole episode
        Tall  = torch.cat([Tobs, TY], dim=1)
        t0    = Tall.min(dim=1, keepdim=True).values
        t1    = Tall.max(dim=1, keepdim=True).values
        cutoff = t0 + ratio * (t1 - t0)

        # observations: keep <= cutoff (ensure at least one)
        keep_obs = (Tobs <= cutoff)                                          # [B, Lo]
        none_obs = ~keep_obs.any(dim=1)
        if none_obs.any():
            keep_obs[none_obs, 0] = True
        M2 = Mn * keep_obs.unsqueeze(-1).float()                             # [B, Lo, D]

        # targets: strictly > cutoff
        fut_mask = (TY > cutoff)                                             # [B, Ly]
        if forecast_all:
            sel_mask = fut_mask.clone()
        else:
            sel_mask = torch.zeros_like(fut_mask)
            for i in range(TY.size(0)):
                idx = torch.nonzero(fut_mask[i], as_tuple=False).flatten()
                if idx.numel() > 0:
                    sel_mask[i, idx[:K]] = True
        YM2 = YM * sel_mask.unsqueeze(-1).float()                            # [B, Ly, D]

        return Tobs, Xn, M2, TY, Yn, YM2   # still normalized; we’ll scale later

    @torch.no_grad()
    def evaluate_task(self, loader, nbatches: int, task: str):
        total, total_abs, cnt = 0.0, 0.0, 0.0
        for _ in range(nbatches):
            b = utils.get_next_batch(loader)
            T, Xn, M, TY, Yn, YM = self.rebucket_batch(b, task)
            if YM.sum() == 0:
                continue

            # model predicts on normalized scale:
            YHn = self.model(T, Xn, M, TY)

            # convert both Y and YH to RAW, then to requested report scale
            Yr  = self._to_raw(Yn)
            YHr = self._to_raw(YHn)

            Ys  = self._apply_scale(Yr)
            YHs = self._apply_scale(YHr)

            diff = Ys - YHs
            num = ((diff) ** 2 * YM).sum().item()
            abs_sum = (diff.abs() * YM).sum().item()
            den = YM.sum().item()
            total += num
            total_abs += abs_sum
            cnt   += den

        mse  = total / max(1.0, cnt)
        rmse = (mse + 1e-8) ** 0.5
        mae  = total_abs / max(1.0, cnt)
        return {"mse": mse, "rmse": rmse, "mae": mae, "targets_scored": int(cnt)}

    def report_all(self, data_obj: dict, writer=None, tasks=("75-3","75-25","50-50")):
        print("\n=== FLD-style TSDM report (Val/Test) ===")
        out = {}
        for task in tasks:
            v = self.evaluate_task(data_obj["val_dataloader"],  data_obj["n_val_batches"],  task)
            t = self.evaluate_task(data_obj["test_dataloader"], data_obj["n_test_batches"], task)
            print(f"  Task {task:>5} | "
                  f"val_mse={v['mse']:.6f} val_rmse={v['rmse']:.6f} val_mae={v['mae']:.6f} (targets={v['targets_scored']}) | "
                  f"test_mse={t['mse']:.6f} test_rmse={t['rmse']:.6f} test_mae={t['mae']:.6f} (targets={t['targets_scored']})")

            key = task.replace("-", "_")
            out[f"fld_val_mse_{key}"]   = float(v["mse"])
            out[f"fld_val_rmse_{key}"]  = float(v["rmse"])
            out[f"fld_val_mae_{key}"]   = float(v["mae"])
            out[f"fld_test_mse_{key}"]  = float(t["mse"])
            out[f"fld_test_rmse_{key}"] = float(t["rmse"])
            out[f"fld_test_mae_{key}"]  = float(t["mae"])

            if writer:
                writer.add_scalar(f"fld/val_mse_{key}",  v["mse"])
                writer.add_scalar(f"fld/val_rmse_{key}", v["rmse"])
                writer.add_scalar(f"fld/val_mae_{key}",  v["mae"])
                writer.add_scalar(f"fld/test_mse_{key}",  t["mse"])
                writer.add_scalar(f"fld/test_rmse_{key}", t["rmse"])
                writer.add_scalar(f"fld/test_mae_{key}",  t["mae"])
        return out
# ---------- end FLD/TSDM Reporter ----------

# ---------------- CLI ----------------
p = argparse.ArgumentParser(
    description="FLD-ICC training with t-PatchGNN preprocessing (no patches)"
)
p.add_argument("-d", "--dataset", type=str, default="physionet",
               choices=["physionet", "mimic", "ushcn", "activity"])
p.add_argument("-ot", "--observation-time", type=int, default=24,
               help="hours (ms for activity) as historical window")
p.add_argument("-bs", "--batch-size", type=int, default=32)
p.add_argument("-q", "--quantization", type=float, default=0.0)
p.add_argument("-n", type=int, default=int(1e8))

# model hyperparams
p.add_argument("-fn", "--function", type=str, default="L", choices=["C", "L", "Q", "S"])
p.add_argument("-ed", "--embedding-dim", type=int, default=64, help="total embedding dim")
p.add_argument("-nh", "--num-heads", type=int, default=2)
p.add_argument("-ld", "--latent-dim", type=int, default=64)
p.add_argument("--depth", type=int, default=2, help="decoder depth (layers)")
p.add_argument("--harmonics", type=int, default=2)

# residual cycle options
p.add_argument("--use-cycle", action="store_true")
p.add_argument("--cycle-length", type=int, default=24)
p.add_argument("--time-max-hours", type=int, default=48,
               help="un-normalized time max for cycle phases")

# training hyperparams
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--early-stop", type=int, default=10)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--gpu", type=str, default="0")
p.add_argument("--resume", type=str, default="", help='"" or "auto" or path to .pt')

# tensorboard
p.add_argument("--tbon", action="store_true")
p.add_argument("--logdir", type=str, default="runs")
p.add_argument("--tbgraph", action="store_true")

# --- PATCH: FLD/TSDM reporting flags ---
p.add_argument("--fldReport", action="store_true",
               help="Report FLD-style TSDM metrics (75-3, 75-25, 50-50) at the end.")
p.add_argument("--fldTasks", type=str, default="75-3,75-25,50-50",
               help='Comma-separated tasks from {"75-3","75-25","50-50"}.')
p.add_argument("--fldScale", type=str, default="zscore",
               choices=["zscore", "minmax", "none", "auto"],
               help="Scale used for FLD report: zscore (TSDM default), minmax, none, or auto.")
p.add_argument("--fldStatsFrom", type=str, default="obs+targets",
               choices=["obs", "obs+targets"],
               help="Which training values to estimate scaling from (observations only, or obs+targets).")

args = p.parse_args()

# ---------------- Setup ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
experiment_id = int(SystemRandom().random() * 10000000)

# ---- TensorBoard ----
writer = None
if args.tbon:
    from torch.utils.tensorboard import SummaryWriter
    ts = int(time.time())
    # example: mimic_runs_169....  (dataset + '_' + base + '_' + ts)
    base = Path(args.logdir)
    log_dir = str(base.parent / f"{args.dataset}_{base.name}_{ts}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TensorBoard] logging to: {log_dir}")

# checkpoints
SAVE_DIR = HERE / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
base = f"ICFLD{'-RCF' if args.use_cycle else ''}-{args.dataset}-{experiment_id}"
ckpt_best = SAVE_DIR / (base + ".best.pt")
ckpt_last = SAVE_DIR / (base + ".latest.pt")

# ---------------- Data via tPatchGNN preprocess (no patches) ----------------
dataset_map = {"physionet":"physionet","p12":"physionet",
               "mimic":"mimic","mimiciii":"mimic","ushcn":"ushcn","activity":"activity"}
ds_name = dataset_map.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def",
    n=args.n,
    hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.epochs, patience=args.early_stop,
    history=int(args.observation_time),
    patch_size=8.0, stride=8.0,
    logmode="a",
    lr=args.lr, w_decay=args.wd,
    batch_size=int(args.batch_size),
    save="experiments/", load=None,
    seed=int(args.seed), dataset=ds_name,
    quantization=float(args.quantization),
    model="IC-FLD", outlayer="Linear",
    hid_dim=64, te_dim=10, node_dim=10,
    gpu=args.gpu,
)
pd_args.npatch = int(np.ceil((pd_args.history - pd_args.patch_size) / pd_args.stride)) + 1
pd_args.device = DEVICE

logger.info("Loading dataset...")
dataset_load_start = time.time()
data_obj = parse_datasets(pd_args, patch_ts=False)
dataset_load_time = time.time() - dataset_load_start
INPUT_DIM = data_obj["input_dim"]
logger.info(f"Dataset loaded in {dataset_load_time:.2f} seconds")
logger.info(f"Dataset: {ds_name}, Input Dim: {INPUT_DIM}, History: {pd_args.history}")

# ---------------- Model ----------------
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

logger.info("Creating IC-FLD model...")
model_create_start = time.time()
MODEL = IC_FLD(**model_kwargs).to(DEVICE)
model_create_time = time.time() - model_create_start
logger.info(f"Model created in {model_create_time:.2f} seconds")
logger.info(f"Model: IC-FLD(input_dim={INPUT_DIM}, latent_dim={args.latent_dim}, function={args.function}, heads={args.num_heads}, depth={model_kwargs.get('depth', 'N/A')})")

# --- Enable patchify mode ---
MODEL.use_patchify = True          # activates time×channel patch aggregation
MODEL.num_patches = 12             # number of temporal windows (tune e.g., 6–12)
MODEL.patch_agg = "mean"           # "mean" or "sum"
if hasattr(MODEL, "_ensure_patch_pos"):
    MODEL._ensure_patch_pos(MODEL.num_patches)

if args.tbon and writer:
    writer.add_text("config/patchify_mode", "on")

# ---------------- Loss / utils ----------------
def mse_masked(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (mask * (y - yhat) ** 2).sum() / mask.sum().clamp_min(1.0)

def mae_masked(y: torch.Tensor, yhat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    return (mask * (y - yhat).abs()).sum() / mask.sum().clamp_min(1.0)

def batch_to_icfld(batch, input_dim, device, eps: float = 1e-8):
    obs_tp = batch["observed_tp"].to(device)        # [B, Lo]
    obs_x  = batch["observed_data"].to(device)      # [B, Lo, D] or [B, D, Lo]
    obs_m  = batch["observed_mask"].to(device)      # [B, Lo, D] or [B, D, Lo]
    tp_pred = batch["tp_to_predict"].to(device)     # [B, Ly]
    y = batch["data_to_predict"].to(device)         # [B, Ly, D]
    y_mask = batch["mask_predicted_data"].to(device)# [B, Ly, D]

    # Ensure [B, L, D]
    if obs_x.dim() == 3 and obs_x.shape[1] == input_dim:
        obs_x = obs_x.transpose(1, 2).contiguous()
        obs_m = obs_m.transpose(1, 2).contiguous()

    # --- Strict future-only mask (handles ties at the boundary) ---
    last_obs = obs_tp.max(dim=1, keepdim=True).values          # [B,1]
    future_ok = (tp_pred > (last_obs + eps))                   # [B, Ly]
    y_mask = y_mask * future_ok.unsqueeze(-1)                  # [B, Ly, D]

    return obs_tp, obs_x, obs_m, tp_pred, y, y_mask

# ---------------- Optimizer / scheduler ----------------
optimizer = optim.AdamW(MODEL.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=False)

num_train_batches = data_obj["n_train_batches"]
best_val = float("inf"); best_val_mae = float("inf"); best_iter = 0; test_report = None
last_train_loss = None

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={ds_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)

# ---------------- Resume++ ----------------
start_epoch = 1
if args.resume:
    def _state_dict_from(ck):
        return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck

    def _is_compatible_ckpt(ck_args: dict) -> bool:
        if not isinstance(ck_args, dict):
            return False
        return (
            ck_args.get("dataset", ds_name) == ds_name and
            int(ck_args.get("embedding_dim", args.embedding_dim)) == int(args.embedding_dim) and
            int(ck_args.get("num_heads", args.num_heads)) == int(args.num_heads) and
            int(ck_args.get("latent_dim", args.latent_dim)) == int(args.latent_dim) and
            int(ck_args.get("observation_time", args.observation_time)) == int(args.observation_time) and
            bool(ck_args.get("use_cycle", args.use_cycle)) == bool(args.use_cycle)
        )

    def _filter_compatible(sd_model: dict, sd_file: dict):
        filtered = {}
        skipped = []
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
            print(f"[resume] skipped {len(skipped)} incompatible keys (shape mismatch).")
        if missing:
            print(f"[resume] model missing {len(missing)} keys (left at init).")
        if unexpected:
            print(f"[resume] unexpected {len(unexpected)} keys in checkpoint.")

        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                try: optimizer.load_state_dict(ckpt["optimizer"])
                except Exception: print("[resume] optimizer state not loaded (param mismatch).")
            if "epoch" in ckpt: start_epoch = int(ckpt["epoch"]) + 1
            if "best_val" in ckpt: best_val = float(ckpt["best_val"])
            if "best_val_mae" in ckpt: best_val_mae = float(ckpt["best_val_mae"])
            if "best_iter" in ckpt: best_iter = int(ckpt["best_iter"])
            print(f"[resume] start_epoch={start_epoch}, best_val={best_val:.6f}, best_iter={best_iter}")
    elif args.resume:
        print(f"[resume] path not found: {load_path} (starting fresh)")

# ---------------- Eval helper ----------------
def evaluate(loader, nb):
    total = 0.0; total_abs = 0.0; cnt = 0.0
    for _ in range(nb):
        b = utils.get_next_batch(loader)
        T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
        with torch.no_grad():
            if YM.sum() == 0:
                print("[warn] empty-target batch (all TY <= last observed); skipping.")
                continue
            YH = MODEL(T, X, M, TY, denorm_time_max=(args.time_max_hours if args.use_cycle else None))
        diff = Y - YH
        total += float((YM * (diff) ** 2).sum().item())
        total_abs += float((YM * diff.abs()).sum().item())
        cnt   += float(YM.sum().item())
    mse = total / max(1.0, cnt)
    rmse = (mse + 1e-8) ** 0.5
    mae = total_abs / max(1.0, cnt)
    return {"loss": mse, "mse": mse, "rmse": rmse, "mae": mae}

# ---------------- Train loop ----------------
logger.info("="*80)
logger.info("TRAINING LOOP STARTED")
logger.info(f"Epochs: {args.epochs}, Early Stop: {args.early_stop}, Batch Size: {args.batch_size}")
logger.info(f"Learning Rate: {args.lr}, Weight Decay: {args.wd}")
logger.info("="*80)
training_start_time = time.time()
for epoch in range(start_epoch, args.epochs + 1):
    st = time.time()
    MODEL.train()
    for _ in range(num_train_batches):
        optimizer.zero_grad(set_to_none=True)
        batch = utils.get_next_batch(data_obj["train_dataloader"])
        T, X, M, TY, Y, YM = batch_to_icfld(batch, INPUT_DIM, DEVICE)
        YH = MODEL(T, X, M, TY, denorm_time_max=(args.time_max_hours if args.use_cycle else None))
        loss = mse_masked(Y, YH, YM)
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.item())

    MODEL.eval()
    val_res  = evaluate(data_obj["val_dataloader"], data_obj["n_val_batches"])

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

    dt = time.time() - st
    print(
        f"- Epoch {epoch:03d} | train_loss(one-batch): {loss.item():.6f} | "
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
        if args.tbgraph and epoch == 1:
            try:
                dummy_b = utils.get_next_batch(data_obj["train_dataloader"])
                T, X, M, TY, Y, YM = batch_to_icfld(dummy_b, INPUT_DIM, DEVICE)
                writer.add_graph(MODEL, (T, X, M, TY))
            except Exception as e:
                print(f"[tbgraph] skipped: {e}")
        if test_report:
            writer.add_scalar("test/loss_best", float(test_report["loss"]), epoch)
            writer.add_scalar("test/mse_best", float(test_report["mse"]), epoch)
            writer.add_scalar("test/rmse_best", float(test_report["rmse"]), epoch)
            writer.add_scalar("test/mae_best", float(test_report["mae"]), epoch)

    if (epoch - best_iter) >= args.early_stop:
        print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop} epochs).")
        logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {args.early_stop} epochs)")
        break

training_duration = time.time() - training_start_time
logger.info("="*80)
logger.info("TRAINING LOOP COMPLETED")
logger.info(f"Training duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
logger.info(f"Best epoch: {best_iter}, Best val MSE: {best_val:.6f}")
logger.info("="*80)

# If no train step ran (resume past max), compute last_train_loss for CSV
if last_train_loss is None:
    try:
        b = utils.get_next_batch(data_obj["train_dataloader"])
        T, X, M, TY, Y, YM = batch_to_icfld(b, INPUT_DIM, DEVICE)
        with torch.no_grad():
            YH = MODEL(T, X, M, TY, denorm_time_max=(args.time_max_hours if args.use_cycle else None))
        last_train_loss = float(mse_masked(Y, YH, YM).item())
    except Exception:
        last_train_loss = None

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

# ---------------- Result metrics ----------------
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

# ---------------- Result CSV / Excel ----------------
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
        "harmonics": args.harmonics,
        "use_cycle": args.use_cycle,
        "cycle_length": args.cycle_length,
        "time_max_hours": args.time_max_hours,
        "seed": args.seed,
        "observation_time": args.observation_time,
    }
    metrics_to_log = dict(metrics)

    # --- FLD report trigger (compute and merge into metrics, and print to console) ---
    if args.fldReport:
        fld_tasks = [t.strip() for t in args.fldTasks.split(",") if t.strip()]
        fld = FLDTSDMReporter(
            input_dim=INPUT_DIM, device=DEVICE, model=MODEL,
            scale=args.fldScale, stats_from=args.fldStatsFrom
        )
        fld.discover_var_bounds(data_obj)                      # find dataset min/max
        fld.fit_stats(data_obj["train_dataloader"], data_obj["n_train_batches"])  # fit train stats
        fld_metrics = fld.report_all(data_obj, writer=writer, tasks=fld_tasks)    # print + TB
        metrics_to_log.update(fld_metrics)                     # persist into CSV/Excel

    write_result(
        model_name="IC-FLD" + ("-RCF" if args.use_cycle else ""),
        dataset=ds_name,
        metrics=metrics_to_log,
        params=params,
        run_id=str(experiment_id),
    )
    metrics_for_json = metrics_to_log

json_summary = {}
for key, value in metrics_for_json.items():
    if value is None:
        continue
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

# ===== FINAL TIMING SUMMARY =====
script_total_time = time.time() - SCRIPT_START_TIME
logger.info("="*80)
logger.info("IC-FLD TRAINING COMPLETED SUCCESSFULLY")
logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Total script duration: {script_total_time:.2f} seconds ({script_total_time/60:.2f} minutes)")
logger.info(f"  - Dataset loading: {dataset_load_time:.2f}s")
logger.info(f"  - Model creation: {model_create_time:.2f}s")
logger.info(f"  - Training loop: {training_duration:.2f}s ({training_duration/60:.2f} min)")
logger.info(f"  - Other overhead: {(script_total_time - dataset_load_time - model_create_time - training_duration):.2f}s")
logger.info("="*80)
