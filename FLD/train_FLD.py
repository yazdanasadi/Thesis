#!/usr/bin/env python
# coding: utf-8

import argparse, sys, os, time, random, warnings, logging
from random import SystemRandom
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime
from write_result import write_result

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ===== TIMING & LOGGING SETUP =====
SCRIPT_START_TIME = time.time()
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("FLD TRAINING STARTED")
logger.info(f"Script: {__file__}")
logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)

# ---- tPatchGNN libs ----
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets
    from lib import evaluation
except ModuleNotFoundError:
    sys.path.append(str(REPO_ROOT))
    import lib.utils as utils
    from lib.parse_datasets import parse_datasets
    from lib import evaluation

from FLD import FLD 

# --------- CLI ---------
parser = argparse.ArgumentParser(description="FLD training with t-PatchGNN preprocessing (no patches)")
parser.add_argument("-r", "--run_id", default=None, type=str)
parser.add_argument("-e", "--epochs", default=1000, type=int)
parser.add_argument("-es", "--early-stop", default=200, type=int)
parser.add_argument("-bs", "--batch-size", default=64, type=int)
parser.add_argument("-lr", "--learn-rate", default=1e-3, type=float)
parser.add_argument("-wd", "--weight-decay", default=0.0, type=float)
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-d", "--dataset", default="ushcn", type=str, help="physionet | mimic | ushcn | activity")
parser.add_argument("-ot", "--observation-time", default=24, type=int, help="history window length")
parser.add_argument("-fn", "--function", default="C", choices=("L", "S", "C", "Q"))
parser.add_argument("-ed", "--embedding-dim", default=4, type=int)
parser.add_argument("-nh", "--num-heads", default=2, type=int)
parser.add_argument("-dp", "--depth", default=1, type=int)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--resume", default="", type=str, help="'auto' or path to a .pt checkpoint")
parser.add_argument("--patience", type=int, default=10, help="patience for early stop")
# TensorBoard
parser.add_argument("--tbon", action="store_true", help="Enable TensorBoard logging")
parser.add_argument("--logdir", type=str, default="runs", help="TensorBoard log root")
args = parser.parse_args()


def _flag_in_argv(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv)


if not _flag_in_argv("--patience"):
    args.patience = args.early_stop
else:
    args.early_stop = args.patience

# --------- Setup ---------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

experiment_id = int(SystemRandom().random() * 10000000)
Path("saved_models").mkdir(parents=True, exist_ok=True)
model_best_path = f"saved_models/FLD-{args.function}_{args.dataset}_{experiment_id}.best.pt"
model_latest_path = f"saved_models/FLD-{args.function}_{args.dataset}_{experiment_id}.latest.pt"

# --------- Build loaders via tPatchGNN (no patches) ---------
dataset_map = {"physionet":"physionet","p12":"physionet","mimic":"mimic","mimiciii":"mimic","ushcn":"ushcn","activity":"activity"}
dataset_name = dataset_map.get(args.dataset.lower(), args.dataset)

pd_args = SimpleNamespace(
    state="def", n=int(1e8), hop=1, nhead=1, tf_layer=1, nlayer=1,
    epoch=args.epochs, patience=args.patience, history=int(args.observation_time),
    patch_size=8.0, stride=8.0, logmode="a",
    lr=args.learn_rate,
    w_decay=args.weight_decay, batch_size=int(args.batch_size),
    save="experiments/", load=None, seed=int(args.seed),
    dataset=dataset_name, quantization=0.0, model="FLD",
    outlayer="Linear", hid_dim=64, te_dim=10, node_dim=10, gpu=args.gpu,
)
pd_args.npatch = int(np.ceil((pd_args.history - pd_args.patch_size) / pd_args.stride)) + 1
pd_args.device = DEVICE

logger.info("Loading dataset...")
dataset_load_start = time.time()
data_obj = parse_datasets(pd_args, patch_ts=False)
dataset_load_time = time.time() - dataset_load_start
INPUT_DIM = data_obj["input_dim"]
num_train_batches = data_obj["n_train_batches"]
logger.info(f"Dataset loaded in {dataset_load_time:.2f} seconds")

print("PID, device:", os.getpid(), DEVICE)
print(f"Dataset={dataset_name}, INPUT_DIM={INPUT_DIM}, history={pd_args.history}")
print("n_train_batches:", num_train_batches)
logger.info(f"Dataset: {dataset_name}, Input Dim: {INPUT_DIM}, History: {pd_args.history}, Train Batches: {num_train_batches}")

# --------- Build model ---------
logger.info("Creating FLD model...")
model_create_start = time.time()
MODEL = FLD(
    input_dim=INPUT_DIM,
    latent_dim=20,
    embed_dim_per_head=args.embedding_dim,
    num_heads=args.num_heads,
    function=args.function,
    depth=args.depth,
    device=DEVICE,
).to(DEVICE)
model_create_time = time.time() - model_create_start
logger.info(f"Model created in {model_create_time:.2f} seconds")
logger.info(f"Model: FLD(input_dim={INPUT_DIM}, latent_dim=20, function={args.function}, heads={args.num_heads}, depth={args.depth})")

# --------- Helper to orient data for graph logging ---------
def _orient_time_last(x: torch.Tensor, input_dim: int) -> torch.Tensor:
    if x.dim() != 3: raise ValueError(f"Expected 3D tensor, got {x.shape}")
    if x.shape[-1] == input_dim: return x
    if x.shape[1] == input_dim:  return x.transpose(1, 2).contiguous()
    raise ValueError(f"Cannot infer feature axis from {x.shape} with D={input_dim}")
# --------- Optim / sched ---------
optimizer = optim.AdamW(MODEL.parameters(),
                        lr=args.learn_rate if hasattr(args,"learn_rate") else args.__dict__.get("learn-rate",1e-3),
                        weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=True)

best_val = float("inf")
best_val_mae = float("inf")
best_val_mape = float("inf")
best_iter = 0
test_report = None
last_train_loss = None
last_train_metrics = None

# --------- TensorBoard (optional) ---------
writer = None
if args.tbon:
    run_name = f"FLD_{dataset_name}_{int(time.time())}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))
    try:
        b = utils.get_next_batch(data_obj["train_dataloader"])
        T  = b["observed_tp"].to(DEVICE)
        X  = _orient_time_last(b["observed_data"].to(DEVICE), INPUT_DIM)
        M  = _orient_time_last(b["observed_mask"].to(DEVICE), INPUT_DIM)
        TY = b["tp_to_predict"].to(DEVICE)
        writer.add_graph(MODEL, (T, X, M, TY))
    except Exception:
        pass

# --------- Train ---------
logger.info("="*80)
logger.info("TRAINING LOOP STARTED")
logger.info(f"Epochs: {args.epochs}, Patience: {args.patience}, Batch Size: {args.batch_size}")
logger.info(f"Learning Rate: {args.learn_rate}, Weight Decay: {args.weight_decay}")
logger.info("="*80)
training_start_time = time.time()
try:
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        MODEL.train()
        for _ in range(num_train_batches):
            optimizer.zero_grad(set_to_none=True)
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
            train_res = evaluation.compute_all_losses(MODEL, batch_dict)
            loss = train_res["loss"]
            loss.backward()
            optimizer.step()
            last_train_loss = float(loss.item())
            last_train_metrics = train_res

        MODEL.eval()
        with torch.no_grad():
            val_res = evaluation.evaluation(MODEL, data_obj["val_dataloader"], data_obj["n_val_batches"])
            if val_res["mse"] < best_val:
                best_val = val_res["mse"]
                best_val_mae = val_res["mae"]
                best_val_mape = val_res["mape"]
                best_iter = epoch
                test_report = evaluation.evaluation(MODEL, data_obj["test_dataloader"], data_obj["n_test_batches"])
                torch.save({
                    "state_dict": MODEL.state_dict(),
                    "args": vars(args),
                    "input_dim": INPUT_DIM,
                    "best_val": best_val,
                    "best_val_mae": best_val_mae,
                    "best_val_mape": best_val_mape,
                    "best_iter": best_iter,
                }, model_best_path)

        scheduler.step(val_res["loss"])

        dt = time.time() - t0
        val_mape_pct = val_res["mape"] * 100.0
        train_loss_display = last_train_loss if last_train_loss is not None else float("nan")
        logger.info(f"- Epoch {epoch:03d} | ExpID {experiment_id}")
        logger.info(f"Train - Loss (one batch): {train_loss_display:.5f}")
        logger.info(
            "Val - Loss, MSE, RMSE, MAE, MAPE: "
            f"{val_res['loss']:.5f}, {val_res['mse']:.5f}, {val_res['rmse']:.5f}, {val_res['mae']:.5f}, {val_mape_pct:.2f}%"
        )
        if test_report:
            logger.info(
                "Test - Best epoch, Loss, MSE, RMSE, MAE, MAPE: "
                f"{best_iter}, {test_report['loss']:.5f}, {test_report['mse']:.5f}, "
                f"{test_report['rmse']:.5f}, {test_report['mae']:.5f}, {test_report['mape']*100.0:.2f}%"
            )
        logger.info(f"Time spent: {dt:.2f}s")

        print(
            f"- Epoch {epoch:03d} | train_loss(one-batch): {train_loss_display:.6f} | "
            f"val_loss: {val_res['loss']:.6f} | val_mse: {val_res['mse']:.6f} | "
            f"val_rmse: {val_res['rmse']:.6f} | val_mae: {val_res['mae']:.6f} | val_mape: {val_mape_pct:.2f}% | "
            + (
                f"best@{best_iter} test_loss: {test_report['loss']:.6f} "
                f"mse: {test_report['mse']:.6f} rmse: {test_report['rmse']:.6f} "
                f"mae: {test_report['mae']:.6f} mape: {test_report['mape']*100.0:.2f}% | "
                if test_report else ""
            )
            + f"time: {dt:.2f}s"
        )

        if writer:
            if last_train_loss is not None:
                writer.add_scalar("train/loss_one_batch", last_train_loss, epoch)
            if last_train_metrics is not None:
                writer.add_scalar("train/mse_one_batch", last_train_metrics["mse"], epoch)
                writer.add_scalar("train/rmse_one_batch", last_train_metrics["rmse"], epoch)
                writer.add_scalar("train/mae_one_batch", last_train_metrics["mae"], epoch)
            writer.add_scalar("val/loss",  float(val_res["loss"]),  epoch)
            writer.add_scalar("val/mse",  float(val_res["mse"]),  epoch)
            writer.add_scalar("val/rmse", float(val_res["rmse"]), epoch)
            writer.add_scalar("val/mae",  float(val_res["mae"]),  epoch)
            writer.add_scalar("val/mape", float(val_res["mape"]), epoch)
            if test_report:
                writer.add_scalar("test/loss_best", float(test_report["loss"]), epoch)
                writer.add_scalar("test/mse_best",  float(test_report["mse"]),  epoch)
                writer.add_scalar("test/rmse_best", float(test_report["rmse"]), epoch)
                writer.add_scalar("test/mae_best", float(test_report["mae"]), epoch)
                writer.add_scalar("test/mape_best", float(test_report["mape"]), epoch)

        if (epoch - best_iter) >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience}).")
            logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

        # latest (per-epoch)
        torch.save({
            "state_dict": MODEL.state_dict(),
            "args": vars(args),
            "input_dim": INPUT_DIM,
            "best_val": best_val,
            "best_val_mae": best_val_mae,
            "best_val_mape": best_val_mape,
            "best_iter": best_iter,
        }, model_latest_path)

except KeyboardInterrupt:
    print("\n[interrupt] KeyboardInterrupt — saving latest and exiting.")
    torch.save({
        "state_dict": MODEL.state_dict(),
        "args": vars(args),
        "input_dim": INPUT_DIM,
        "best_val": best_val,
        "best_val_mae": best_val_mae,
        "best_val_mape": best_val_mape,
        "best_iter": best_iter,
    }, model_latest_path)
    raise

training_duration = time.time() - training_start_time
logger.info("="*80)
logger.info("TRAINING LOOP COMPLETED")
logger.info(f"Training duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
logger.info(f"Best epoch: {best_iter}, Best val MSE: {best_val:.6f}")
logger.info("="*80)

print(f"Best val MSE: {best_val:.6f} @ epoch {best_iter}")
print(f"Saved best:   {model_best_path}")
print(f"Saved latest: {model_latest_path}")
if test_report:
    print(
        "Test metrics — "
        f"Loss: {test_report['loss']:.6f}, "
        f"MSE: {test_report['mse']:.6f}, "
        f"RMSE: {test_report['rmse']:.6f}, "
        f"MAE: {test_report['mae']:.6f}, "
        f"MAPE: {test_report['mape']*100.0:.2f}%"
    )
    logger.info(
        "Test metrics: "
        f"MSE={test_report['mse']:.6f}, RMSE={test_report['rmse']:.6f}, "
        f"MAE={test_report['mae']:.6f}, MAPE={test_report['mape']*100.0:.2f}%"
    )
# ---- write shared results row ----

if last_train_loss is None:
    try:
        aux_batch = utils.get_next_batch(data_obj["train_dataloader"])
        with torch.no_grad():
            aux_res = evaluation.compute_all_losses(MODEL, aux_batch)
        last_train_loss = float(aux_res["loss"].item())
    except Exception:
        last_train_loss = None

params = {
    "epochs": args.epochs,
    "early_stop": args.patience,
    "patience": args.patience,
    "batch_size": args.batch_size,
    "learn_rate": (args.learn_rate if hasattr(args, "learn_rate")
                   else args.__dict__.get("learn-rate", None)),
    "weight_decay": args.weight_decay,
    "function": args.function,
    "embedding_dim_per_head": args.embedding_dim,
    "num_heads": args.num_heads,
    "depth": args.depth,
    "seed": args.seed,
    "observation_time": args.observation_time,
}
metrics = {
    "best_epoch": best_iter if "best_iter" in locals() else None,
    "val_mse_best": best_val,
    "val_rmse_best": float((best_val + 1e-8) ** 0.5),
    "val_mae_best": (float(best_val_mae) if best_val < float("inf") else None),
    "val_mape_best": (float(best_val_mape) if best_val < float("inf") else None),
    "train_loss_last_batch": (float(last_train_loss) if last_train_loss is not None else None),
    "test_mse_best": (float(test_report["mse"]) if test_report else None),
    "test_rmse_best": (float(test_report["rmse"]) if test_report else None),
    "test_mae_best": (float(test_report["mae"]) if test_report else None),
    "test_mape_best": (float(test_report["mape"]) if test_report else None),
}
write_result(
    model_name="FLD",
    dataset=dataset_name,
    metrics=metrics,
    params=params,
    run_id=str(experiment_id),
)

# ---- Output JSON for benchmark scripts ----
import json
json_summary = {}
for key, value in metrics.items():
    if value is None:
        continue
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        continue
    if isinstance(numeric, float) and (numeric != numeric or abs(numeric) == float("inf")):
        continue  # Skip NaN and inf
    json_summary[key] = numeric

print(json.dumps(json_summary))

if writer:
    writer.flush(); writer.close()

# ===== FINAL TIMING SUMMARY =====
script_total_time = time.time() - SCRIPT_START_TIME
logger.info("="*80)
logger.info("FLD TRAINING COMPLETED SUCCESSFULLY")
logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Total script duration: {script_total_time:.2f} seconds ({script_total_time/60:.2f} minutes)")
logger.info(f"  - Dataset loading: {dataset_load_time:.2f}s")
logger.info(f"  - Model creation: {model_create_time:.2f}s")
logger.info(f"  - Training loop: {training_duration:.2f}s ({training_duration/60:.2f} min)")
logger.info(f"  - Other overhead: {(script_total_time - dataset_load_time - model_create_time - training_duration):.2f}s")
logger.info("="*80)
