# Preprocessing Details

This repository routes **every** dataset through the shared loader in `lib/parse_datasets.py`, so all models see consistent splits, normalization, and batching. Below is a walk-through of each stage, followed by dataset-specific notes and a recap of the min–max statistics used throughout training and evaluation.

---

## 1. Dataset Discovery and Base Workflow

1. **Path resolution**  
   `_get_data_path()` (see `lib/parse_datasets.py:16-34`) probes `data/<dataset>`, `../data/<dataset>`, and the absolute path derived from `lib/` so the loader works whether you run inside `tPatchGNN/`, `FLD/`, `FLD_ICC/`, etc.

2. **Dataset class**  
   The loader instantiates the appropriate dataset object (`PhysioNet`, `MIMIC`, `USHCN`, `PersonActivity`). Each class converts raw files into a list of tuples `(record_id, tt, vals, mask)` where:
   - `record_id` – patient or station identifier.  
   - `tt` – tensor of observation timestamps (hours for PhysioNet/MIMIC, milliseconds for Activity, months for USHCN).  
   - `vals` – tensor of observed measurement values.  
   - `mask` – tensor with `1` where a variable is observed, `0` otherwise.

3. **Splitting**  
   All datasets use the same deterministic splits via `sklearn.model_selection.train_test_split` (see `lib/parse_datasets.py:63-89`). First, 80 % of records become the “seen” subset; the remaining 20 % are held out for test. The “seen” records are then split into 75 % training and 25 % validation **without reshuffling**, yielding 60/20/20 overall. Test IDs are printed for traceability.

4. **Normalization stats**  
   `get_data_min_max()` runs over the union of training and validation data to find per-variable minima and maxima, along with the maximum timestamp observed (`time_max`). These scalars drive every downstream normalization call.

5. **Collate function**  
   - `variable_time_collate_fn` (no patching) packs each batch into `[B, L, D]` tensors with padding, min–max normalizes observations and prediction targets separately, and normalizes time to `[0,1]`.  
   - `patch_variable_time_collate_fn` (patching) first gathers the full observed window, then splits it into overlapping temporal patches controlled by `args.patch_size`, `args.stride`, and `args.npatch`. Patching only affects the observation tensors—the future targets remain sequential.

6. **Data object**  
   The function returns dictionaries with infinite-data generators (`utils.inf_generator(...)`), batch counts, and the statistics used later by models (input dimension, `data_min`, `data_max`, `time_max`). When `patch_ts=True`, the object also includes `min`, `max`, `global_mean`, and `global_std` so the tPatchGNN script can convert its min–max outputs back to TSDM-style z-scores.

---

## 2. Dataset-Specific Notes

### PhysioNet (Challenge 2012 ICU)
- **Raw to processed** – `lib/physionet.PhysioNet` downloads and converts each ICU stay into hourly-aligned tensors, saving `set-a_0.0.pt`, etc.
- **Quantization** – optional rounding of timestamps (`--quantization`) before saving.
- **Collate behavior** – patching slices the 24h observation window into `args.npatch` segments for the transformer-style models; non-patched loaders feed standard `[B, L, D]` tensors to FLD/IC-FLD/others.
- **Targets** – everything after `args.history` hours becomes the forecast horizon.

### MIMIC-III Derived Dataset
- **Processing** – `lib/mimic.MIMIC` reads `data/mimic/raw/full_dataset.csv`, groups by patient ID, converts minutes to hours (divide by 60), and builds tensors for every `Value*` column with its matching `Mask*`. The processed data are cached as `data/mimic/processed/mimic.pt`.
- **Splits** – identical 60/20/20 record split; no additional filtering.
- **Normalization & batching** – exactly the same as PhysioNet once the tuples are loaded. The difference between IC-FLD/FLD and tPatchGNN is only whether the collate function patchifies the observed window.

### USHCN (Monthly Climate)
- **Preprocessing** – `lib.ushcn.USHCN` loads sequences of monthly measurements (length 48). For modeling, `USHCN_time_chunk()` further slices each sequence into history and prediction pieces respecting `args.history`, `args.n_months`, and `args.pred_window`.
- **Collate functions** – `USHCN_variable_time_collate_fn` and its patch counterpart consume the chunked records. Time normalization differs slightly: since timestamps are regular monthly indices, `time_max` is computed from the data and reused post-chunking.
- **Observation horizon** – defaults to 48 months of context with a 1-month prediction target, but these values can be overridden before calling `parse_datasets`.

### Person Activity
- **Raw data** – `lib.person_activity.PersonActivity` fetches the dataset and outputs `(record_id, tt, vals, mask)` with millisecond timestamps.
- **Observation / prediction setup** – The loader fixes `args.pred_window = 1000` ms, so the maximum time fed to `normalize_masked_tp` becomes `args.history + args.pred_window`.
- **Chunking** – `Activity_time_chunk()` produces overlapping sequences similar to the USHCN pipeline, ensuring each batch contains uniform history/target lengths before applying patch or non-patch collate functions.

---

## 3. Min–Max Metrics Explained

**Where do the numbers come from?**  
`get_data_min_max()` scans every training + validation record for each variable, capturing the true min and max observed values. These tensors (`data_min`, `data_max`) characterize the marginal range of each vital sign / feature before any scaling.

**How are they applied?**  
`lib/utils.normalize_masked_data()` subtracts `data_min` and divides by `(data_max - data_min)` element-wise, leaving masked entries at zero. This is done separately for:
- Observed inputs (history window)
- Future targets (`data_to_predict`)

Timestamps go through `normalize_masked_tp()` using `[0, time_max]`, keeping time in `[0,1]` as well.

**Interpreting metrics**  
Because both the model inputs and the evaluation targets are min–max scaled, losses such as MSE/RMSE/MAE are reported in normalized units (0–1 range) unless explicitly converted back (e.g., the TSDM z-score reporting path). The evaluation code now mirrors `lib/evaluation.py`: it first computes the average error per variable (each channel’s error is divided by its number of valid targets) and then averages those channel-level scores. This prevents densely sampled vitals from dominating the metric and aligns IC‑FLD/FLD with GraFITi/TSDM reporting. When the tPatchGNN trainer needs TSDM metrics, it fetches `data_obj["global_mean"]` and `["global_std"]`, inverts the min–max scaling, and re-normalizes to z-scores before logging.

**Key implications**
1. **Consistency** – All models are evaluated on the same scaled targets, so cross-model comparisons are meaningful even when architectures differ.
2. **Reconstruction** – You can recover physical units by inverting the scaling: `y_original = y_scaled * (data_max - data_min) + data_min`.
3. **Per-variable fairness** – Because each metric averages channel-wise errors before collapsing to a scalar, datasets with sparse variables (e.g., MIMIC after 24 h) no longer look artificially worse than dense ones purely due to target counts.
4. **Reporting choices** – If you want z-score metrics everywhere, use the `--fldReport` flag (IC-FLD/FLD) or ensure `global_mean/std` are available (tPatchGNN). Otherwise, the default logs min–max-normalized errors.

---

## 4. Practical Checklist

| Step | PhysioNet | MIMIC | USHCN | Activity |
|------|-----------|-------|-------|----------|
| Raw → Processed | `lib/physionet` downloads & saves `.pt` files | `lib/mimic` parses `full_dataset.csv` | `lib/ushcn` loads monthly sequences | `lib/person_activity` fetches data |
| Record splits | 60/20/20, random seed 42 | same | same | same |
| History/Targets | `args.history` (default 24h) | same | `args.n_months`, `args.pred_window` (default 48/1) | `args.history`, `args.pred_window=1000` ms |
| Normalization stats | `get_data_min_max(seen_data)` | same | same | same (`time_max` manually set to `history+pred_window`) |
| Collate options | Patched or non-patched | Patched or non-patched | USHCN-specific collates | Activity-specific chunking before collate |
| Metrics | Min–max by default, optional TSDM | same | same | same |

Keep this sheet handy whenever you tweak preprocessing so you can confirm splits, normalization, and batching are still aligned across models.

---

## Important Note on Training Defaults

Some summaries of this project mention “Adam with patience 5 and gradient clipping at norm 1.0”. The current trainers actually:
1. use `torch.optim.AdamW` (not Adam) paired with `ReduceLROnPlateau` whose patience is fixed at 10 epochs, and  
2. **do not** perform any gradient clipping—the code never calls `clip_grad_norm_` or similar.

All other training details referenced in that summary (learning rate `1e-3`, dataset-specific batch sizes as configured on the CLI, masked losses, etc.) remain accurate. Keep these two corrections in mind when reproducing or extending experiments.
