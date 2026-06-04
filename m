# ============== IV-SURFACE — 2-YEAR ROLLING RETRAIN (first test) ==============
import sys, importlib
sys.path.insert(0, r"C:\Users\<you>\Documents\claude_file\claude_cowork")  # EDIT (folder with the .py + pickles)
import iv_surface_forecast_pipeline as P; importlib.reload(P)

P.CONFIG.update(
    # ---- data (pickles sit next to the notebook) ----
    data_dir         = ".",
    force_rebuild    = False,          # True the first time / after changing data
    resample_monthly = False,          # keep all obs for training; portfolio is monthly

    # ---- 2-YEAR ROLLING RETRAIN ----
    window_scheme    = "rolling",      # retrain each month on a trailing window
    roll_years       = 2,              # <-- the 2-year retraining window
    oos_start        = "1998-01-01",   # first OOS month (>= 2yr after data start)
    oos_end          = "2022-01-01",
    embargo_months   = 1,
    warmup_epochs    = 10,             # epochs on the first window
    transfer_epochs  = 5,              # epochs per monthly refit

    # ---- quick-test slice (raise / set None for full universe) ----
    max_stocks         = 100,
    subset_min_oos_obs = 6,

    # ---- targets / models ----
    targets   = ["TRET_SP500_EX_F20D"],                 # add F20D / F5D variants later
    models    = ["rf", "deepmlp", "cnn", "cnn1", "cnn4", "cnn5", "lstm"],
    ensemble_K = 5,                    # bump to 25-50 once it looks right

    # ---- standardisation & compute ----
    surface_norm = "pixel_zscore",     # or "xs_zscore"
    use_cuda     = False,              # CPU-only
    n_cpus       = 20,                 # distributed across cores
    n_deciles    = 10,
)

STAGE = "check"    # "check" -> validate schema/coverage first; "run" -> full pipeline
if STAGE == "check":
    panel, schema, summary = P.data_check(P.CONFIG)
else:
    results = P.run_grid(P.CONFIG)     # ranked table incl. R2_is / R2_oos
    display(results)
# outputs -> iv_surface_po/{results,figures}
