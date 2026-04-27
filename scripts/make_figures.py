import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cluster import KMeans

def _align(*arrs):
    L = min(len(np.asarray(a).reshape(-1)) for a in arrs)
    return [np.asarray(a).reshape(-1)[:L] for a in arrs], L

def _save(figdir, fname):
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[OK] {path}")

def _acf(resid, max_lag=40):
    resid = np.asarray(resid).reshape(-1)
    resid = resid - resid.mean()
    denom = np.sum(resid**2) + 1e-12
    out = []
    for lag in range(1, max_lag+1):
        out.append(np.sum(resid[:-lag] * resid[lag:]) / denom)
    return np.asarray(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/predictions_test.csv")
    ap.add_argument("--raw_processed", default="data/processed/lithium_processed.csv",
                    help="Processed CSV (Date, Close) used to compute regimes on TRAIN only.")
    ap.add_argument("--window", type=int, default=175)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--outdir", default="outputs/figures")
    args = ap.parse_args()

    dfp = pd.read_csv(args.pred)
    dates = pd.to_datetime(dfp["Date"])
    y = dfp["TruePrice"].values
    p_bi = dfp["BiLSTM_Pred"].values
    p_q  = dfp["QBiLSTM_Pred"].values
    p_h  = dfp["Hybrid_Pred"].values

    # ---------- Figs 2–5: True vs Pred ----------
    plt.figure(figsize=(14,6)); plt.plot(dates, y, label="True"); plt.plot(dates, p_bi, label="BiLSTM"); plt.legend()
    plt.title("True vs Predicted (BiLSTM)"); plt.xlabel("Date"); plt.ylabel("Price")
    _save(args.outdir, "Fig2_BiLSTM_true_vs_pred.png")

    plt.figure(figsize=(14,6)); plt.plot(dates, y, label="True"); plt.plot(dates, p_q, label="QBiLSTM"); plt.legend()
    plt.title("True vs Predicted (QBiLSTM)"); plt.xlabel("Date"); plt.ylabel("Price")
    _save(args.outdir, "Fig3_QBiLSTM_true_vs_pred.png")

    plt.figure(figsize=(14,6)); plt.plot(dates, y, label="True"); plt.plot(dates, p_h, label="Hybrid"); plt.legend()
    plt.title("True vs Predicted (Hybrid)"); plt.xlabel("Date"); plt.ylabel("Price")
    _save(args.outdir, "Fig4_Hybrid_true_vs_pred.png")

    plt.figure(figsize=(14,6))
    plt.plot(dates, y, label="True")
    plt.plot(dates, p_bi, label="BiLSTM")
    plt.plot(dates, p_q, label="QBiLSTM")
    plt.plot(dates, p_h, label="Hybrid")
    plt.legend()
    plt.title("True vs Predicted (All Models)"); plt.xlabel("Date"); plt.ylabel("Price")
    _save(args.outdir, "Fig5_AllModels_true_vs_pred.png")

    # ---------- Residuals (for diagnostics figs) ----------
    # Residuals in PRICE space (for error/regime plots)
    r_bi_price = p_bi - y
    r_q_price  = p_q  - y
    r_h_price  = p_h  - y

    # Residuals in SCALED space (for ACF + diagnostics figures)
    y_s = dfp["TrueScaled"].values
    p_bi_s = dfp["BiLSTM_Pred_Scaled"].values
    p_q_s  = dfp["QBiLSTM_Pred_Scaled"].values
    p_h_s  = dfp["Hybrid_Pred_Scaled"].values
    r_bi = p_bi_s - y_s
    r_q  = p_q_s  - y_s
    r_h  = p_h_s  - y_s

    # ---------- Fig 6: Residual ACF (1x3) ----------
    max_lag = 40
    fig = plt.figure(figsize=(15,4))
    for k,(name,resid) in enumerate([("BiLSTM",r_bi),("QBiLSTM",r_q),("Hybrid",r_h)], start=1):
        ax = plt.subplot(1,3,k)
        acf_vals = _acf(resid, max_lag=max_lag)
        markerline, stemlines, baseline = ax.stem(np.arange(1,max_lag+1), acf_vals)
        try: baseline.set_visible(False)
        except Exception: pass
        ax.set_title(name); ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
    plt.suptitle("Residual ACF (lags 1..40)")
    _save(args.outdir, "Fig6_Residual_ACF.png")

    # ---------- Fig 7: Parity plots (scaled units) ----------
    y_s = dfp["TrueScaled"].values
    p_bi_s = dfp["BiLSTM_Pred_Scaled"].values
    p_q_s  = dfp["QBiLSTM_Pred_Scaled"].values
    p_h_s  = dfp["Hybrid_Pred_Scaled"].values

    fig = plt.figure(figsize=(15,4))
    for k,(name, pred) in enumerate([("BiLSTM", p_bi_s), ("QBiLSTM", p_q_s), ("Hybrid", p_h_s)], start=1):
        ax = plt.subplot(1,3,k)
        ax.scatter(y_s, pred, s=10, alpha=0.7)
        lims = [min(y_s.min(), pred.min()), max(y_s.max(), pred.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("True (scaled)")
        ax.set_ylabel("Pred (scaled)")
    plt.suptitle("Parity plots (scaled)")
    _save(args.outdir, "Fig7_Parity.png")

    # ---------- Fig 8: Residual diagnostics (3x3) ----------
    def resid_row(ax_abs, ax_hist, ax_qq, resid, title):
        ax_abs.plot(np.abs(resid), lw=0.8); ax_abs.set_title(f"|r_t| - {title}")
        ax_hist.hist(resid, bins=30, alpha=0.85); ax_hist.set_title(f"Hist - {title}")
        stats.probplot(resid, dist="norm", plot=ax_qq); ax_qq.set_title(f"QQ - {title}")

    fig = plt.figure(figsize=(15,10))
    rows = [("BiLSTM", r_bi), ("QBiLSTM", r_q), ("Hybrid", r_h)]
    for i,(name,resid) in enumerate(rows):
        ax1 = plt.subplot(3,3,3*i+1)
        ax2 = plt.subplot(3,3,3*i+2)
        ax3 = plt.subplot(3,3,3*i+3)
        resid_row(ax1, ax2, ax3, resid, name)
    plt.suptitle("Residual diagnostics on test set (scaled residuals)")
    _save(args.outdir, "Fig8_Residual_Diagnostics.png")

    # ---------- Regime detection (TRAIN only) + test regime timeline ----------
    # Rebuild regimes from processed series
    dfr = pd.read_csv(args.raw_processed)
    dfr["Date"] = pd.to_datetime(dfr["Date"])
    close = pd.to_numeric(dfr["Close"], errors="coerce").values.astype(float)
    logp = np.log(close + 1e-12)
    rets = np.diff(logp)

    rwin = 20
    roll_mean = pd.Series(rets).rolling(rwin).mean().fillna(0.0).values
    roll_std  = pd.Series(rets).rolling(rwin).std().fillna(0.0).values

    N = len(close)
    num_samples = N - args.window
    split = int(num_samples * args.train_frac)

    target_idx = np.arange(args.window, N)  # price indices for targets
    feat_all = np.column_stack([roll_mean[target_idx - 1], roll_std[target_idx - 1]])
    feat_train = feat_all[:split]
    feat_test  = feat_all[split:]

    kmeans = KMeans(n_clusters=3, random_state=7, n_init=10)
    kmeans.fit(feat_train)

    # map clusters by increasing train volatility (std)
    vols = []
    for cid in range(3):
        m = (kmeans.labels_ == cid)
        vols.append((cid, float(np.mean(feat_train[m, 1])) if np.any(m) else 1e9))
    vols = sorted(vols, key=lambda x: x[1])
    cluster_to_reg = {vols[0][0]: 0, vols[1][0]: 1, vols[2][0]: 2}

    reg_test = np.array([cluster_to_reg[c] for c in kmeans.predict(feat_test)], dtype=int)

    # Align regime length with predictions length (test targets)
    L = min(len(reg_test), len(y))
    reg_test = reg_test[:L]
    dates_r = dates[:L]

    # ---------- Fig 11: Regime timeline ----------
    plt.figure(figsize=(14,3))
    plt.plot(dates_r, reg_test, lw=1)
    plt.yticks([0,1,2], ["Calm","Medium","Volatile"])
    plt.title("Detected regime timeline (test set)")
    plt.xlabel("Date")
    _save(args.outdir, "Fig11_Regime_Timeline.png")

    # ---------- Fig 9: Error by regime (boxplots per model) ----------
    abs_err_bi = np.abs(r_bi_price[:L])
    abs_err_q  = np.abs(r_q_price[:L])
    abs_err_h  = np.abs(r_h_price[:L])

    fig = plt.figure(figsize=(15,4))
    for k,(name,err) in enumerate([("BiLSTM",abs_err_bi),("QBiLSTM",abs_err_q),("Hybrid",abs_err_h)], start=1):
        ax = plt.subplot(1,3,k)
        data = [err[reg_test==0], err[reg_test==1], err[reg_test==2]]
        ax.boxplot(data, labels=["Calm","Medium","Volatile"], showfliers=False)
        ax.set_title(name); ax.set_ylabel("|Error|")
    plt.suptitle("Error distribution by regime (test set)")
    _save(args.outdir, "Fig9_Error_by_Regime.png")

    # ---------- Fig 10: Rolling MAE ----------
    win = 30
    def rolling_mae(err):
        s = pd.Series(np.abs(err))
        return s.rolling(win).mean().bfill().values

    plt.figure(figsize=(14,4))
    plt.plot(dates, rolling_mae(r_bi_price), label="BiLSTM")
    plt.plot(dates, rolling_mae(r_q_price),  label="QBiLSTM")
    plt.plot(dates, rolling_mae(r_h_price),  label="Hybrid")
    plt.title(f"Rolling MAE on test set (window={win})")
    plt.xlabel("Date"); plt.ylabel("Rolling MAE")
    plt.legend()
    _save(args.outdir, "Fig10_Rolling_MAE.png")

if __name__ == "__main__":
    main()
