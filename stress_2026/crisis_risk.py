"""
2026 Gulf crisis: Brent-WTI dislocation and spread-book risk post-mortem.

Questions:
  1. How far outside the 2023-25 regime was 2026 (vol, correlation, tail moves in calm-regime sigmas)?
  2. Would a risk model have caught it? Static (calm-calibrated) vs EWMA 99% VaR breaches on a 1-lot spread.
  3. What did the crisis do to a mean-reversion spread book, and does volatility-targeted sizing help?

Data: EIA daily spot Brent (Dated Brent) and WTI Cushing, $/bbl, via datahub.io oil-prices.
Usage: python crisis_risk.py --brent brent-daily.csv --wti wti-daily.csv --out results
"""
import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BBL = 1_000
CALM = ("2023-01-01", "2025-12-31")
CRISIS = ("2026-03-01", "2026-11-17")
Z99 = 2.326
LAMBDA = 0.94                 # RiskMetrics EWMA decay
COST = 25.0                   # $ per spread per entry/exit


def load(bp, wp):
    b = pd.read_csv(bp, parse_dates=["Date"]).set_index("Date")["Price"].rename("brent")
    w = pd.read_csv(wp, parse_dates=["Date"]).set_index("Date")["Price"].rename("wti")
    df = pd.concat([b, w], axis=1, join="inner").sort_index().dropna().loc["2015":]
    df["spread"] = df["brent"] - df["wti"]
    df["dS"] = df["spread"].diff()                       # $/bbl change in spread
    df["rb"] = np.log(df["brent"]).diff(); df["rw"] = np.log(df["wti"]).diff()
    return df.dropna()


def regime_stats(df):
    out = {}
    for name, (a, b) in {"calm_2023_25": CALM, "crisis_2026": CRISIS}.items():
        x = df.loc[a:b]
        out[name] = {
            "days": len(x),
            "spread_mean": round(x["spread"].mean(), 2), "spread_max": round(x["spread"].max(), 2),
            "spread_daily_sd_usd_bbl": round(x["dS"].std(), 3),
            "brent_ann_vol": round(x["rb"].std() * np.sqrt(252), 3),
            "wti_ann_vol": round(x["rw"].std() * np.sqrt(252), 3),
            "brent_wti_return_corr": round(x["rb"].corr(x["rw"]), 3),
            "worst_1d_spread_move_usd_per_lot": round(x["dS"].abs().max() * BBL, 0),
        }
    calm_sd = df.loc[CALM[0]:CALM[1], "dS"].std()
    cr = df.loc[CRISIS[0]:CRISIS[1], "dS"]
    sig = (cr.abs() / calm_sd)
    out["crisis_tail"] = {
        "calm_daily_sd_usd_per_lot": round(calm_sd * BBL, 0),
        "days_beyond_4_calm_sigma": int((sig > 4).sum()),
        "days_beyond_10_calm_sigma": int((sig > 10).sum()),
        "max_move_in_calm_sigmas": round(float(sig.max()), 1),
        "max_move_date": str(sig.idxmax().date()),
        "expected_4sigma_days_if_normal": round(len(cr) * 2 * 3.17e-5, 3),
        "vol_ratio_crisis_vs_calm": round(cr.std() / calm_sd, 1),
    }
    return out


def var_backtest(df):
    """1-lot spread position, daily P&L = dS * 1000. Compare two 99% 1-day VaR models on 2026."""
    pnl = df["dS"] * BBL
    calm_sd = pnl.loc[CALM[0]:CALM[1]].std()
    static_var = Z99 * calm_sd
    # EWMA variance, forecast for day t uses info up to t-1
    var = np.empty(len(pnl)); var[0] = pnl.iloc[:60].var()
    for t in range(1, len(pnl)):
        var[t] = LAMBDA * var[t - 1] + (1 - LAMBDA) * pnl.iloc[t - 1] ** 2
    ewma_var = pd.Series(Z99 * np.sqrt(var), index=pnl.index)
    out = {}
    for name, (a, b) in {"calm_2023_25": CALM, "crisis_2026": CRISIS}.items():
        p = pnl.loc[a:b]; e = ewma_var.loc[a:b]
        n = len(p)
        out[name] = {"days": n, "expected_breaches_per_side": round(n * 0.01, 1),
                     "static_breaches_two_sided": int((p.abs() > static_var).sum()),
                     "ewma_breaches_two_sided": int((p.abs() > e).sum()),
                     "static_var_usd": round(static_var, 0),
                     "ewma_var_median_usd": round(float(e.median()), 0),
                     "ewma_var_max_usd": round(float(e.max()), 0)}
    # Kupiec POF on crisis, two-sided at 2% (1% each tail)
    from scipy.stats import chi2
    def kupiec(x, n, p=0.02):
        if x == 0: return 1.0
        ph = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
        return float(1 - chi2.cdf(lr, 1))
    for k in out:
        n = out[k]["days"]
        out[k]["kupiec_p_static"] = round(kupiec(out[k]["static_breaches_two_sided"], n), 4)
        out[k]["kupiec_p_ewma"] = round(kupiec(out[k]["ewma_breaches_two_sided"], n), 4)
    return out, ewma_var, static_var


def mr_signal(spread, lookback=20, entry=2.0, exit_z=0.5, stop_z=4.0, max_hold=60, lag=1):
    """Same rule as the brent-wti-spread project (params chosen there on 2000-14, next-day execution)."""
    z = ((spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()).values
    pos = np.zeros(len(z)); p = 0; held = 0
    for t in range(len(z)):
        if np.isnan(z[t]): continue
        if p == 0:
            if z[t] > entry: p, held = -1, 0
            elif z[t] < -entry: p, held = 1, 0
        else:
            held += 1
            if abs(z[t]) < exit_z or abs(z[t]) > stop_z or held >= max_hold \
               or (p == 1 and z[t] > exit_z) or (p == -1 and z[t] < -exit_z):
                p = 0
        pos[t] = p
    return pd.Series(pos, index=spread.index).shift(lag).fillna(0)


def sizing_compare(df, ewma_var):
    """Fixed 3 lots vs volatility-targeted lots. Risk budget = calm-regime (2023-25) median 99% VaR of
    3 lots; lots = floor(budget / EWMA VaR of 1 lot), capped at 3 -> ~3 lots in calm markets, fewer in stress."""
    sig = mr_signal(df["spread"])
    budget = 3 * ewma_var.loc[CALM[0]:CALM[1]].median()
    lots_vt = np.floor((budget / ewma_var).clip(upper=3)).fillna(0)
    res = {}
    for name, lots in {"fixed_3_lots": pd.Series(3.0, index=df.index), "vol_targeted": lots_vt}.items():
        pos = sig * lots.shift(1).fillna(0)            # size set from yesterday's risk estimate
        gross = pos.shift(1).fillna(0) * df["dS"] * BBL
        cost = pos.diff().abs().fillna(0) * COST
        net = gross - cost
        row = {}
        for per, (a, b) in {"2015_2025": ("2015", "2025"), "crisis_2026": CRISIS}.items():
            x = net.loc[a:b]; eq = x.cumsum()
            row[per] = {"net_pnl_usd": round(float(x.sum()), 0),
                        "max_drawdown_usd": round(float((eq - eq.cummax()).min()), 0),
                        "worst_day_usd": round(float(x.min()), 0),
                        "best_day_usd": round(float(x.max()), 0),
                        "sharpe": round(float(x.mean() / x.std() * np.sqrt(252)), 2) if x.std() > 0 else 0.0,
                        "pnl_per_unit_worst_day": round(float(x.sum() / abs(x.min())), 2) if x.min() < 0 else None}
        res[name] = row
        res[name]["_series"] = net
    res["vol_targeted"]["avg_lots_crisis"] = round(float(lots_vt.loc[CRISIS[0]:CRISIS[1]].mean()), 2)
    res["vol_targeted"]["avg_lots_2015_2025"] = round(float(lots_vt.loc["2015":"2025"].mean()), 2)
    res["vol_targeted"]["pct_days_zero_lots_crisis"] = round(float((lots_vt.loc[CRISIS[0]:CRISIS[1]] == 0).mean()), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brent", required=True); ap.add_argument("--wti", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load(a.brent, a.wti)

    reg = regime_stats(df)
    var_res, ewma_var, static_var = var_backtest(df)
    siz = sizing_compare(df, ewma_var)
    series = {k: siz[k].pop("_series") for k in ["fixed_3_lots", "vol_targeted"]}
    top = (df.loc[CRISIS[0]:CRISIS[1], "dS"] * BBL).abs().nlargest(8)
    top_moves = [{"date": str(d.date()), "spread_move_usd_per_lot": round(float(df.loc[d, "dS"] * BBL), 0),
                  "brent": float(df.loc[d, "brent"]), "wti": float(df.loc[d, "wti"])} for d in top.index]

    res = {"data": {"start": str(df.index[0].date()), "end": str(df.index[-1].date())},
           "regimes": reg, "var_backtest": var_res, "sizing": siz, "largest_crisis_spread_moves": top_moves}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))

    # charts
    x = df.loc["2025-07":]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
    ax[0].plot(x.index, x["brent"], label="Brent (Dated, spot)", color="#1f6f8b")
    ax[0].plot(x.index, x["wti"], label="WTI Cushing (spot)", color="#c0392b")
    ax[0].set_ylabel("$/bbl"); ax[0].legend(loc="upper left"); ax[0].set_title("2026 crude shock: Brent vs WTI")
    ax0b = ax[0].twinx(); ax0b.fill_between(x.index, x["spread"], color="grey", alpha=0.25); ax0b.set_ylabel("spread $/bbl")
    p = x["dS"] * BBL
    ax[1].bar(p.index, p, color=np.where(p.abs() > ewma_var.loc[x.index], "#c0392b", "#999"), width=1.2)
    ax[1].plot(ewma_var.loc[x.index], color="#1f6f8b", lw=1, label="EWMA 99% VaR")
    ax[1].plot(-ewma_var.loc[x.index], color="#1f6f8b", lw=1)
    ax[1].axhline(static_var, color="k", ls="--", lw=0.8, label="static VaR (2023-25 calibration)")
    ax[1].axhline(-static_var, color="k", ls="--", lw=0.8)
    ax[1].set_ylabel("1-lot spread P&L ($/day)"); ax[1].legend(loc="upper left", fontsize=8)
    for k, c in [("fixed_3_lots", "#c0392b"), ("vol_targeted", "#1f6f8b")]:
        ax[2].plot(series[k].loc[x.index[0]:].cumsum(), color=c, label=k.replace("_", " "))
    ax[2].set_ylabel("Mean-reversion book cum. P&L ($)"); ax[2].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out / "crisis_risk.png", dpi=130)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
