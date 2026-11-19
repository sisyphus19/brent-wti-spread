"""
Brent-WTI spread: statistical analysis and mean-reversion backtest.

Data: EIA daily spot prices for Brent and WTI ($/bbl), via datahub.io
(github.com/datasets/oil-prices). Spot is a proxy for the front-month
ICE Brent / NYMEX WTI futures spread; both contracts are 1,000 bbl, so a
1:1 spread is 1 Brent vs 1 WTI contract.

Usage:  python spread_backtest.py --brent brent-daily.csv --wti wti-daily.csv --out results
"""
import argparse, json, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm

BBL = 1_000                 # barrels per contract
COST_PER_SPREAD_SIDE = 25.0 # $ per spread per entry or exit: 2 legs x (1 tick $10 slippage + $2.5 fees)
IS_END = "2014-12-31"       # in-sample (parameter selection) ends here
START = "2000-01-01"


def load(brent_path, wti_path):
    b = pd.read_csv(brent_path, parse_dates=["Date"]).set_index("Date")["Price"].rename("brent")
    w = pd.read_csv(wti_path, parse_dates=["Date"]).set_index("Date")["Price"].rename("wti")
    df = pd.concat([b, w], axis=1, join="inner").dropna().loc[START:]
    df["spread"] = df["brent"] - df["wti"]
    return df


def half_life(s):
    """OU half-life from AR(1): dS_t = a + b*S_{t-1} + e  ->  HL = -ln2 / ln(1+b)."""
    s = s.dropna()
    ds = s.diff().dropna()
    x = sm.add_constant(s.shift(1).loc[ds.index])
    b = sm.OLS(ds, x).fit().params.iloc[1]
    return float(-np.log(2) / np.log(1 + b)) if -1 < b < 0 else np.inf


def stats_tests(df):
    out = {}
    for name, sub in {"full": df, "2000-2010": df.loc[:"2010"], "2011-2014": df.loc["2011":"2014"],
                      "2015-2026": df.loc["2015":]}.items():
        p_adf = adfuller(sub["spread"], autolag="AIC", result_object=False)[1]
        p_eg = coint(np.log(sub["brent"].clip(lower=1)), np.log(sub["wti"].clip(lower=1)))[1]
        out[name] = {"adf_p_spread": round(p_adf, 4), "engle_granger_p": round(p_eg, 4),
                     "half_life_days": round(half_life(sub["spread"]), 1),
                     "spread_mean": round(sub["spread"].mean(), 2), "spread_std": round(sub["spread"].std(), 2)}
    return out


def backtest(spread, lookback, entry, exit_z=0.5, stop_z=4.0, max_hold=60, lag=1):
    """Signal computed at close t. With lag=0 the position is put on at the same close and earns t -> t+1;
    lag=1 (default, realistic) executes one day later, which removes the artifact from Brent (London) and
    WTI (New York) being assessed at different times of day (see README).
    +1 = long spread (long Brent, short WTI), -1 = short spread."""
    mu = spread.rolling(lookback).mean()
    sd = spread.rolling(lookback).std()
    z = (spread - mu) / sd
    pos = np.zeros(len(spread)); held = 0; p = 0
    zv = z.values
    for t in range(len(spread)):
        if np.isnan(zv[t]):
            pos[t] = 0; continue
        if p == 0:
            if zv[t] > entry: p, held = -1, 0
            elif zv[t] < -entry: p, held = 1, 0
        else:
            held += 1
            if abs(zv[t]) < exit_z or abs(zv[t]) > stop_z or held >= max_hold \
               or (p == 1 and zv[t] > exit_z) or (p == -1 and zv[t] < -exit_z):
                p = 0
        pos[t] = p
    pos = pd.Series(pos, index=spread.index).shift(lag).fillna(0)
    gross = pos.shift(1).fillna(0) * spread.diff().fillna(0) * BBL
    trades = pos.diff().abs().fillna(pos.abs())          # 1 = open or close, 2 = flip
    cost = trades * COST_PER_SPREAD_SIDE
    net = gross - cost
    return pd.DataFrame({"spread": spread, "z": z, "pos": pos, "gross": gross, "cost": cost, "net": net})


def metrics(bt):
    pnl = bt["net"]
    eq = pnl.cumsum()
    dd = eq - eq.cummax()
    # trade-level stats: a trade is a run of identical non-zero positions; its P&L is the gross
    # earned on the following days (position at t earns t->t+1) minus entry and exit costs
    pos = bt["pos"]
    seg = (pos != pos.shift()).cumsum()
    earned_by = bt["gross"].shift(-1).fillna(0)          # gross on t+1 belongs to the position held at t
    g = pd.DataFrame({"seg": seg, "pos": pos, "pnl": earned_by})
    g = g[g["pos"] != 0].groupby("seg").agg(pnl=("pnl", "sum"), hold=("pnl", "size"))
    trade_pnls = (g["pnl"] - 2 * COST_PER_SPREAD_SIDE).tolist(); holds = g["hold"].tolist()
    sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
    tp = np.array(trade_pnls)
    return {"net_pnl_usd": round(float(eq.iloc[-1]), 0), "sharpe": round(float(sharpe), 2),
            "max_drawdown_usd": round(float(dd.min()), 0), "n_trades": int(len(tp)),
            "win_rate": round(float((tp > 0).mean()), 3) if len(tp) else None,
            "avg_trade_usd": round(float(tp.mean()), 0) if len(tp) else None,
            "avg_hold_days": round(float(np.mean(holds)), 1) if holds else None,
            "pct_time_in_market": round(float((bt["pos"] != 0).mean()), 3),
            "total_costs_usd": round(float(bt["cost"].sum()), 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brent", required=True); ap.add_argument("--wti", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = load(a.brent, a.wti)
    tests = stats_tests(df)

    # --- parameter selection on in-sample only ---
    grid = []
    for lb, en in itertools.product([20, 40, 60, 120], [1.5, 2.0, 2.5]):
        bt = backtest(df["spread"], lb, en)
        m_is = metrics(bt.loc[:IS_END])
        grid.append({"lookback": lb, "entry_z": en, **{f"is_{k}": v for k, v in m_is.items()}})
    grid = pd.DataFrame(grid).sort_values("is_sharpe", ascending=False)
    best = grid.iloc[0]
    lb, en = int(best["lookback"]), float(best["entry_z"])

    # run on full history (rolling stats only use past data), report out-of-sample separately
    bt = backtest(df["spread"], lb, en)
    oos = bt.loc[pd.Timestamp(IS_END) + pd.Timedelta(days=1):].copy()
    m_is, m_oos = metrics(bt.loc[:IS_END]), metrics(oos)
    # cost sensitivity out-of-sample
    global COST_PER_SPREAD_SIDE
    base_cost = COST_PER_SPREAD_SIDE; sens = {}
    for c in [0, 25, 50, 100]:
        COST_PER_SPREAD_SIDE = c
        sens[f"cost_{c}"] = metrics(backtest(df["spread"], lb, en).loc[oos.index[0]:])["sharpe"]
    COST_PER_SPREAD_SIDE = base_cost
    # execution-lag sensitivity at the same parameters: same-day fill vs 1-2 day delay
    lag_sens = {f"lag_{L}": {"is_sharpe": metrics(backtest(df["spread"], lb, en, lag=L).loc[:IS_END])["sharpe"],
                             "oos_sharpe": metrics(backtest(df["spread"], lb, en, lag=L).loc[oos.index[0]:])["sharpe"]}
                for L in [0, 1, 2]}
    # what a same-day backtest would have chosen and reported (the biased version)
    naive = max(itertools.product([20, 40, 60, 120], [1.5, 2.0, 2.5]),
                key=lambda q: metrics(backtest(df["spread"], q[0], q[1], lag=0).loc[:IS_END])["sharpe"])
    naive_res = {"params": naive, "is": metrics(backtest(df["spread"], *naive, lag=0).loc[:IS_END]),
                 "oos": metrics(backtest(df["spread"], *naive, lag=0).loc[oos.index[0]:]),
                 "oos_if_executed_next_day": metrics(backtest(df["spread"], *naive, lag=1).loc[oos.index[0]:])}
    ds = df["spread"].diff()
    autocorr = {k: round(float(ds.loc[a:b].autocorr()), 3) for k, (a, b) in
                {"2000-2010": ("2000", "2010"), "2011-2014": ("2011", "2014"), "2015-2026": ("2015", "2026")}.items()}
    # robustness: OOS sharpe for every grid point (is the chosen one an outlier?)
    oos_grid = {f"lb{int(r.lookback)}_z{r.entry_z}": metrics(backtest(df['spread'], int(r.lookback), r.entry_z)
                .loc[oos.index[0]:])["sharpe"] for r in grid.itertuples()}

    res = {"data": {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "n_days": len(df)},
           "stationarity_tests": tests, "chosen_params": {"lookback": lb, "entry_z": en, "exit_z": 0.5,
           "stop_z": 4.0, "max_hold": 60}, "in_sample_2000_2014": m_is, "out_of_sample_2015_on": m_oos,
           "oos_sharpe_vs_cost_per_side": sens, "execution_lag_sensitivity": lag_sens,
           "naive_same_day_backtest": naive_res, "lag1_autocorr_daily_spread_change": autocorr, "oos_sharpe_all_grid_points": oos_grid,
           "spread_max": round(float(df["spread"].max()), 2), "spread_max_date": str(df["spread"].idxmax().date()),
           "spread_min": round(float(df["spread"].min()), 2), "spread_min_date": str(df["spread"].idxmin().date())}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    grid.to_csv(out / "in_sample_grid.csv", index=False)

    # --- charts ---
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(df.index, df["spread"], lw=0.7, color="#333"); ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_ylabel("Brent − WTI ($/bbl)"); ax[0].set_title("Brent–WTI spread")
    ax[1].plot(bt.index, bt["z"], lw=0.5, color="#555")
    for y in (en, -en): ax[1].axhline(y, color="#c0392b", lw=0.7, ls="--")
    ax[1].set_ylabel(f"z-score ({lb}d)")
    bt0 = backtest(df["spread"], lb, en, lag=0)
    ax[2].plot(bt0.index, bt0["net"].cumsum(), color="#c0392b", lw=1, label="same-day fill (biased)")
    ax[2].plot(bt.index, bt["net"].cumsum(), color="#1f6f8b", label="next-day fill (realistic)")
    ax[2].legend(loc="upper left")
    ax[2].axvline(pd.Timestamp(IS_END), color="k", ls=":", lw=1); ax[2].text(pd.Timestamp(IS_END), ax[2].get_ylim()[1]*0.9, "  out-of-sample →")
    ax[2].set_ylabel("Cum. net P&L ($/1 spread)")
    fig.tight_layout(); fig.savefig(out / "spread_backtest.png", dpi=130)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
