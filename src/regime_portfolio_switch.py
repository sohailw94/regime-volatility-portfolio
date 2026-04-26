from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import PROCESSED_DIR
from src.features.build_features import build_features
from src.signals.hmm_short_term_continuation import run_short_term_continuation_signal
from src.config import PROCESSED_DIR

# =========================================================
# CONFIG
# =========================================================

@dataclass
class RegimePortfolioConfig:
    initial_capital: float = 100_000.0

    # Risk-on allocation
    max_spy_weight: float = 0.50
    max_qqq_weight: float = 0.50

    # Convex hedge allocation
    max_uvix_weight: float = 0.30
    min_uvix_weight: float = 0.00

    # Regime thresholds
    min_b_votes_for_uvix: int = 2
    vvix_lead_threshold: float = 0.03
    conviction_threshold: float = 0.35

    # Cost model on turnover
    normal_cost_bps: float = 2.0
    stress_cost_bps: float = 12.0

    # Risk controls
    max_portfolio_drawdown: float = 0.15
    uvix_daily_stop: float = -0.08

    # Smoothing
    weight_smoothing_span: int = 5

    # Walk-forward
    train_days: int = 504
    test_days: int = 126

    save_prefix: str = "regime_portfolio_engine"


# =========================================================
# ENGINE
# =========================================================

class RegimePortfolioEngine:
    def __init__(self, config: Optional[RegimePortfolioConfig] = None):
        self.config = config or RegimePortfolioConfig()

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def rank(self, s: pd.Series, window: int = 126) -> pd.Series:
        s = s.astype(float)
        lo = s.rolling(window, min_periods=20).min()
        hi = s.rolling(window, min_periods=20).max()
        return ((s - lo) / (hi - lo + 1e-9)).clip(0, 1)

    def download_close(self, ticker: str, start: str) -> pd.Series:
        data = yf.download(ticker, start=start, progress=False, auto_adjust=False)

        if data.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]

        close = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
        close.name = f"{ticker.lower()}_close"
        close.index = pd.to_datetime(close.index)

        return close

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        build_features("spy")
        base, _, _ = run_short_term_continuation_signal("spy")

        overlay = pd.read_parquet(PROCESSED_DIR / "uvix_overlay_market_data.parquet")
        df = base.join(overlay, how="inner").copy()

        if "spy_close" not in df.columns:
            raise KeyError("spy_close missing from overlay data")

        start = str(df.index.min().date())

        qqq = self.download_close("QQQ", start)
        df = df.join(qqq, how="left")

        for col in ["spy_close", "qqq_close", "uvix_close"]:
            df[col] = df[col].ffill()

        df["spy_ret"] = df["spy_close"].pct_change().fillna(0.0)
        df["qqq_ret"] = df["qqq_close"].pct_change().fillna(0.0)
        df["uvix_ret"] = df["uvix_close"].pct_change().fillna(0.0)

        return df.dropna(subset=["spy_close", "qqq_close", "uvix_close"])

    # -----------------------------------------------------
    # Regime features
    # -----------------------------------------------------
    def build_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["vvix_lead"] = (
            df["vvix_close"].pct_change(2)
            - df["vix_close"].pct_change(2)
        )

        df["crash_regime"] = (
            df["vix_close"] > df["vix_close"].rolling(252).quantile(0.60)
        ).astype(int)

        vix_momo = self.rank(df["vix_close"].pct_change(2).fillna(0.0))
        vvix_momo = self.rank(df["vvix_close"].pct_change(2).fillna(0.0))
        downside = self.rank((-df["spy_close"].pct_change(3)).fillna(0.0))

        df["conviction"] = (
            0.40 * vix_momo
            + 0.35 * vvix_momo
            + 0.25 * downside
        ).clip(0, 1)

        df["entry_b"] = (
            (df["spy_close"].pct_change(3) < -0.02)
            | (df["vix_close"].pct_change(2) > 0.06)
            | (df["vvix_close"].pct_change(1) > 0.07)
        ).astype(int)

        df["b_votes"] = 0
        df["b_votes"] += (df["crash_regime"] == 1).astype(int)
        df["b_votes"] += (df["vvix_lead"] > self.config.vvix_lead_threshold).astype(int)
        df["b_votes"] += (df["conviction"] > self.config.conviction_threshold).astype(int)

        # IMPORTANT FIX:
        # Regime B should activate from stress votes, not entry_b AND votes.
        df["regime_b"] = (
            df["b_votes"] >= self.config.min_b_votes_for_uvix
        ).astype(int)

        raw_uvix_score = (
            0.40 * df["crash_regime"]
            + 0.35 * (df["b_votes"] / 3.0)
            + 0.25 * df["conviction"]
        ).clip(0, 1)

        df["uvix_score"] = np.where(df["regime_b"] == 1, raw_uvix_score, 0.0)

        return df

    # -----------------------------------------------------
    # Target weights
    # -----------------------------------------------------
    def compute_target_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["target_uvix"] = (
            self.config.min_uvix_weight
            + (df["uvix_score"])
            * (self.config.max_uvix_weight - self.config.min_uvix_weight)
        )
        df["target_uvix"] = np.where(
                df["regime_b"] == 1,
                np.maximum(df["target_uvix"], 0.08),  # 8% floor
                df["target_uvix"]
            )
        equity_weight = 1.0 - df["target_uvix"]

        df["target_spy"] = equity_weight * self.config.max_spy_weight
        df["target_qqq"] = equity_weight * self.config.max_qqq_weight

        for col in ["target_spy", "target_qqq", "target_uvix"]:
            df[col] = (
                df[col]
                .ewm(span=self.config.weight_smoothing_span, adjust=False)
                .mean()
                .clip(0, 1)
            )

        gross = df[["target_spy", "target_qqq", "target_uvix"]].sum(axis=1)
        scale = np.where(gross > 1.0, 1.0 / gross, 1.0)

        df["target_spy"] *= scale
        df["target_qqq"] *= scale
        df["target_uvix"] *= scale

        return df

    # -----------------------------------------------------
    # Costs
    # -----------------------------------------------------
    def dynamic_cost_rate(self, row: pd.Series) -> float:
        if int(row["crash_regime"]) == 1:
            return self.config.stress_cost_bps / 10_000.0
        return self.config.normal_cost_bps / 10_000.0

    # -----------------------------------------------------
    # Backtest
    # -----------------------------------------------------
    def run_on_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.build_regime_features(df)
        df = self.compute_target_weights(df)

        equity = self.config.initial_capital
        peak_equity = equity
        kill_switch = False

        prev_weights = {
            "spy": 0.0,
            "qqq": 0.0,
            "uvix": 0.0,
        }

        records = []

        prev_vol_scale = 1.0

        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]

            target_weights = {
                "spy": float(row["target_spy"]),
                "qqq": float(row["target_qqq"]),
                "uvix": float(row["target_uvix"]),
            }

            drawdown = equity / peak_equity - 1.0

            if drawdown <= -self.config.max_portfolio_drawdown:
                kill_switch = True

            # IMPORTANT FIX:
            # Kill switch is recoverable when stress regime is gone.
            if kill_switch:
                if int(row["crash_regime"]) == 0:
                    kill_switch = False
                else:
                    target_weights = {
                        "spy": 0.0,
                        "qqq": 0.0,
                        "uvix": 0.0,
                    }

            if drawdown < -0.5:
                target_weights["spy"] *= 0.7
                target_weights["qqq"] *= 0.75

            recent_port_ret = (
                prev_weights["spy"] * df["spy_ret"].iloc[max(0, i-20):i].std()
                + prev_weights["qqq"] * df["qqq_ret"].iloc[max(0, i-20):i].std()
                + prev_weights["uvix"] * df["uvix_ret"].iloc[max(0, i-20):i].std()
            )


            if i > 20:
                recent_port_ret = (
                    prev_weights["spy"] * df["spy_ret"].iloc[max(0, i-20):i]
                    + prev_weights["qqq"] * df["qqq_ret"].iloc[max(0, i-20):i]
                    + prev_weights["uvix"] * df["uvix_ret"].iloc[max(0, i-20):i]
                )

                realized_vol = recent_port_ret.std() * np.sqrt(252)
                target_vol = 0.12

                raw_vol_scale = min(1.0, target_vol / realized_vol) if realized_vol > 0 else 1.0

                # Smooth scaling to reduce turnover
                vol_scale = 0.85 * prev_vol_scale + 0.15 * raw_vol_scale

                target_weights["spy"] *= vol_scale
                target_weights["qqq"] *= vol_scale
                target_weights["uvix"] *= vol_scale

                prev_vol_scale = vol_scale


            # if i > 20 and recent_port_ret > 0:
            #     realized_vol = recent_port_ret * np.sqrt(252)
            #     target_vol = 0.12
            #     vol_scale = min(1.0, target_vol / realized_vol)

            #     target_weights["spy"] *= vol_scale
            #     target_weights["qqq"] *= vol_scale
            #     target_weights["uvix"] *= vol_scale

            # if row["regime_b"] == 1:
            #     stress_intensity = float(row["uvix_score"])

            #     equity_scale = 1.0 - 0.8 * stress_intensity
            #     equity_scale = max(0.2, equity_scale)

            #     target_weights["spy"] *= equity_scale
            #     target_weights["qqq"] *= equity_scale
            # UVIX daily failure stop
            if row["uvix_ret"] <= self.config.uvix_daily_stop:
                target_weights["uvix"] = 0.0

                # do not instantly reallocate failed UVIX risk into equities
                gross = sum(target_weights.values())
                if gross > 1.0:
                    target_weights = {k: v / gross for k, v in target_weights.items()}

            if row["regime_b"] == 1 and not kill_switch:
                target_weights["uvix"] = max(target_weights["uvix"], 0.08)

                # Keep total portfolio gross exposure <= 1
                gross = sum(target_weights.values())
                if gross > 1.0:
                    target_weights = {k: v / gross for k, v in target_weights.items()}

            turnover = (
                abs(target_weights["spy"] - prev_weights["spy"])
                + abs(target_weights["qqq"] - prev_weights["qqq"])
                + abs(target_weights["uvix"] - prev_weights["uvix"])
            )
            if turnover < 0.02:
                turnover = 0.0
            cost_rate = self.dynamic_cost_rate(row)
            cost_return = turnover * cost_rate

            gross_return = (
                prev_weights["spy"] * row["spy_ret"]
                + prev_weights["qqq"] * row["qqq_ret"]
                + prev_weights["uvix"] * row["uvix_ret"]
            )

            net_return = gross_return - cost_return

            equity *= 1.0 + net_return
            peak_equity = max(peak_equity, equity)
            drawdown = equity / peak_equity - 1.0

            records.append(
                {
                    "date": date,
                    "equity": equity,
                    "gross_return": gross_return,
                    "cost_return": cost_return,
                    "net_return": net_return,
                    "drawdown": drawdown,

                    "spy_weight": prev_weights["spy"],
                    "qqq_weight": prev_weights["qqq"],
                    "uvix_weight": prev_weights["uvix"],

                    "target_spy": target_weights["spy"],
                    "target_qqq": target_weights["qqq"],
                    "target_uvix": target_weights["uvix"],

                    "turnover": turnover,
                    "cost_rate": cost_rate,

                    "regime_b": int(row["regime_b"]),
                    "b_votes": int(row["b_votes"]),
                    "conviction": float(row["conviction"]),
                    "crash_regime": int(row["crash_regime"]),

                    "uvix_score": float(row["uvix_score"]),
                    "uvix_ret": float(row["uvix_ret"]),
                    "spy_ret": float(row["spy_ret"]),
                    "qqq_ret": float(row["qqq_ret"]),
                    "kill_switch": int(kill_switch),
                }
            )

            prev_weights = target_weights

        result = pd.DataFrame(records).set_index("date")
        summary = self.summarize(result)

        return result, summary

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    def summarize(self, result: pd.DataFrame) -> pd.Series:
        r = result["net_return"]

        sharpe = 0.0
        if r.std() > 1e-9:
            sharpe = r.mean() / r.std() * np.sqrt(252)

        return pd.Series(
            {
                "final_equity": result["equity"].iloc[-1],
                "total_return": result["equity"].iloc[-1] / self.config.initial_capital - 1.0,
                "sharpe": sharpe,
                "max_drawdown": result["drawdown"].min(),
                "annualized_vol": r.std() * np.sqrt(252),
                "avg_spy_weight": result["spy_weight"].mean(),
                "avg_qqq_weight": result["qqq_weight"].mean(),
                "avg_uvix_weight": result["uvix_weight"].mean(),
                "max_uvix_weight": result["uvix_weight"].max(),
                "avg_turnover": result["turnover"].mean(),
                "total_cost_return": result["cost_return"].sum(),
                "regime_b_days": result["regime_b"].sum(),
                "kill_switch_days": result["kill_switch"].sum(),
            }
        )

    # -----------------------------------------------------
    # Public run
    # -----------------------------------------------------
    def run(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.load_data()
        result, summary = self.run_on_df(df)

        result.to_parquet(PROCESSED_DIR / f"{self.config.save_prefix}_daily.parquet")

        print("\n===== REGIME PORTFOLIO ENGINE =====")
        print(summary.round(4))

        return result, summary


# =========================================================
# WALK-FORWARD
# =========================================================

def walk_forward_test(
    engine: RegimePortfolioEngine,
    df: pd.DataFrame,
    train_days: int = 504,
    test_days: int = 126,
) -> pd.DataFrame:
    results = []

    start = 0

    while start + train_days + test_days < len(df):
        test = df.iloc[start + train_days:start + train_days + test_days].copy()

        result, summary = engine.run_on_df(test)

        results.append(
            {
                "train_start": df.index[start],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "return": summary["total_return"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "avg_spy_weight": summary["avg_spy_weight"],
                "avg_qqq_weight": summary["avg_qqq_weight"],
                "avg_uvix_weight": summary["avg_uvix_weight"],
                "max_uvix_weight": summary["max_uvix_weight"],
                "regime_b_days": summary["regime_b_days"],
                "kill_switch_days": summary["kill_switch_days"],
                "total_cost_return": summary["total_cost_return"],
            }
        )

        start += test_days

    return pd.DataFrame(results)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    engine = RegimePortfolioEngine()

    df = engine.load_data()
    daily, summary = engine.run_on_df(df)

    print("\n===== FULL SAMPLE =====")
    print(summary.round(4))

    print("\nRecent weights:")
    print(
        daily[
            [
                "spy_weight",
                "qqq_weight",
                "uvix_weight",
                "target_spy",
                "target_qqq",
                "target_uvix",
                "regime_b",
                "b_votes",
                "conviction",
                "drawdown",
                "kill_switch",
            ]
        ].tail(20)
    )

    print("\n===== WALK FORWARD =====")
    wf = walk_forward_test(
        engine,
        df,
        train_days=engine.config.train_days,
        test_days=engine.config.test_days,
    )

    pd.set_option("display.max_columns", None)
    print(wf.round(4))
daily.to_parquet(PROCESSED_DIR / "regime_portfolio_engine_daily.parquet")
print("Saved daily results")