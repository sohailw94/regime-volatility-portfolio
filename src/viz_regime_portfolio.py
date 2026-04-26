import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import PROCESSED_DIR


FIG_DIR = PROCESSED_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_results():
    daily = pd.read_parquet(PROCESSED_DIR / "regime_portfolio_engine_daily.parquet")
    return daily


def plot_equity_curve_with_stress_regimes(daily):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Original daily stress signal
    stress_raw = daily["regime_b"].astype(bool)

    # Persistence filter:
    # only show stress if it lasts at least 5 consecutive days
    min_regime_days = 5
    stress_filtered = (
        stress_raw
        .rolling(min_regime_days)
        .sum()
        .fillna(0)
        >= min_regime_days
    )

    # Expand the filtered region slightly backward so the shaded block
    # captures the start of the stress regime, not just day 5 onward
    stress_filtered = (
        stress_filtered
        .rolling(min_regime_days, min_periods=1)
        .max()
        .astype(bool)
    )

    # Build continuous stress blocks
    groups = (stress_filtered != stress_filtered.shift()).cumsum()

    regime_blocks = []
    for _, block in daily.groupby(groups):
        block_idx = block.index
        is_stress_block = bool(stress_filtered.loc[block_idx].iloc[0])

        if is_stress_block:
            regime_blocks.append(
                {
                    "start": block_idx[0],
                    "end": block_idx[-1],
                }
            )

    # Plot
    plt.figure(figsize=(14, 6))

    plt.plot(
        daily.index,
        daily["equity"],
        label="Equity",
        linewidth=1.8,
    )

    y_min = daily["equity"].min()
    y_max = daily["equity"].max()

    for block in regime_blocks:
        plt.axvspan(
            block["start"],
            block["end"],
            alpha=0.18,
            color="lightblue",
        )

    plt.title("Regime-Switching Portfolio Equity Curve with Persistent Stress Regimes")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend(["Equity", "Persistent Stress Regime"])
    plt.grid(True, alpha=0.35)
    plt.tight_layout()

    plt.savefig(FIG_DIR / "equity_curve_with_persistent_stress.png", dpi=150)
    plt.close()


def plot_drawdown(daily):
    plt.figure(figsize=(12, 5))
    daily["drawdown"].plot()
    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "drawdown.png", dpi=150)
    plt.close()


def plot_weights(daily):
    plt.figure(figsize=(12, 6))
    daily[["spy_weight", "qqq_weight", "uvix_weight"]].plot()
    plt.title("Dynamic Portfolio Weights: SPY / QQQ / UVIX")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "portfolio_weights.png", dpi=150)
    plt.close()


def plot_uvix_weight_vs_regime(daily):
    plt.figure(figsize=(12, 5))
    daily["uvix_weight"].plot(label="UVIX Weight")
    daily["regime_b"].mul(daily["uvix_weight"].max()).plot(
        alpha=0.4, label="Stress Regime Indicator"
    )
    plt.title("UVIX Allocation During Stress Regimes")
    plt.xlabel("Date")
    plt.ylabel("Weight / Regime")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "uvix_weight_vs_regime.png", dpi=150)
    plt.close()


def plot_costs(daily):
    cumulative_cost = daily["cost_return"].cumsum()

    plt.figure(figsize=(12, 5))
    cumulative_cost.plot()
    plt.title("Cumulative Transaction Cost Drag")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Cost Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cost_drag.png", dpi=150)
    plt.close()


def plot_regime_returns(daily):
    regime_returns = daily.groupby("regime_b")["net_return"].agg(
        ["mean", "std", "count"]
    )

    regime_returns["annualized_return"] = regime_returns["mean"] * 252
    regime_returns["annualized_vol"] = regime_returns["std"] * (252 ** 0.5)

    regime_returns[["annualized_return", "annualized_vol"]].plot(
        kind="bar", figsize=(8, 5)
    )

    plt.title("Performance by Regime")
    plt.xlabel("Regime: 0 = Risk-On, 1 = Stress")
    plt.ylabel("Annualized Return / Vol")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "performance_by_regime.png", dpi=150)
    plt.close()


def plot_return_distribution(daily):
    plt.figure(figsize=(10, 5))
    daily["net_return"].hist(bins=60)
    plt.title("Daily Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "return_distribution.png", dpi=150)
    plt.close()


def main():
    daily = load_results()

    plot_equity_curve_with_stress_regimes(daily)
    plot_drawdown(daily)
    plot_weights(daily)
    plot_uvix_weight_vs_regime(daily)
    plot_costs(daily)
    plot_regime_returns(daily)
    plot_return_distribution(daily)

    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()