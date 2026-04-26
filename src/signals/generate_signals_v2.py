import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR


def load_features(ticker: str = "spy") -> pd.DataFrame:
    """
    Load processed feature dataset.
    """
    path = PROCESSED_DIR / f"{ticker.lower()}_features.parquet"
    df = pd.read_parquet(path).copy()
    return df


def zscore(series: pd.Series) -> pd.Series:
    """
    Standard z-score with safe fallback.
    """
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def add_compression_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite compression score.
    Higher score = more compressed.
    """
    df = df.copy()

    base_features = [
        "rv_ratio_5_20",
        "rv_ratio_20_60",
        "rv_20_pct_252",
        "bb_width_20",
        "range_ratio",
    ]

    for col in base_features:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    for col in base_features:
        df[f"z_{col}"] = zscore(df[col])

    # lower raw feature values mean more compression, so invert
    df["compression_score"] = -(
        df["z_rv_ratio_5_20"]
        + df["z_rv_ratio_20_60"]
        + df["z_rv_20_pct_252"]
        + df["z_bb_width_20"]
        + df["z_range_ratio"]
    ) / len(base_features)

    return df


def add_price_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-based trigger features using only real price data.
    """
    df = df.copy()

    price_col = "adj close" if "adj close" in df.columns else "close"

    if price_col not in df.columns:
        raise KeyError(f"Missing required price column: {price_col}")

    # absolute return context
    df["abs_ret_1d"] = df["ret_1d"].abs()
    df["abs_ret_20_mean"] = df["abs_ret_1d"].rolling(20).mean()
    df["abs_ret_20_std"] = df["abs_ret_1d"].rolling(20).std()

    # shock trigger: today is meaningfully larger than recent average
    df["shock_trigger"] = (
        df["abs_ret_1d"] > (df["abs_ret_20_mean"] + 0.5 * df["abs_ret_20_std"])
    ).astype(int)

    # breakout trigger: price breaks 20-day range
    rolling_max_20 = df[price_col].rolling(20).max()
    rolling_min_20 = df[price_col].rolling(20).min()

    df["breakout_up"] = (df[price_col] > rolling_max_20.shift(1)).astype(int)
    df["breakout_down"] = (df[price_col] < rolling_min_20.shift(1)).astype(int)
    df["breakout_trigger"] = ((df["breakout_up"] == 1) | (df["breakout_down"] == 1)).astype(int)

    # directional label for later backtest
    df["breakout_direction"] = np.select(
        [df["breakout_up"] == 1, df["breakout_down"] == 1],
        [1, -1],
        default=0
    )

    return df


def add_regime_flags(
    df: pd.DataFrame,
    compression_quantile: float = 0.80
) -> pd.DataFrame:
    """
    Create binary compression regime flag.
    """
    df = df.copy()

    threshold = df["compression_score"].quantile(compression_quantile)
    df["compression_flag"] = (df["compression_score"] >= threshold).astype(int)

    return df


def add_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build v2 entry signals:
    - compression + breakout
    - compression + shock
    - compression + either trigger
    """
    df = df.copy()

    df["entry_breakout"] = (
        (df["compression_flag"] == 1) &
        (df["breakout_trigger"] == 1)
    ).astype(int)

    df["entry_shock"] = (
        (df["compression_flag"] == 1) &
        (df["shock_trigger"] == 1)
    ).astype(int)

    df["entry_either"] = (
        (df["compression_flag"] == 1) &
        (
            (df["breakout_trigger"] == 1) |
            (df["shock_trigger"] == 1)
        )
    ).astype(int)

    return df


def summarize_subset(df: pd.DataFrame, signal_col: str) -> pd.Series:
    """
    Summarize forward behavior for rows where a given signal is active.
    """
    if signal_col not in df.columns:
        raise KeyError(f"Missing signal column: {signal_col}")

    subset = df[df[signal_col] == 1].copy()

    if subset.empty:
        return pd.Series({
            "signal_count": 0,
            "signal_rate": 0.0,
            "avg_future_rv_5": np.nan,
            "median_future_rv_5": np.nan,
            "avg_vol_expansion_ratio": np.nan,
            "median_vol_expansion_ratio": np.nan,
            "hit_rate": np.nan,
            "avg_future_abs_ret_5d": np.nan,
            "median_future_abs_ret_5d": np.nan,
        })

    return pd.Series({
        "signal_count": len(subset),
        "signal_rate": len(subset) / len(df),
        "avg_future_rv_5": subset["future_rv_5"].mean(),
        "median_future_rv_5": subset["future_rv_5"].median(),
        "avg_vol_expansion_ratio": subset["vol_expansion_ratio"].mean(),
        "median_vol_expansion_ratio": subset["vol_expansion_ratio"].median(),
        "hit_rate": subset["target_binary"].mean(),
        "avg_future_abs_ret_5d": subset["future_abs_ret_5d"].mean(),
        "median_future_abs_ret_5d": subset["future_abs_ret_5d"].median(),
    })


def compare_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare base rate vs triggered signal subsets.
    """
    base = pd.Series({
        "signal_count": len(df),
        "signal_rate": 1.0,
        "avg_future_rv_5": df["future_rv_5"].mean(),
        "median_future_rv_5": df["future_rv_5"].median(),
        "avg_vol_expansion_ratio": df["vol_expansion_ratio"].mean(),
        "median_vol_expansion_ratio": df["vol_expansion_ratio"].median(),
        "hit_rate": df["target_binary"].mean(),
        "avg_future_abs_ret_5d": df["future_abs_ret_5d"].mean(),
        "median_future_abs_ret_5d": df["future_abs_ret_5d"].median(),
    })

    breakout = summarize_subset(df, "entry_breakout")
    shock = summarize_subset(df, "entry_shock")
    either = summarize_subset(df, "entry_either")

    summary = pd.DataFrame({
        "base_rate": base,
        "compression_plus_breakout": breakout,
        "compression_plus_shock": shock,
        "compression_plus_either": either,
    }).T

    return summary


def save_output(df: pd.DataFrame, ticker: str = "spy") -> None:
    """
    Save enriched signal dataset for later backtest.
    """
    output_path = PROCESSED_DIR / f"{ticker.lower()}_signals_v2.parquet"
    df.to_parquet(output_path)
    print(f"\nSaved v2 signal dataset to: {output_path}")


def run_signal_validation_v2(
    ticker: str = "spy",
    compression_quantile: float = 0.80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full v2 signal validation pipeline.
    """
    df = load_features(ticker=ticker)
    df = add_compression_score(df)
    df = add_price_context(df)
    df = add_regime_flags(df, compression_quantile=compression_quantile)
    df = add_entry_signals(df)

    # drop rows with fresh rolling NaNs from trigger calculations
    required_cols = [
        "compression_score",
        "compression_flag",
        "shock_trigger",
        "breakout_trigger",
        "entry_breakout",
        "entry_shock",
        "entry_either",
        "future_rv_5",
        "vol_expansion_ratio",
        "target_binary",
        "future_abs_ret_5d",
    ]
    df = df.dropna(subset=required_cols).copy()

    summary = compare_signals(df)

    print("\n===== V2 SIGNAL DIAGNOSTICS =====")
    print(f"Rows: {len(df):,}")
    print(f"Base hit rate        : {df['target_binary'].mean():.4%}")
    print(f"Base vol expansion   : {df['vol_expansion_ratio'].mean():.4f}")
    print(f"Compression threshold: {df['compression_score'].quantile(compression_quantile):.4f}")

    print("\n===== SIGNAL COMPARISON =====")
    print(summary.round(4))

    save_output(df, ticker=ticker)

    return df, summary


if __name__ == "__main__":
    run_signal_validation_v2(ticker="spy", compression_quantile=0.80)