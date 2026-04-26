import itertools
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR


def load_features(ticker: str = "spy") -> pd.DataFrame:
    path = PROCESSED_DIR / f"{ticker.lower()}_features.parquet"
    return pd.read_parquet(path).copy()


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def add_compression_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    base_features = [
        "rv_ratio_5_20",
        "rv_ratio_20_60",
        "rv_20_pct_252",
        "bb_width_20",
        "range_ratio",
    ]

    for col in base_features:
        df[f"z_{col}"] = zscore(df[col])

    df["compression_score"] = -(
        df["z_rv_ratio_5_20"]
        + df["z_rv_ratio_20_60"]
        + df["z_rv_20_pct_252"]
        + df["z_bb_width_20"]
        + df["z_range_ratio"]
    ) / len(base_features)

    return df


def add_triggers(
    df: pd.DataFrame,
    shock_std_mult: float,
    breakout_buffer: float,
) -> pd.DataFrame:
    df = df.copy()

    price_col = "adj close" if "adj close" in df.columns else "close"

    df["abs_ret_1d"] = df["ret_1d"].abs()
    df["abs_ret_20_mean"] = df["abs_ret_1d"].rolling(20).mean()
    df["abs_ret_20_std"] = df["abs_ret_1d"].rolling(20).std()

    df["shock_trigger"] = (
        df["abs_ret_1d"] > (df["abs_ret_20_mean"] + shock_std_mult * df["abs_ret_20_std"])
    ).astype(int)

    rolling_max_20 = df[price_col].rolling(20).max().shift(1)
    rolling_min_20 = df[price_col].rolling(20).min().shift(1)

    df["breakout_up"] = (df[price_col] > rolling_max_20 * (1 + breakout_buffer)).astype(int)
    df["breakout_down"] = (df[price_col] < rolling_min_20 * (1 - breakout_buffer)).astype(int)
    df["breakout_trigger"] = ((df["breakout_up"] == 1) | (df["breakout_down"] == 1)).astype(int)

    return df


def evaluate_signal(df: pd.DataFrame, signal_col: str) -> dict:
    subset = df[df[signal_col] == 1].copy()

    if subset.empty:
        return {
            "signal_count": 0,
            "signal_rate": 0.0,
            "avg_vol_expansion_ratio": np.nan,
            "median_vol_expansion_ratio": np.nan,
            "hit_rate": np.nan,
            "avg_future_abs_ret_5d": np.nan,
            "median_future_abs_ret_5d": np.nan,
        }

    return {
        "signal_count": len(subset),
        "signal_rate": len(subset) / len(df),
        "avg_vol_expansion_ratio": subset["vol_expansion_ratio"].mean(),
        "median_vol_expansion_ratio": subset["vol_expansion_ratio"].median(),
        "hit_rate": subset["target_binary"].mean(),
        "avg_future_abs_ret_5d": subset["future_abs_ret_5d"].mean(),
        "median_future_abs_ret_5d": subset["future_abs_ret_5d"].median(),
    }


def run_threshold_sweep(ticker: str = "spy") -> pd.DataFrame:
    df = load_features(ticker=ticker)
    df = add_compression_score(df)

    base_rate = {
        "signal_type": "base_rate",
        "compression_quantile": np.nan,
        "shock_std_mult": np.nan,
        "breakout_buffer": np.nan,
        "signal_count": len(df),
        "signal_rate": 1.0,
        "avg_vol_expansion_ratio": df["vol_expansion_ratio"].mean(),
        "median_vol_expansion_ratio": df["vol_expansion_ratio"].median(),
        "hit_rate": df["target_binary"].mean(),
        "avg_future_abs_ret_5d": df["future_abs_ret_5d"].mean(),
        "median_future_abs_ret_5d": df["future_abs_ret_5d"].median(),
    }

    compression_quantiles = [0.80, 0.85, 0.90, 0.95]
    shock_std_mults = [1.0, 1.5, 2.0]
    breakout_buffers = [0.0, 0.0025, 0.005, 0.01]

    results = [base_rate]

    for compression_quantile, shock_std_mult, breakout_buffer in itertools.product(
        compression_quantiles, shock_std_mults, breakout_buffers
    ):
        temp = add_triggers(
            df,
            shock_std_mult=shock_std_mult,
            breakout_buffer=breakout_buffer,
        ).copy()

        threshold = temp["compression_score"].quantile(compression_quantile)
        temp["compression_flag"] = (temp["compression_score"] >= threshold).astype(int)

        temp["entry_breakout"] = (
            (temp["compression_flag"] == 1) &
            (temp["breakout_trigger"] == 1)
        ).astype(int)

        temp["entry_shock"] = (
            (temp["compression_flag"] == 1) &
            (temp["shock_trigger"] == 1)
        ).astype(int)

        temp["entry_either"] = (
            (temp["compression_flag"] == 1) &
            (
                (temp["breakout_trigger"] == 1) |
                (temp["shock_trigger"] == 1)
            )
        ).astype(int)

        temp["entry_both"] = (
            (temp["compression_flag"] == 1) &
            (temp["breakout_trigger"] == 1) &
            (temp["shock_trigger"] == 1)
        ).astype(int)

        temp = temp.dropna(
            subset=[
                "vol_expansion_ratio",
                "target_binary",
                "future_abs_ret_5d",
                "shock_trigger",
                "breakout_trigger",
            ]
        ).copy()

        for signal_type in ["entry_breakout", "entry_shock", "entry_either", "entry_both"]:
            stats = evaluate_signal(temp, signal_type)
            stats["signal_type"] = signal_type
            stats["compression_quantile"] = compression_quantile
            stats["shock_std_mult"] = shock_std_mult
            stats["breakout_buffer"] = breakout_buffer
            results.append(stats)

    out = pd.DataFrame(results)

    out["hit_rate_lift"] = out["hit_rate"] / base_rate["hit_rate"]
    out["vol_expansion_lift"] = (
        out["avg_vol_expansion_ratio"] / base_rate["avg_vol_expansion_ratio"]
    )
    out["abs_move_lift"] = (
        out["avg_future_abs_ret_5d"] / base_rate["avg_future_abs_ret_5d"]
    )

    output_path = PROCESSED_DIR / f"{ticker.lower()}_threshold_sweep.parquet"
    out.to_parquet(output_path)

    return out


if __name__ == "__main__":
    results = run_threshold_sweep("spy")

    filtered = results[
        (results["signal_type"] != "base_rate") &
        (results["signal_count"] >= 25)
    ].copy()

    print("\n===== TOP BY HIT RATE =====")
    print(
        filtered.sort_values(
            ["hit_rate", "avg_vol_expansion_ratio", "signal_count"],
            ascending=[False, False, False]
        ).head(15).round(4)
    )

    print("\n===== TOP BY VOL EXPANSION RATIO =====")
    print(
        filtered.sort_values(
            ["avg_vol_expansion_ratio", "hit_rate", "signal_count"],
            ascending=[False, False, False]
        ).head(15).round(4)
    )

    print("\n===== TOP BY ABS MOVE =====")
    print(
        filtered.sort_values(
            ["avg_future_abs_ret_5d", "hit_rate", "signal_count"],
            ascending=[False, False, False]
        ).head(15).round(4)
    )