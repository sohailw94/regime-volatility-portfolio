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
    Standard z-score.
    """
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def build_compression_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a composite compression score from:
    - short vs medium realized vol
    - medium vs long realized vol
    - percentile rank of current vol
    - band width
    - range compression

    Higher compression_score = more compressed regime.
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

    # Lower raw values imply more compression, so invert sign
    df["compression_score"] = -(
        df["z_rv_ratio_5_20"]
        + df["z_rv_ratio_20_60"]
        + df["z_rv_20_pct_252"]
        + df["z_bb_width_20"]
        + df["z_range_ratio"]
    ) / 5.0

    return df


def create_deciles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign compression deciles.
    0 = least compressed
    9 = most compressed
    """
    df = df.copy()

    # rank first to avoid qcut duplicate-edge issues
    ranked = df["compression_score"].rank(method="first")
    df["compression_decile"] = pd.qcut(
        ranked,
        10,
        labels=False,
    )

    return df


def create_compression_flag(df: pd.DataFrame, quantile: float = 0.80) -> pd.DataFrame:
    """
    Binary flag for compressed regimes.
    Default: top 20% most compressed observations.
    """
    df = df.copy()
    threshold = df["compression_score"].quantile(quantile)
    df["is_compressed"] = (df["compression_score"] >= threshold).astype(int)
    return df


def decile_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize forward behavior by compression decile.
    """
    required_cols = [
        "compression_decile",
        "future_rv_5",
        "vol_expansion_ratio",
        "target_binary",
        "future_abs_ret_5d",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    summary = (
        df.groupby("compression_decile")
        .agg(
            avg_future_rv_5=("future_rv_5", "mean"),
            median_future_rv_5=("future_rv_5", "median"),
            avg_vol_expansion_ratio=("vol_expansion_ratio", "mean"),
            median_vol_expansion_ratio=("vol_expansion_ratio", "median"),
            hit_rate=("target_binary", "mean"),
            avg_future_abs_ret_5d=("future_abs_ret_5d", "mean"),
            median_future_abs_ret_5d=("future_abs_ret_5d", "median"),
            obs_count=("target_binary", "count"),
        )
        .sort_index()
    )

    return summary


def compression_vs_non(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare compressed vs non-compressed regimes.
    """
    required_cols = [
        "is_compressed",
        "future_rv_5",
        "vol_expansion_ratio",
        "target_binary",
        "future_abs_ret_5d",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    summary = (
        df.groupby("is_compressed")
        .agg(
            avg_future_rv_5=("future_rv_5", "mean"),
            median_future_rv_5=("future_rv_5", "median"),
            avg_vol_expansion_ratio=("vol_expansion_ratio", "mean"),
            median_vol_expansion_ratio=("vol_expansion_ratio", "median"),
            hit_rate=("target_binary", "mean"),
            avg_future_abs_ret_5d=("future_abs_ret_5d", "mean"),
            median_future_abs_ret_5d=("future_abs_ret_5d", "median"),
            obs_count=("target_binary", "count"),
        )
        .sort_index()
    )

    summary.index = summary.index.map({0: "not_compressed", 1: "compressed"})
    return summary


def print_signal_diagnostics(df: pd.DataFrame) -> None:
    """
    Print quick diagnostic stats.
    """
    print("\n===== SIGNAL DIAGNOSTICS =====")
    print(f"Rows: {len(df):,}")
    print(f"Compression score mean: {df['compression_score'].mean():.4f}")
    print(f"Compression score std : {df['compression_score'].std():.4f}")
    print(f"Base hit rate         : {df['target_binary'].mean():.4%}")
    print(f"Avg vol expansion     : {df['vol_expansion_ratio'].mean():.4f}")
    print(f"Median vol expansion  : {df['vol_expansion_ratio'].median():.4f}")


def save_signal_dataset(df: pd.DataFrame, ticker: str = "spy") -> None:
    """
    Save signal-enriched dataset for later backtesting and plotting.
    """
    output_path = PROCESSED_DIR / f"{ticker.lower()}_signals.parquet"
    df.to_parquet(output_path)
    print(f"\nSaved signal dataset to: {output_path}")


def run_signal_validation(ticker: str = "spy", compression_quantile: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full signal validation pipeline.
    Returns:
    - enriched dataframe
    - decile summary
    - compressed vs non-compressed summary
    """
    df = load_features(ticker=ticker)
    df = build_compression_score(df)
    df = create_deciles(df)
    df = create_compression_flag(df, quantile=compression_quantile)

    deciles = decile_analysis(df)
    regime_summary = compression_vs_non(df)

    print_signal_diagnostics(df)

    print("\n===== DECILE ANALYSIS =====")
    print(deciles.round(4))

    print("\n===== COMPRESSED VS NON-COMPRESSED =====")
    print(regime_summary.round(4))

    top_decile = deciles.loc[9]
    bottom_decile = deciles.loc[0]

    print("\n===== TOP VS BOTTOM DECILE CHECK =====")
    print(f"Bottom decile avg vol expansion ratio : {bottom_decile['avg_vol_expansion_ratio']:.4f}")
    print(f"Top decile avg vol expansion ratio    : {top_decile['avg_vol_expansion_ratio']:.4f}")
    print(f"Bottom decile hit rate                : {bottom_decile['hit_rate']:.4%}")
    print(f"Top decile hit rate                   : {top_decile['hit_rate']:.4%}")

    save_signal_dataset(df, ticker=ticker)

    return df, deciles, regime_summary


if __name__ == "__main__":
    run_signal_validation(ticker="spy", compression_quantile=0.80)