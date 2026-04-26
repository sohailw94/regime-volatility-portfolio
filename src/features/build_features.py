import numpy as np
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Returns the percentile rank of the most recent observation
    within a rolling window.
    """
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )


def build_features(ticker: str = "qqq", interval: str = "1d") -> pd.DataFrame:
    """
    Build volatility-compression and volatility-expansion features
    from OHLCV data.
    """
    input_path = RAW_DIR / f"{ticker.lower()}_{interval}.parquet"
    df = pd.read_parquet(input_path).copy()

    # Choose adjusted close if available
    price_col = "adj close" if "adj close" in df.columns else "close"

    # ----------------------------
    # 1. Returns
    # ----------------------------
    df["ret_1d"] = np.log(df[price_col] / df[price_col].shift(1))

    # ----------------------------
    # 2. Realized volatility
    # ----------------------------
    df["rv_5"] = df["ret_1d"].rolling(5).std() * np.sqrt(252)
    df["rv_20"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
    df["rv_60"] = df["ret_1d"].rolling(60).std() * np.sqrt(252)

    # ----------------------------
    # 3. Compression ratios
    # ----------------------------
    df["rv_ratio_5_20"] = df["rv_5"] / df["rv_20"]
    df["rv_ratio_20_60"] = df["rv_20"] / df["rv_60"]

    # ----------------------------
    # 4. Intraday range compression
    # ----------------------------
    df["range_pct"] = (df["high"] - df["low"]) / df[price_col]
    df["range_5"] = df["range_pct"].rolling(5).mean()
    df["range_20"] = df["range_pct"].rolling(20).mean()
    df["range_ratio"] = df["range_5"] / df["range_20"]

    # ----------------------------
    # 5. Price compression / band width
    # ----------------------------
    rolling_max_20 = df[price_col].rolling(20).max()
    rolling_min_20 = df[price_col].rolling(20).min()
    rolling_mean_20 = df[price_col].rolling(20).mean()
    df["bb_width_20"] = (rolling_max_20 - rolling_min_20) / rolling_mean_20

    # ----------------------------
    # 6. Historical percentile of current vol
    # ----------------------------
    df["rv_20_pct_252"] = rolling_percentile_rank(df["rv_20"], 252)

    # ----------------------------
    # 7. Forward volatility expansion target
    # ----------------------------
    future_ret = df["ret_1d"].shift(-1)
    df["future_rv_5"] = future_ret.rolling(5).std() * np.sqrt(252)
    df["vol_expansion_ratio"] = df["future_rv_5"] / df["rv_20"]
    df["target_binary"] = (df["vol_expansion_ratio"] > 1.5).astype(int)

    # Optional: future absolute move target
    df["future_abs_ret_5d"] = future_ret.rolling(5).sum().abs()

    # ----------------------------
    # 8. Clean output
    # ----------------------------
    feature_cols = [
        price_col,
        "volume",
        "ret_1d",
        "rv_5",
        "rv_20",
        "rv_60",
        "rv_ratio_5_20",
        "rv_ratio_20_60",
        "range_pct",
        "range_5",
        "range_20",
        "range_ratio",
        "bb_width_20",
        "rv_20_pct_252",
        "future_rv_5",
        "vol_expansion_ratio",
        "target_binary",
        "future_abs_ret_5d",
    ]

    out = df[feature_cols].dropna().copy()

    output_path = PROCESSED_DIR / f"{ticker.lower()}_features.parquet"
    out.to_parquet(output_path)

    return out


if __name__ == "__main__":
    features = build_features("qqq")
    print(features.tail())
    print("\nColumns:")
    print(features.columns.tolist())