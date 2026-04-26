import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM

from src.config import PROCESSED_DIR


def load_features(ticker: str = "qqq") -> pd.DataFrame:
    path = PROCESSED_DIR / f"{ticker.lower()}_features.parquet"
    return pd.read_parquet(path).copy()


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def prepare_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "ret_1d",
        "rv_5",
        "rv_20",
        "rv_ratio_5_20",
        "rv_ratio_20_60",
        "range_ratio",
        "bb_width_20",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df["abs_ret_1d"] = df["ret_1d"].abs()

    hmm_cols = [
        "rv_5",
        "rv_20",
        "rv_ratio_5_20",
        "rv_ratio_20_60",
        "abs_ret_1d",
        "range_ratio",
        "bb_width_20",
    ]

    for col in hmm_cols:
        df[f"z_{col}"] = zscore(df[col])

    z_cols = [f"z_{col}" for col in hmm_cols]
    df = df.dropna(subset=z_cols).copy()
    return df


def fit_hmm(df: pd.DataFrame, n_states: int = 3, random_state: int = 42) -> tuple[pd.DataFrame, GaussianHMM]:
    df = df.copy()

    feature_cols = [
        "z_rv_5",
        "z_rv_20",
        "z_rv_ratio_5_20",
        "z_rv_ratio_20_60",
        "z_abs_ret_1d",
        "z_range_ratio",
        "z_bb_width_20",
    ]

    X = df[feature_cols].values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=random_state,
    )
    model.fit(X)

    hidden_states = model.predict(X)
    state_probs = model.predict_proba(X)

    df["hidden_state"] = hidden_states
    for i in range(n_states):
        df[f"state_prob_{i}"] = state_probs[:, i]

    return df, model


def summarize_states(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("hidden_state")
        .agg(
            avg_rv_5=("rv_5", "mean"),
            avg_rv_20=("rv_20", "mean"),
            avg_rv_ratio_5_20=("rv_ratio_5_20", "mean"),
            avg_abs_ret_1d=("abs_ret_1d", "mean"),
            avg_future_rv_5=("future_rv_5", "mean"),
            avg_vol_expansion_ratio=("vol_expansion_ratio", "mean"),
            hit_rate=("target_binary", "mean"),
            count=("target_binary", "count"),
        )
        .sort_values("avg_rv_20")
    )


def identify_high_vol_state(state_summary: pd.DataFrame) -> int:
    return int(state_summary["avg_rv_20"].idxmax())


def add_trade_features(df: pd.DataFrame, breakout_window: int = 20, breakout_buffer: float = 0.0) -> pd.DataFrame:
    df = df.copy()

    price_col = "adj close" if "adj close" in df.columns else "close"

    rolling_high = df[price_col].rolling(breakout_window).max().shift(1)
    rolling_low = df[price_col].rolling(breakout_window).min().shift(1)

    df["breakout_up"] = (df[price_col] > rolling_high * (1 + breakout_buffer)).astype(int)
    df["breakout_down"] = (df[price_col] < rolling_low * (1 - breakout_buffer)).astype(int)

    df["ma_10"] = df[price_col].rolling(10).mean()
    df["ma_20"] = df[price_col].rolling(20).mean()
    df["ma_50"] = df[price_col].rolling(50).mean()

    df["uptrend_flag"] = (
        (df[price_col] > df["ma_10"]) &
        (df["ma_10"] > df["ma_20"]) &
        (df["ma_20"] > df["ma_50"])
    ).astype(int)

    df["downtrend_flag"] = (
        (df[price_col] < df["ma_10"]) &
        (df["ma_10"] < df["ma_20"]) &
        (df["ma_20"] < df["ma_50"])
    ).astype(int)

    df["positive_day"] = (df["ret_1d"] > 0).astype(int)
    df["negative_day"] = (df["ret_1d"] < 0).astype(int)

    # avoid entering when short-term vol is already fading too much
    df["vol_cont_flag"] = (df["rv_ratio_5_20"] >= 0.95).astype(int)

    # optional shock strength
    df["abs_ret_20_mean"] = df["abs_ret_1d"].rolling(20).mean()
    df["impulse_flag"] = (df["abs_ret_1d"] > df["abs_ret_20_mean"]).astype(int)

    return df


def add_entry_signals(
    df: pd.DataFrame,
    high_vol_state: int,
    prob_threshold: float = 0.70
) -> pd.DataFrame:
    df = df.copy()

    prob_col = f"state_prob_{high_vol_state}"
    df["high_vol_prob"] = df[prob_col]
    df["high_vol_flag"] = (df["high_vol_prob"] >= prob_threshold).astype(int)

    df["long_entry"] = (
        (df["high_vol_flag"] == 1) &
        (df["uptrend_flag"] == 1) &
        (df["breakout_up"] == 1) &
        (df["positive_day"] == 1) &
        (df["vol_cont_flag"] == 1) &
        (df["impulse_flag"] == 1)
    ).astype(int)

    df["short_entry"] = (
        (df["high_vol_flag"] == 1) &
        (df["downtrend_flag"] == 1) &
        (df["breakout_down"] == 1) &
        (df["negative_day"] == 1) &
        (df["vol_cont_flag"] == 1) &
        (df["impulse_flag"] == 1)
    ).astype(int)

    df["entry_signal"] = np.select(
        [df["long_entry"] == 1, df["short_entry"] == 1],
        [1, -1],
        default=0
    )

    return df


def summarize_signal_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name, subset in [
        ("all_rows", df),
        ("signal_rows", df[df["entry_signal"] != 0]),
        ("long_rows", df[df["long_entry"] == 1]),
        ("short_rows", df[df["short_entry"] == 1]),
    ]:
        if subset.empty:
            rows.append({
                "subset": name,
                "count": 0,
                "avg_future_rv_5": np.nan,
                "avg_vol_expansion_ratio": np.nan,
                "hit_rate": np.nan,
                "avg_future_abs_ret_5d": np.nan,
            })
        else:
            rows.append({
                "subset": name,
                "count": len(subset),
                "avg_future_rv_5": subset["future_rv_5"].mean(),
                "avg_vol_expansion_ratio": subset["vol_expansion_ratio"].mean(),
                "hit_rate": subset["target_binary"].mean(),
                "avg_future_abs_ret_5d": subset["future_abs_ret_5d"].mean(),
            })

    return pd.DataFrame(rows)


def run_short_term_continuation_signal(
    ticker: str = "qqq",
    n_states: int = 3,
    prob_threshold: float = 0.60,
    breakout_window: int = 20,
    breakout_buffer: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_features(ticker)
    df = prepare_hmm_features(df)
    df, _ = fit_hmm(df, n_states=n_states)

    state_summary = summarize_states(df)
    high_vol_state = identify_high_vol_state(state_summary)

    df = add_trade_features(
        df,
        breakout_window=breakout_window,
        breakout_buffer=breakout_buffer,
    )

    df = add_entry_signals(
        df,
        high_vol_state=high_vol_state,
        prob_threshold=prob_threshold,
    )

    df = df.dropna(
        subset=[
            "ma_10", "ma_20", "ma_50",
            "high_vol_prob", "entry_signal",
            "abs_ret_20_mean"
        ]
    ).copy()

    signal_summary = summarize_signal_rows(df)

    output_path = PROCESSED_DIR / f"{ticker.lower()}_hmm_short_term_signals.parquet"
    df.to_parquet(output_path)

    print("\n===== HMM STATE SUMMARY =====")
    print(state_summary.round(4))
    print(f"\nChosen high-vol state: {high_vol_state}")
    print(f"Probability threshold: {prob_threshold:.2f}")

    print("\n===== SIGNAL SUMMARY =====")
    print(signal_summary.round(4))

    print("\n===== ENTRY COUNTS =====")
    print(df["entry_signal"].value_counts(dropna=False).sort_index())

    print(f"\nSaved short-term continuation signals to: {output_path}")

    return df, state_summary, signal_summary


if __name__ == "__main__":
    run_short_term_continuation_signal(
        ticker="qqq",
        n_states=3,
        prob_threshold=0.60,
        breakout_window=20,
        breakout_buffer=0.0,
    )