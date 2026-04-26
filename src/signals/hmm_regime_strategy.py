import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM

from src.config import PROCESSED_DIR


def load_features(ticker: str = "spy") -> pd.DataFrame:
    path = PROCESSED_DIR / f"{ticker.lower()}_features.parquet"
    return pd.read_parquet(path).copy()


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def prepare_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the observable feature set used by the HMM.
    """
    df = df.copy()

    required_cols = [
        "ret_1d",
        "rv_5",
        "rv_20",
        "rv_60",
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


def fit_hmm(
    df: pd.DataFrame,
    n_states: int = 3,
    random_state: int = 42
) -> tuple[pd.DataFrame, GaussianHMM, list[str]]:
    """
    Fit Gaussian HMM and attach hidden states and state probabilities.
    """
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

    return df, model, feature_cols


def summarize_states(df: pd.DataFrame, n_states: int = 3) -> pd.DataFrame:
    """
    Summarize hidden states to determine which regime each state represents.
    """
    summary = (
        df.groupby("hidden_state")
        .agg(
            avg_rv_5=("rv_5", "mean"),
            avg_rv_20=("rv_20", "mean"),
            avg_rv_ratio_5_20=("rv_ratio_5_20", "mean"),
            avg_abs_ret_1d=("abs_ret_1d", "mean"),
            avg_range_ratio=("range_ratio", "mean"),
            avg_bb_width_20=("bb_width_20", "mean"),
            avg_future_rv_5=("future_rv_5", "mean"),
            avg_vol_expansion_ratio=("vol_expansion_ratio", "mean"),
            hit_rate=("target_binary", "mean"),
            count=("target_binary", "count"),
        )
        .sort_values("avg_rv_20")
    )

    return summary


def identify_high_vol_state(state_summary: pd.DataFrame) -> int:
    """
    Choose the state with the highest average rv_20 as the high-vol state.
    """
    return int(state_summary["avg_rv_20"].idxmax())


def add_breakout_logic(
    df: pd.DataFrame,
    breakout_window: int = 20,
    breakout_buffer: float = 0.0025
) -> pd.DataFrame:
    """
    Add directional breakout rules.
    """
    df = df.copy()

    price_col = "adj close" if "adj close" in df.columns else "close"

    rolling_max = df[price_col].rolling(breakout_window).max().shift(1)
    rolling_min = df[price_col].rolling(breakout_window).min().shift(1)

    df["breakout_up"] = (df[price_col] > rolling_max * (1 + breakout_buffer)).astype(int)
    df["breakout_down"] = (df[price_col] < rolling_min * (1 - breakout_buffer)).astype(int)

    return df


def add_hmm_trade_signals(
    df: pd.DataFrame,
    high_vol_state: int,
    prob_threshold: float = 0.60
) -> pd.DataFrame:
    """
    Trade only when HMM says high-vol state probability is strong.
    """
    df = df.copy()

    prob_col = f"state_prob_{high_vol_state}"
    if prob_col not in df.columns:
        raise KeyError(f"Missing probability column: {prob_col}")

    df["high_vol_prob"] = df[prob_col]
    df["high_vol_flag"] = (df["high_vol_prob"] >= prob_threshold).astype(int)

    # continuation filter: short-term vol above medium-term baseline
    df["vol_continuation_flag"] = (df["rv_ratio_5_20"] > 1.05).astype(int)

    df["long_entry"] = (
        (df["high_vol_flag"] == 1) &
        (df["vol_continuation_flag"] == 1) &
        (df["breakout_up"] == 1)
    ).astype(int)

    df["short_entry"] = (
        (df["high_vol_flag"] == 1) &
        (df["vol_continuation_flag"] == 1) &
        (df["breakout_down"] == 1)
    ).astype(int)

    df["entry_signal"] = np.select(
        [df["long_entry"] == 1, df["short_entry"] == 1],
        [1, -1],
        default=0
    )

    return df


def run_hmm_regime_detection(
    ticker: str = "spy",
    n_states: int = 3,
    prob_threshold: float = 0.60,
    breakout_window: int = 20,
    breakout_buffer: float = 0.0025
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Full HMM regime detection + trade signal generation.
    """
    df = load_features(ticker)
    df = prepare_hmm_features(df)
    df, model, feature_cols = fit_hmm(df, n_states=n_states)

    state_summary = summarize_states(df, n_states=n_states)
    high_vol_state = identify_high_vol_state(state_summary)

    df = add_breakout_logic(
        df,
        breakout_window=breakout_window,
        breakout_buffer=breakout_buffer
    )

    df = add_hmm_trade_signals(
        df,
        high_vol_state=high_vol_state,
        prob_threshold=prob_threshold
    )

    output_path = PROCESSED_DIR / f"{ticker.lower()}_hmm_signals.parquet"
    df.to_parquet(output_path)

    print("\n===== HMM STATE SUMMARY =====")
    print(state_summary.round(4))

    print(f"\nChosen high-vol state: {high_vol_state}")
    print(f"Saved HMM signal dataset to: {output_path}")

    print("\n===== SIGNAL COUNTS =====")
    print(df["entry_signal"].value_counts(dropna=False).sort_index())

    return df, state_summary, high_vol_state


if __name__ == "__main__":
    run_hmm_regime_detection(
        ticker="spy",
        n_states=3,
        prob_threshold=0.60,
        breakout_window=20,
        breakout_buffer=0.0025,
    )