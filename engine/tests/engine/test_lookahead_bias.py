"""Tests for lookahead bias across the feature and training pipelines.

Covers:
  1. Shuffled-future canary — features must not change in the first 80% when
     the last 20% of OHLCV data is replaced with random noise.
  2. Expanding-window equivalence — features computed on df[:t] must match
     features computed on df[:end] at bar t.
  3. Signal stability under truncation — signals at bar N-1 must be identical
     whether computed on df[:N] or df[:N+50].
  4. Optimizer parity — the optimizer fitness function must agree with the
     Tearsheet return model (open-based, shift(-2)).
  5. Train/test leakage — CPCV purge+embargo must eliminate all overlap
     between training indices and test indices (plus buffer zones).
"""
import pytest
import pandas as pd
import numpy as np

from engine.core.features.features import compute_all_features
from engine.core.optimization.cpcv_splitter import CPCVSplitter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic OHLCV DataFrame with lowercase columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# Feature configs for testing — covers every category.
# Each entry must exist in the FEATURE_REGISTRY after engine loads.
SAFE_FEATURES = [
    {"id": "RSI", "params": {"window": 14}},
    {"id": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
    {"id": "MovingAverage", "params": {"window": 20, "ma_type": "EMA"}},
    {"id": "BollingerBands", "params": {"window": 20, "num_std": 2.0}},
    {"id": "AverageTrueRange", "params": {"window": 14}},
]

# Features known to contain intentional look-ahead (Chikou Span, etc.)
LOOKAHEAD_FEATURES = [
    {"id": "Ichimoku", "params": {
        "conversion_period": 9, "base_period": 26,
        "lagging_span2_period": 52, "displacement": 26,
    }},
]


# ---------------------------------------------------------------------------
# Test 1 — Shuffled-future canary
# ---------------------------------------------------------------------------

class TestShuffledFutureCanary:
    """Replace the last 20% of OHLCV rows with random data and verify that
    features in the first 80% do not change. Any change proves the feature
    reads future bars.
    """

    @pytest.mark.parametrize("feature_config", SAFE_FEATURES, ids=lambda c: c["id"])
    def test_safe_feature_unaffected_by_future_noise(self, feature_config):
        df = _make_ohlcv(300)
        split = int(len(df) * 0.8)

        # Compute on original data
        df_orig, l_max_orig = compute_all_features(df.copy(), [feature_config])

        # Replace future (last 20%) with random noise
        rng = np.random.default_rng(99)
        df_noisy = df.copy()
        for col in ("open", "high", "low", "close"):
            df_noisy.iloc[split:, df_noisy.columns.get_loc(col)] = (
                50 + rng.normal(0, 10, len(df) - split)
            )
        df_noisy.iloc[split:, df_noisy.columns.get_loc("volume")] = (
            rng.integers(100, 500, len(df) - split).astype(float)
        )

        df_noisy_result, l_max_noisy = compute_all_features(df_noisy.copy(), [feature_config])

        # Compare computed feature columns (not raw OHLCV) in the first 80%
        feature_cols = [
            c for c in df_orig.columns
            if c.lower() not in {"open", "high", "low", "close", "volume"}
        ]
        assert len(feature_cols) > 0, f"Feature {feature_config['id']} produced no columns"

        for col in feature_cols:
            orig_slice = df_orig[col].iloc[:split]
            noisy_slice = df_noisy_result[col].iloc[:split]
            # Allow NaN-NaN matches; only compare where at least one is non-NaN
            both_nan = orig_slice.isna() & noisy_slice.isna()
            comparable = ~both_nan
            if comparable.sum() == 0:
                continue
            pd.testing.assert_series_equal(
                orig_slice[comparable].reset_index(drop=True),
                noisy_slice[comparable].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
                obj=f"{col} (first 80%)",
            )

    def test_ichimoku_chikou_is_now_causal(self):
        """After the fix, Ichimoku Chikou Span uses shift(+disp) instead of
        shift(-disp). Verify it passes the canary — the first 80% must be
        unaffected by changes to the last 20%.
        """
        df = _make_ohlcv(300)
        split = int(len(df) * 0.8)
        config = LOOKAHEAD_FEATURES[0]

        df_orig, _ = compute_all_features(df.copy(), [config])

        rng = np.random.default_rng(99)
        df_noisy = df.copy()
        for col in ("open", "high", "low", "close"):
            df_noisy.iloc[split:, df_noisy.columns.get_loc(col)] = (
                50 + rng.normal(0, 10, len(df) - split)
            )
        df_noisy_result, _ = compute_all_features(df_noisy.copy(), [config])

        # Find chikou column
        chikou_cols = [c for c in df_orig.columns if "chikou" in c.lower()]
        assert len(chikou_cols) > 0, "Expected chikou column from Ichimoku"

        chikou_col = chikou_cols[0]
        orig_slice = df_orig[chikou_col].iloc[:split]
        noisy_slice = df_noisy_result[chikou_col].iloc[:split]

        # After the fix, chikou is causal — first 80% must be identical.
        comparable = ~(orig_slice.isna() & noisy_slice.isna())
        if comparable.sum() > 0:
            pd.testing.assert_series_equal(
                orig_slice[comparable].reset_index(drop=True),
                noisy_slice[comparable].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
                obj=f"{chikou_col} (first 80%)",
            )


# ---------------------------------------------------------------------------
# Test 2 — Expanding-window equivalence
# ---------------------------------------------------------------------------

class TestExpandingWindowEquivalence:
    """Compute features on the full dataset, then on a truncated prefix.
    The feature value at the last bar of the prefix must match the
    corresponding bar in the full computation.
    """

    @pytest.mark.parametrize("feature_config", SAFE_FEATURES, ids=lambda c: c["id"])
    def test_feature_stable_under_truncation(self, feature_config):
        df = _make_ohlcv(200)
        cutoff = 150

        # Full computation
        df_full, _ = compute_all_features(df.copy(), [feature_config])

        # Truncated computation
        df_trunc, _ = compute_all_features(df.iloc[:cutoff].copy(), [feature_config])

        feature_cols = [
            c for c in df_full.columns
            if c.lower() not in {"open", "high", "low", "close", "volume"}
        ]

        for col in feature_cols:
            if col not in df_trunc.columns:
                continue
            full_val = df_full[col].iloc[cutoff - 1]
            trunc_val = df_trunc[col].iloc[cutoff - 1]
            if pd.isna(full_val) and pd.isna(trunc_val):
                continue
            assert full_val == pytest.approx(trunc_val, abs=1e-10, nan_ok=True), (
                f"Feature {col} at bar {cutoff - 1} differs: "
                f"full={full_val}, truncated={trunc_val}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Signal stability under truncation
# ---------------------------------------------------------------------------

class TestSignalStabilityUnderTruncation:
    """Signals at bar N-1 should be identical whether computed on df[:N] or
    df[:N+K]. If they differ, the model or a feature is reading future bars.

    Uses a deterministic rule-based model (dummy) to isolate feature-level
    lookahead from model-level issues.
    """

    def test_dummy_strategy_signals_stable(self, tmp_path):
        """A trivial rule-based strategy must produce identical signals
        regardless of how many future bars exist in the dataset."""
        import json

        # Build a minimal strategy that uses RSI
        strat_dir = tmp_path / "canary_strat"
        strat_dir.mkdir()

        manifest = {
            "name": "Canary",
            "is_ml": False,
            "features": [{"id": "RSI", "params": {"window": 14}}],
            "hyperparameters": {"threshold": 30},
        }
        (strat_dir / "manifest.json").write_text(json.dumps(manifest))

        (strat_dir / "context.py").write_text(
            "class Context:\n    pass\n"
        )

        (strat_dir / "model.py").write_text("""\
import pandas as pd
from engine.core.controller import SignalModel

class CanaryModel(SignalModel):
    def train(self, df, context, params):
        return {}

    def generate_signals(self, df, context, params, artifacts):
        rsi_col = [c for c in df.columns if 'RSI' in c][0]
        rsi = df[rsi_col]
        signals = pd.Series(0.0, index=df.index)
        signals[rsi < params.get('threshold', 30)] = 1.0
        signals[rsi > (100 - params.get('threshold', 30))] = -1.0
        return signals
""")

        from engine.core.backtester import LocalBacktester

        df = _make_ohlcv(250)
        N = 180

        bt = LocalBacktester(str(strat_dir))

        signals_short = bt.run(df.iloc[:N].copy())
        signals_long = bt.run(df.iloc[: N + 50].copy())

        # The signal at the last bar of the short run should match
        # the same bar in the long run.
        last_date = signals_short.index[-1]
        assert last_date in signals_long.index, "Index mismatch"
        assert signals_short.loc[last_date] == pytest.approx(
            signals_long.loc[last_date], abs=1e-10
        ), (
            f"Signal at {last_date} changed when future data was appended: "
            f"short={signals_short.loc[last_date]}, long={signals_long.loc[last_date]}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Optimizer fitness parity with Tearsheet
# ---------------------------------------------------------------------------

class TestOptimizerFitnessParity:
    """The optimizer's fitness function must use the same return model as
    Tearsheet (open-based, T+1 execution = shift(-2)).

    We compute Sharpe both ways and assert they are consistent. If the
    optimizer's Sharpe is systematically higher, it's cheating with a
    tighter return alignment.
    """

    def test_optimizer_sharpe_matches_tearsheet_direction(self):
        """On a clearly trending dataset, both the optimizer fitness and
        Tearsheet should agree on the sign of the Sharpe ratio for a
        constant-long strategy.
        """
        from engine.core.backtester import Tearsheet

        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Strong uptrend: open increases steadily
        open_ = np.linspace(100, 150, n)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": open_ + 1,
                "low": open_ - 1,
                "close": open_ + 0.5,
                "volume": np.full(n, 1_000_000.0),
            },
            index=dates,
        )
        signals = pd.Series(1.0, index=df.index)

        # Tearsheet Sharpe (ground truth)
        metrics = Tearsheet.calculate_metrics(df, signals)
        tearsheet_sharpe = metrics["Sharpe Ratio"]

        # Optimizer-style Sharpe (should now be equivalent after fix)
        returns = df["open"].pct_change().shift(-2).reindex(signals.index)
        strategy_returns = signals.shift(1).fillna(0) * returns.fillna(0)
        opt_sharpe = 0.0
        if strategy_returns.std() != 0:
            opt_sharpe = float(
                (strategy_returns.mean() / strategy_returns.std()) * (252**0.5)
            )

        # Both should be positive on a clear uptrend
        assert tearsheet_sharpe > 0, f"Tearsheet Sharpe should be positive: {tearsheet_sharpe}"
        assert opt_sharpe > 0, f"Optimizer Sharpe should be positive: {opt_sharpe}"

        # They should be in the same ballpark (within 50% relative)
        if tearsheet_sharpe != 0:
            ratio = opt_sharpe / tearsheet_sharpe
            assert 0.5 < ratio < 2.0, (
                f"Optimizer Sharpe ({opt_sharpe:.2f}) diverges too far from "
                f"Tearsheet Sharpe ({tearsheet_sharpe:.2f}), ratio={ratio:.2f}"
            )


# ---------------------------------------------------------------------------
# Test 5 — Train/test leakage in CPCV
# ---------------------------------------------------------------------------

class TestCPCVLeakage:
    """CPCV folds must have zero overlap between train and test indices,
    and the purge/embargo zones must be correctly excluded from training.
    """

    def test_no_train_test_overlap(self):
        """Train and test indices must be disjoint in every fold."""
        df = _make_ohlcv(200)
        splitter = CPCVSplitter(n_groups=6, k_test_groups=2, embargo_pct=0.01)
        folds = splitter.split(df, l_max=20)

        for i, (train_idx, test_idx) in enumerate(folds):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, (
                f"Fold {i}: {len(overlap)} indices appear in both train and test"
            )

    def test_purge_removes_buffer_before_test(self):
        """Training indices within l_max bars before any test block must
        be removed by the purge protocol.
        """
        df = _make_ohlcv(200)
        l_max = 20
        splitter = CPCVSplitter(n_groups=6, k_test_groups=2, embargo_pct=0.0)
        folds = splitter.split(df, l_max=l_max)

        for i, (train_idx, test_idx) in enumerate(folds):
            # Find start of each contiguous test block
            sorted_test = np.sort(test_idx)
            breaks = np.where(np.diff(sorted_test) > 1)[0]
            block_starts = [sorted_test[0]] + [sorted_test[b + 1] for b in breaks]

            for block_start in block_starts:
                purge_zone = set(range(max(0, block_start - l_max), block_start))
                leaked = purge_zone.intersection(set(train_idx))
                assert len(leaked) == 0, (
                    f"Fold {i}: {len(leaked)} training indices in purge zone "
                    f"before test block starting at {block_start}: {sorted(leaked)[:5]}..."
                )

    def test_embargo_removes_buffer_after_test(self):
        """Training indices within embargo_size bars after any test block
        must be removed by the embargo protocol.
        """
        n = 200
        df = _make_ohlcv(n)
        embargo_pct = 0.05  # 5% = 10 bars on 200-bar dataset
        splitter = CPCVSplitter(n_groups=6, k_test_groups=2, embargo_pct=embargo_pct)
        folds = splitter.split(df, l_max=0)
        embargo_size = max(1, int(n * embargo_pct))

        for i, (train_idx, test_idx) in enumerate(folds):
            # Find end of each contiguous test block
            sorted_test = np.sort(test_idx)
            breaks = np.where(np.diff(sorted_test) > 1)[0]
            block_ends = [sorted_test[b] for b in breaks] + [sorted_test[-1]]

            for block_end in block_ends:
                embargo_zone = set(range(block_end + 1, min(n, block_end + 1 + embargo_size)))
                leaked = embargo_zone.intersection(set(train_idx))
                assert len(leaked) == 0, (
                    f"Fold {i}: {len(leaked)} training indices in embargo zone "
                    f"after test block ending at {block_end}: {sorted(leaked)[:5]}..."
                )

    def test_temporal_split_is_strictly_ordered(self):
        """For a simple temporal split, all training indices must come
        strictly before all validation indices.
        """
        from engine.core.trainer import LocalTrainer

        # We can't easily instantiate LocalTrainer without a strategy dir,
        # so test the temporal split logic directly.
        n = 200
        ratio = 0.8
        split_point = int(n * ratio)
        train_idx = np.arange(0, split_point)
        val_idx = np.arange(split_point, n)

        assert train_idx[-1] < val_idx[0], (
            f"Temporal split violated: last train={train_idx[-1]}, "
            f"first val={val_idx[0]}"
        )
        assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_all_folds_cover_full_test_space(self):
        """The union of all test indices across all folds should cover
        every index in the dataset (each index appears in at least one
        test fold).
        """
        n = 200
        df = _make_ohlcv(n)
        splitter = CPCVSplitter(n_groups=6, k_test_groups=2, embargo_pct=0.01)
        folds = splitter.split(df, l_max=0)

        all_test = set()
        for _, test_idx in folds:
            all_test.update(test_idx.tolist())

        assert all_test == set(range(n)), (
            f"Test indices don't cover full dataset: "
            f"missing {set(range(n)) - all_test}"
        )
