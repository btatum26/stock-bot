"""
PortfolioBacktester — event-driven multi-asset portfolio simulation.

Uses a T+1 execution model: signals generated at bar T are executed at bar
T+1 open.  Position sizing is governed by the 2% risk rule (never risk more
than `risk_per_trade_pct` of portfolio on a single stop-out) scaled by signal
strength.  Rebalancing is lazy by default — positions are only touched on stop
hits, signal exits, signal-strength-based evictions, or (optionally) when
`rebalance_on_strength=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .logger import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConfig:
    starting_capital:      float = 100_000.0
    max_positions:         int   = 10
    risk_per_trade_pct:    float = 0.02    # fraction of portfolio risked per trade
    stop_loss_pct:         float = 0.05    # fixed stop as fraction of entry price
    max_position_pct:      float = 0.20    # signal-scaled size cap per position
    entry_threshold:       float = 0.10    # min |signal| to open a position
    eviction_margin:       float = 0.15    # new signal must exceed weakest by this
    friction:              float = 0.001   # one-way transaction cost (fraction)
    rebalance_on_strength: bool  = False   # resize open positions when signal shifts
    rebalance_delta:       float = 0.10    # min |Δsignal| to trigger a resize
    allow_short:           bool  = True


# ---------------------------------------------------------------------------
# Internal position record
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    ticker:          str
    direction:       int            # +1 = long, -1 = short
    entry_bar:       int            # integer index into unified_index
    entry_date:      pd.Timestamp
    entry_price:     float
    shares:          float
    stop_price:      float
    signal_strength: float          # |signal| at last sizing event


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class PortfolioBacktester:
    """
    Multi-asset, event-driven portfolio simulation.

    Usage
    -----
    result = PortfolioBacktester().run(datasets, all_signals, config)

    Returns a dict with:
        equity_curve            — pd.Series of portfolio value over time
        position_weights        — {ticker: pd.Series} signed weight in portfolio
        trade_log               — pd.DataFrame, one row per completed round-trip
        per_ticker_contribution — {ticker: net_pnl float}
        metrics                 — dict of scalar performance statistics
    """

    def run(
        self,
        datasets:    Dict[str, pd.DataFrame],
        all_signals: Dict[str, pd.Series],
        config:      PortfolioConfig,
    ) -> dict:
        valid = [t for t, s in all_signals.items() if not s.empty and t in datasets]
        if not valid:
            return self._empty_result()

        # ------------------------------------------------------------------ #
        # 0. Build unified datetime index (union of all signal indices)       #
        # ------------------------------------------------------------------ #
        unified_index: pd.DatetimeIndex = all_signals[valid[0]].index
        for t in valid[1:]:
            unified_index = unified_index.union(all_signals[t].index)
        unified_index = unified_index.sort_values()
        n = len(unified_index)

        # Pre-build numpy arrays aligned to unified_index for fast bar lookup
        opens:  Dict[str, np.ndarray] = {}
        closes: Dict[str, np.ndarray] = {}
        sigs:   Dict[str, np.ndarray] = {}

        for t in valid:
            df  = datasets[t]
            sig = all_signals[t]
            opens[t]  = df['open'].reindex(unified_index).values.astype(float)
            closes[t] = df['close'].reindex(unified_index).values.astype(float)
            sigs[t]   = sig.reindex(unified_index).fillna(0.0).values.astype(float)

        # ------------------------------------------------------------------ #
        # 1. Main simulation loop                                             #
        # ------------------------------------------------------------------ #
        cash: float = float(config.starting_capital)
        open_positions: Dict[str, _Position] = {}

        equity_arr   = np.full(n, np.nan)
        weight_arrs: Dict[str, np.ndarray] = {t: np.zeros(n) for t in valid}
        completed_trades: List[dict] = []

        for i in range(n - 1):
            next_bar = unified_index[i + 1]

            # Mark-to-market at close[i]
            pos_value = _mtm_value(open_positions, closes, i)
            portfolio_value = cash + pos_value

            # ---- 1a. Stop-loss & signal exits → execute at open[i+1] ---- #
            to_close: List[Tuple[str, float, str]] = []

            for t, pos in list(open_positions.items()):
                nx_open = opens[t][i + 1]
                if np.isnan(nx_open):
                    continue

                if pos.direction == 1 and nx_open <= pos.stop_price:
                    to_close.append((t, nx_open, 'STOP'))
                    continue
                if pos.direction == -1 and nx_open >= pos.stop_price:
                    to_close.append((t, nx_open, 'STOP'))
                    continue

                sig_val = sigs[t][i]
                if abs(sig_val) < config.entry_threshold:
                    to_close.append((t, nx_open, 'SIGNAL'))
                elif (sig_val > 0) != (pos.direction > 0):
                    to_close.append((t, nx_open, 'FLIP'))

            for t, exec_price, reason in to_close:
                if t not in open_positions:
                    continue
                trade = _close_position(
                    open_positions.pop(t), exec_price, next_bar, i, config.friction
                )
                trade['exit_reason'] = reason
                completed_trades.append(trade)
                cash += trade['_cash_delta']

            # ---- 1b. Optional rebalance on signal strength -------------- #
            if config.rebalance_on_strength:
                for t, pos in list(open_positions.items()):
                    new_abs = abs(sigs[t][i])
                    if abs(new_abs - pos.signal_strength) < config.rebalance_delta:
                        continue
                    nx_open = opens[t][i + 1]
                    if np.isnan(nx_open) or nx_open <= 0:
                        continue
                    new_shares = _size_position(
                        portfolio_value, nx_open, new_abs, config
                    )
                    delta = new_shares - pos.shares
                    if delta > 0:
                        cost = delta * nx_open * (1 + config.friction)
                        if cost <= cash:
                            cash -= cost
                            pos.shares += delta
                    elif delta < 0:
                        sell = abs(delta)
                        cash += sell * nx_open * (1 - config.friction)
                        pos.shares -= sell
                    pos.signal_strength = new_abs

            # ---- 1c. New entry candidates (sorted strongest first) ------- #
            candidates: List[Tuple[str, float]] = []
            for t in valid:
                if t in open_positions:
                    continue
                s = sigs[t][i]
                if abs(s) < config.entry_threshold:
                    continue
                if not config.allow_short and s < 0:
                    continue
                nx_open = opens[t][i + 1]
                if np.isnan(nx_open) or nx_open <= 0:
                    continue
                candidates.append((t, s))

            candidates.sort(key=lambda x: abs(x[1]), reverse=True)

            for t, s in candidates:
                nx_open = opens[t][i + 1]

                if len(open_positions) >= config.max_positions:
                    if not open_positions:
                        break
                    weakest_t = min(
                        open_positions,
                        key=lambda x: open_positions[x].signal_strength,
                    )
                    weakest_strength = open_positions[weakest_t].signal_strength
                    if abs(s) < weakest_strength + config.eviction_margin:
                        continue  # not strong enough to evict

                    evict_price = opens[weakest_t][i + 1]
                    if np.isnan(evict_price):
                        continue
                    trade = _close_position(
                        open_positions.pop(weakest_t),
                        evict_price, next_bar, i, config.friction,
                    )
                    trade['exit_reason'] = 'EVICTED'
                    completed_trades.append(trade)
                    cash += trade['_cash_delta']

                shares = _size_position(portfolio_value, nx_open, s, config)
                cost   = shares * nx_open * (1 + config.friction)
                if cost > cash:
                    shares = cash / (nx_open * (1 + config.friction))
                if shares <= 0:
                    continue

                cash -= shares * nx_open * (1 + config.friction)
                direction  = 1 if s > 0 else -1
                stop_price = nx_open * (1 - config.stop_loss_pct * direction)

                open_positions[t] = _Position(
                    ticker=t,
                    direction=direction,
                    entry_bar=i + 1,
                    entry_date=next_bar,
                    entry_price=nx_open,
                    shares=shares,
                    stop_price=stop_price,
                    signal_strength=abs(s),
                )

            # ---- 1d. Record equity and weights for this bar -------------- #
            pv = cash + _mtm_value(open_positions, closes, i)
            equity_arr[i] = pv

            if pv > 0:
                for t, pos in open_positions.items():
                    c = closes[t][i]
                    if not np.isnan(c):
                        weight_arrs[t][i] = (pos.shares * c / pv) * pos.direction

        # ---- 1e. Close all remaining positions at final bar -------------- #
        for t, pos in list(open_positions.items()):
            c = closes[t][-1]
            if np.isnan(c):
                c = pos.entry_price
            trade = _close_position(
                pos, c, unified_index[-1], n - 1, config.friction
            )
            trade['exit_reason'] = 'END_OF_DATA'
            completed_trades.append(trade)
            cash += trade['_cash_delta']

        equity_arr[-1] = cash
        for t, pos in open_positions.items():
            c = closes[t][-1]
            if not np.isnan(c) and cash > 0:
                weight_arrs[t][-1] = (pos.shares * c / cash) * pos.direction

        # ------------------------------------------------------------------ #
        # 2. Assemble result                                                  #
        # ------------------------------------------------------------------ #
        equity_curve = (
            pd.Series(equity_arr, index=unified_index)
            .ffill()
            .fillna(config.starting_capital)
        )

        position_weights = {
            t: pd.Series(weight_arrs[t], index=unified_index)
            for t in valid
        }

        trade_df = _build_trade_df(completed_trades)

        per_ticker_contribution = {
            t: round(
                sum(tr['pnl'] for tr in completed_trades if tr['ticker'] == t), 4
            )
            for t in valid
        }

        metrics = _calculate_metrics(
            equity_curve, trade_df, config.starting_capital
        )

        logger.info(
            f"Portfolio simulation complete: {len(completed_trades)} trades, "
            f"final value ${equity_curve.iloc[-1]:,.2f}"
        )

        return {
            'equity_curve':            equity_curve,
            'position_weights':        position_weights,
            'trade_log':               trade_df,
            'per_ticker_contribution': per_ticker_contribution,
            'metrics':                 metrics,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            'equity_curve':            pd.Series(dtype=float),
            'position_weights':        {},
            'trade_log':               pd.DataFrame(),
            'per_ticker_contribution': {},
            'metrics':                 {},
        }


# ---------------------------------------------------------------------------
# Module-level helpers (keep the class methods short)
# ---------------------------------------------------------------------------

def _mtm_value(
    open_positions: Dict[str, '_Position'],
    closes: Dict[str, np.ndarray],
    bar_idx: int,
) -> float:
    """Sum of mark-to-market value of all open positions at bar_idx."""
    total = 0.0
    for t, pos in open_positions.items():
        c = closes[t][bar_idx]
        if not np.isnan(c):
            total += (
                pos.shares * pos.entry_price
                + pos.direction * (c - pos.entry_price) * pos.shares
            )
    return total


def _size_position(
    portfolio_value: float,
    entry_price:     float,
    signal:          float,
    config:          PortfolioConfig,
) -> float:
    """
    Returns share count using the 2% risk rule capped by signal-scaled max size.

    2% rule:   shares = (portfolio * risk_pct) / (entry * stop_pct)
    Signal cap: shares = (portfolio * max_pos_pct * |signal|) / entry
    Final:      min(rule_shares, cap_shares)
    """
    if entry_price <= 0 or config.stop_loss_pct <= 0:
        return 0.0
    risk_dollars = portfolio_value * config.risk_per_trade_pct
    rule_shares  = risk_dollars / (entry_price * config.stop_loss_pct)
    cap_shares   = (portfolio_value * config.max_position_pct * abs(signal)) / entry_price
    return max(0.0, min(rule_shares, cap_shares))


def _close_position(
    pos:        '_Position',
    exec_price: float,
    exit_date:  pd.Timestamp,
    exit_bar:   int,
    friction:   float,
) -> dict:
    """
    Compute realised P&L and cash delta for closing a position.

    Cash model
    ----------
    Entry deducted: shares * entry_price * (1 + friction)
    Exit returned : shares * entry_price + direction*(exit-entry)*shares - exit_friction
                  = committed_capital + gross_pnl - exit_friction
    """
    gross_pnl     = pos.direction * (exec_price - pos.entry_price) * pos.shares
    exit_friction = pos.shares * exec_price * friction
    net_pnl       = gross_pnl - exit_friction
    cash_delta    = pos.shares * pos.entry_price + net_pnl

    return_pct = (
        pos.direction * (exec_price - pos.entry_price) / pos.entry_price * 100
        if pos.entry_price > 0 else 0.0
    )

    return {
        'ticker':      pos.ticker,
        'direction':   'LONG' if pos.direction == 1 else 'SHORT',
        'entry_date':  pos.entry_date,
        'exit_date':   exit_date,
        'entry_price': round(pos.entry_price, 4),
        'exit_price':  round(exec_price, 4),
        'shares':      round(pos.shares, 4),
        'pnl':         round(net_pnl, 4),
        'return_pct':  round(return_pct, 4),
        'bars_held':   max(0, exit_bar - pos.entry_bar),
        'exit_reason': '',          # filled by caller
        '_cash_delta': cash_delta,  # stripped before returning to GUI
    }


def _build_trade_df(trades: List[dict]) -> pd.DataFrame:
    cols = [
        'ticker', 'direction', 'entry_date', 'exit_date',
        'entry_price', 'exit_price', 'shares', 'pnl', 'return_pct',
        'bars_held', 'exit_reason',
    ]
    if not trades:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(trades).drop(columns=['_cash_delta'], errors='ignore')[cols]


def _calculate_metrics(
    equity:           pd.Series,
    trade_df:         pd.DataFrame,
    starting_capital: float,
) -> dict:
    equity = equity.dropna()
    if equity.empty or equity.iloc[0] <= 0:
        return {}

    returns = equity.pct_change().fillna(0)

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # CAGR — use median bar duration to handle non-contiguous indices
    if len(equity.index) > 1:
        diffs_sec      = equity.index.to_series().diff().dropna().dt.total_seconds()
        median_bar_sec = float(diffs_sec.median()) if not diffs_sec.empty else 86400.0
        days           = max(int(median_bar_sec * len(equity) / 86400), 1)
    else:
        days = 1

    end_ratio = float(equity.iloc[-1] / equity.iloc[0])
    if end_ratio > 0 and days > 0:
        exp  = np.clip(np.log(end_ratio) * (365.25 / days), -10.0, 10.0)
        cagr = (np.exp(exp) - 1) * 100
    else:
        cagr = -100.0

    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    max_dd      = float(drawdown.min() * 100)

    vol    = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0.0

    downside_returns = returns[returns < 0]
    dv      = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    sortino = (returns.mean() * 252) / dv if dv > 0 else 0.0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    n_trades = len(trade_df)
    if n_trades > 0:
        win_rate = float((trade_df['pnl'] > 0).mean() * 100)
        wins     = trade_df.loc[trade_df['pnl'] > 0, 'return_pct']
        losses   = trade_df.loc[trade_df['pnl'] < 0, 'return_pct']
        avg_win  = float(wins.mean())  if not wins.empty  else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
    else:
        win_rate = avg_win = avg_loss = 0.0

    # Turnover = total traded notional / (avg portfolio value * years)
    years = max(days / 365.25, 1 / 365.25)
    if n_trades > 0 and equity.mean() > 0:
        total_notional = float((trade_df['shares'] * trade_df['entry_price']).sum() * 2)
        turnover       = round(total_notional / (float(equity.mean()) * years) * 100, 2)
    else:
        turnover = 0.0

    return {
        'Total Return (%)': round(total_return, 2),
        'CAGR (%)':         round(cagr, 2),
        'Sharpe Ratio':     round(sharpe, 2),
        'Sortino Ratio':    round(sortino, 2),
        'Calmar Ratio':     round(calmar, 2),
        'Max Drawdown (%)': round(max_dd, 2),
        'Win Rate (%)':     round(win_rate, 2),
        'Avg Win (%)':      round(avg_win, 4),
        'Avg Loss (%)':     round(avg_loss, 4),
        'Total Trades':     n_trades,
        'Turnover (%)':     turnover,
    }
