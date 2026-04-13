"""
research_cli.py  —  Research Engine CLI

Run from the repo root:
    python research_cli.py <command> [options]

Uses ModelEngine (engine/bridge.py) — the same facade the GUI uses.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional

# ── Bootstrap: ensure the engine package is importable from repo root ─────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import ModelEngine
from engine.core.exceptions import StrategyError, ValidationError

# ── Paths (mirror gui/config.py) ──────────────────────────────────────────────
WORKSPACE_DIR = os.path.join(_ROOT, "strategies")
DB_PATH       = os.path.join(_ROOT, "data", "stocks.db")

# ── Formatting ────────────────────────────────────────────────────────────────
WIDTH = 64

def _header(title: str) -> None:
    """Print a top-level section header with a full-width border."""
    print("\n" + "=" * WIDTH)
    for line in title.splitlines():
        print(f"  {line}")
    print("=" * WIDTH)

def _section(title: str) -> None:
    """Print a sub-section divider line within a command's output block."""
    print(f"\n  -- {title} {'-' * max(0, WIDTH - len(title) - 6)}")

def _row(label: str, value, indent: int = 4) -> None:
    """Print a single key/value row with consistent column alignment."""
    pad = " " * indent
    print(f"{pad}{label:<28}  {value}")

def _progress_bar(pct: int) -> str:
    """Return a 20-character ASCII progress bar string for the given percentage (0-100)."""
    filled = pct // 5
    return "#" * filled + "." * (20 - filled)

# ── Callbacks for progress + log forwarding ───────────────────────────────────
def _make_callbacks(silent: bool = False) -> dict:
    """
    Produces a callbacks dict compatible with ModelEngine execution methods.
    Progress lines overwrite in place; log lines stream below.
    """
    def on_progress(pct: int, msg: str) -> None:
        if not silent:
            bar = _progress_bar(pct)
            print(f"  [{bar}] {pct:3d}%  {msg}")

    def on_log(msg: str) -> None:
        if not silent:
            print(f"       {msg}")

    return {
        "is_cancelled": lambda: False,
        "on_progress":  on_progress,
        "on_log":       on_log,
    }

# ── Type coercion ─────────────────────────────────────────────────────────────
def _coerce(value: str):
    """Try int → float → str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

def _parse_kv_pairs(items: List[str]) -> dict:
    """Parse ['k=v', 'k2=v2'] into a typed dict."""
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected k=v, got: {item!r}")
        k, _, v = item.partition("=")
        result[k.strip()] = _coerce(v.strip())
    return result

# ── Date helpers ──────────────────────────────────────────────────────────────
def _resolve_dates(start_arg: Optional[str], end_arg: Optional[str],
                   default_lookback_days: int = 365):
    """
    Parse optional YYYY-MM-DD date strings into datetime objects.

    If end_arg is omitted, defaults to today. If start_arg is omitted,
    defaults to default_lookback_days before the end date.

    Returns:
        tuple[datetime, datetime]: (start_dt, end_dt)
    """
    end_dt   = datetime.strptime(end_arg,   "%Y-%m-%d") if end_arg   else datetime.now()
    start_dt = datetime.strptime(start_arg, "%Y-%m-%d") if start_arg else end_dt - timedelta(days=default_lookback_days)
    return start_dt, end_dt

# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(engine: ModelEngine, args) -> None:
    """
    List all strategies found in WORKSPACE_DIR.

    Prints each strategy name alongside its configured feature IDs and
    hyperparameter key=value pairs. With --json, emits the full manifest
    for every strategy as a JSON array.
    """
    strategies = engine.list_strategies()

    if args.json:
        out = []
        for name in strategies:
            cfg = engine.get_strategy_config(name)
            out.append({"name": name, "config": cfg})
        print(json.dumps(out, indent=2))
        return

    if not strategies:
        print(f"No strategies found in {WORKSPACE_DIR}")
        return

    _header(f"Strategies  ({len(strategies)} found)")
    for name in strategies:
        cfg      = engine.get_strategy_config(name)
        features = [f["id"] for f in cfg.get("features", [])]
        hparams  = cfg.get("hyperparameters", {})
        print(f"\n  {name}")
        print(f"    Features : {', '.join(features) or '(none)'}")
        hp_str = ", ".join(f"{k}={v}" for k, v in hparams.items())
        print(f"    Hparams  : {hp_str or '(none)'}")


def cmd_inspect(engine: ModelEngine, args) -> None:
    """
    Print the full manifest for a single strategy.

    Shows features (with all params), hyperparameters, parameter bounds,
    optional training config, and the path to model.py. With --json,
    emits the raw manifest dict.
    """
    name = args.strategy
    cfg  = engine.get_strategy_config(name)

    if args.json:
        print(json.dumps({"name": name, "config": cfg}, indent=2))
        return

    _header(f"Strategy: {name}")

    _section("Features")
    for feat in cfg.get("features", []):
        params_str = "  ".join(f"{k}={v}" for k, v in feat.get("params", {}).items())
        print(f"    {feat['id']:<22}  {params_str}")
    if not cfg.get("features"):
        print("    (none)")

    _section("Hyperparameters")
    for k, v in cfg.get("hyperparameters", {}).items():
        _row(k, v)
    if not cfg.get("hyperparameters"):
        print("    (none)")

    _section("Parameter Bounds")
    for k, v in cfg.get("parameter_bounds", {}).items():
        _row(k, v)
    if not cfg.get("parameter_bounds"):
        print("    (none)")

    if "training" in cfg:
        _section("Training Config")
        for k, v in cfg["training"].items():
            _row(k, v)

    # Show model.py path
    model_path = os.path.join(WORKSPACE_DIR, name, "model.py")
    print(f"\n  model.py : {model_path}")


def cmd_features(engine: ModelEngine, args) -> None:
    """
    List every feature registered in the engine's feature registry.

    Entries are grouped by category and sorted alphabetically. Each entry
    shows the feature ID, a one-line description, and its default parameter
    values (list-type params show the first option as the default).
    With --json, emits the raw list of feature dicts.
    """
    features = engine.get_available_features()

    if args.json:
        print(json.dumps(features, indent=2, default=str))
        return

    # Group by category
    by_cat: dict = {}
    for f in features:
        by_cat.setdefault(f["category"], []).append(f)

    _header(f"Available Features  ({len(features)} total)")
    for cat in sorted(by_cat):
        _section(cat)
        for f in sorted(by_cat[cat], key=lambda x: x["id"]):
            print(f"    {f['id']:<22}  {f['description']}")
            if f.get("params"):
                params_display = {}
                for k, v in f["params"].items():
                    # If a param is a list (like normalize options), show the first as default
                    params_display[k] = v[0] if isinstance(v, list) else v
                param_str = "  ".join(f"{k}={v}" for k, v in params_display.items())
                print(f"    {'':22}  Defaults: {param_str}")


def cmd_init(engine: ModelEngine, args) -> None:
    """
    Scaffold a new strategy workspace under WORKSPACE_DIR/<name>.

    Delegates to engine.create_strategy(), which writes a default manifest
    and generates context.py and model.py via WorkspaceManager. Raises
    StrategyError if the name is not a valid Python identifier or already exists.
    """
    name = args.strategy
    engine.create_strategy(name)
    dest = os.path.join(WORKSPACE_DIR, name)
    print(f"[+] Created strategy: {name}")
    print(f"    {dest}")
    print(f"\n  Edit the manifest, then run:")
    print(f"    python research_cli.py inspect {name}")
    print(f"    python research_cli.py edit {name} --add-feature <ID> --feature-params k=v ...")


def cmd_edit(engine: ModelEngine, args) -> None:
    """
    Modify a strategy's manifest and re-sync context.py in one step.

    Supported mutations (all optional, applied in order):
      --clear-features        Remove all features first.
      --remove-feature ID     Drop every feature entry with the given ID.
      --add-feature ID        Append a new feature entry; pair with
                              --feature-params to set its params.
      --set-hparam k=v        Add or overwrite a hyperparameter.
      --delete-hparam key     Remove a hyperparameter.
      --set-bound key=lo,hi   Set the optimisation bound for a parameter.

    After all mutations are applied, calls engine.save_strategy_config()
    which persists the manifest and regenerates context.py via WorkspaceManager.
    Exits with an error message if no mutations were specified or if a
    feature ID is not present in the engine registry.
    """
    name     = args.strategy
    cfg      = engine.get_strategy_config(name)
    features = list(cfg.get("features", []))
    hparams  = dict(cfg.get("hyperparameters", {}))
    bounds   = dict(cfg.get("parameter_bounds", {}))
    changed  = False

    # --clear-features
    if args.clear_features:
        features = []
        changed  = True
        print("  Cleared all features.")

    # --remove-feature ID [ID ...]
    for fid in (args.remove_feature or []):
        before   = len(features)
        features = [f for f in features if f["id"] != fid]
        if len(features) < before:
            print(f"  Removed feature: {fid}")
            changed = True
        else:
            print(f"  Warning: feature '{fid}' not found, nothing removed.")

    # --add-feature ID [--feature-params k=v ...]
    if args.add_feature:
        fid    = args.add_feature
        params = {}
        if args.feature_params:
            try:
                params = _parse_kv_pairs(args.feature_params)
            except ValueError as e:
                print(f"Error parsing --feature-params: {e}")
                sys.exit(1)

        avail = {f["id"] for f in engine.get_available_features()}
        if fid not in avail:
            print(f"Error: '{fid}' is not a registered feature. Run `features` to see valid IDs.")
            sys.exit(1)

        features.append({"id": fid, "params": params})
        print(f"  Added feature: {fid}  params={params}")
        changed = True

    # --set-hparam k=v [k=v ...]
    if args.set_hparam:
        try:
            updates = _parse_kv_pairs(args.set_hparam)
        except ValueError as e:
            print(f"Error parsing --set-hparam: {e}")
            sys.exit(1)
        for k, v in updates.items():
            hparams[k] = v
            print(f"  Set hparam: {k} = {v!r}")
        changed = True

    # --set-bound key=lo,hi [key=lo,hi ...]
    if args.set_bound:
        for item in args.set_bound:
            if "=" not in item:
                print(f"Error: --set-bound expects 'key=lo,hi', got: {item!r}")
                sys.exit(1)
            key, _, val = item.partition("=")
            parts = val.split(",")
            if len(parts) != 2:
                print(f"Error: --set-bound value must be 'lo,hi', got: {val!r}")
                sys.exit(1)
            lo, hi = _coerce(parts[0].strip()), _coerce(parts[1].strip())
            bounds[key.strip()] = [lo, hi]
            print(f"  Set bound: {key.strip()} = [{lo}, {hi}]")
        changed = True

    # --delete-hparam key [key ...]
    for key in (args.delete_hparam or []):
        if key in hparams:
            del hparams[key]
            print(f"  Deleted hparam: {key}")
            changed = True
        else:
            print(f"  Warning: hparam '{key}' not found.")

    if not changed:
        print("No changes specified. Use `--help` for edit options.")
        return

    cfg["features"]         = features
    cfg["hyperparameters"]  = hparams
    cfg["parameter_bounds"] = bounds

    engine.save_strategy_config(name, cfg)
    print(f"\n[+] Saved and synced: {name}")


def cmd_backtest(engine: ModelEngine, args) -> None:
    """
    Run a vectorized backtest for a strategy against one or more tickers.

    Fetches OHLCV data via the engine's DataBroker, computes features,
    runs generate_signals(), and scores each ticker with Tearsheet metrics.
    Progress and engine log records are streamed to stdout as they arrive.

    Output includes:
      - Core performance metrics (return, CAGR, Sharpe, drawdown, etc.)
      - Any additional metrics returned by Tearsheet
      - A trade log showing the last 5 trades per ticker
    """
    tickers   = [t.strip().upper() for t in args.tickers.split(",")]
    start_dt, end_dt = _resolve_dates(args.start, args.end)
    timeframe = {
        "start":    start_dt.isoformat(),
        "end":      end_dt.isoformat(),
        "interval": args.interval,
    }

    _header(
        f"BACKTEST  ·  {args.strategy}  ·  {', '.join(tickers)}  ·  {args.interval}\n"
        f"{start_dt.date()} → {end_dt.date()}  ·  Capital: ${args.capital:,.0f}"
    )

    callbacks = _make_callbacks()
    try:
        result = engine.run_backtest(
            args.strategy, tickers, timeframe, callbacks,
            starting_capital=args.capital,
        )
    except Exception as e:
        print(f"\n  FATAL: {e}")
        sys.exit(1)

    if result.get("cancelled"):
        print("\n  Backtest was cancelled.")
        return

    metrics_map = result.get("metrics", {})
    trade_logs  = result.get("trade_logs", {})

    if not metrics_map:
        print("\n  No results returned — check that data is available for the given tickers/interval.")
        return

    # ── Per-ticker results ────────────────────────────────────────────────────
    METRIC_DISPLAY = [
        ("total_return",     "Total Return",      "{:+.2%}"),
        ("cagr",             "CAGR",              "{:+.2%}"),
        ("sharpe_ratio",     "Sharpe Ratio",      "{:.3f}"),
        ("sortino_ratio",    "Sortino Ratio",      "{:.3f}"),
        ("calmar_ratio",     "Calmar Ratio",      "{:.3f}"),
        ("max_drawdown",     "Max Drawdown",      "{:.2%}"),
        ("win_rate",         "Win Rate",          "{:.1%}"),
        ("total_trades",     "Total Trades",      "{}"),
        ("avg_trade_return", "Avg Trade Return",  "{:+.2%}"),
        ("profit_factor",    "Profit Factor",     "{:.2f}"),
        ("volatility",       "Volatility (ann.)", "{:.2%}"),
    ]

    for ticker, metrics in metrics_map.items():
        _section(f"Results: {ticker}")

        if "error" in metrics:
            print(f"    ERROR: {metrics['error']}")
            continue

        for key, label, fmt in METRIC_DISPLAY:
            val = metrics.get(key)
            if val is None:
                continue
            try:
                formatted = fmt.format(val)
            except (ValueError, TypeError):
                formatted = str(val)
            _row(label, formatted)

        # Remaining metrics not in the display list
        shown_keys = {k for k, _, _ in METRIC_DISPLAY} | {"error"}
        extras = {k: v for k, v in metrics.items() if k not in shown_keys}
        if extras:
            print()
            for k, v in extras.items():
                if isinstance(v, float):
                    _row(k, f"{v:.4f}")
                else:
                    _row(k, v)

        # Trade log
        trades = trade_logs.get(ticker, [])
        if trades:
            print(f"\n    Trade Log — {len(trades)} trades (showing last 5)")
            print(f"    {'Entry':<12}  {'Exit':<12}  {'Side':<6}  {'Return':>8}")
            print(f"    {'-'*48}")
            for t in trades[-5:]:
                entry = str(t.get("entry_date", ""))[:10]
                exit_ = str(t.get("exit_date",  ""))[:10]
                side  = str(t.get("side", ""))
                ret   = t.get("trade_return", t.get("return"))
                ret_s = f"{ret:+.2%}" if isinstance(ret, float) else str(ret or "—")
                print(f"    {entry:<12}  {exit_:<12}  {side:<6}  {ret_s:>8}")


def cmd_train(engine: ModelEngine, args) -> None:
    """
    Run hyperparameter optimisation or model training for a strategy.

    Delegates to engine.run_training(), which uses the ApplicationController
    and Optuna (CPCV) pipeline. The engine's 'model-engine' logger is tapped
    via _CallbackLogHandler in bridge.py so trial-by-trial output, warnings,
    and errors all stream live to stdout.

    The final result dict is rendered recursively — nested dicts are indented,
    floats are shown to 4 decimal places.
    """
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    start_dt, end_dt = _resolve_dates(args.start, args.end)
    timeframe = {
        "start":    start_dt.isoformat(),
        "end":      end_dt.isoformat(),
        "interval": args.interval,
    }

    _header(
        f"TRAINING  ·  {args.strategy}  ·  {', '.join(tickers)}  ·  {args.interval}\n"
        f"{start_dt.date()} → {end_dt.date()}"
    )

    callbacks = _make_callbacks()
    try:
        result = engine.run_training(args.strategy, tickers, timeframe, callbacks)
    except Exception as e:
        print(f"\n  FATAL: {e}")
        sys.exit(1)

    if result.get("cancelled"):
        print("\n  Training was cancelled.")
        return

    _section("Training Results")
    _render_result_dict(result, indent=4)


def cmd_signal(engine: ModelEngine, args) -> None:
    """
    Generate current signals for a strategy against one or more tickers.

    Fetches a warm-up window of recent data (length determined by interval),
    runs the full feature DAG and generate_signals(), then reports the final
    signal value for each ticker. Signals are floats in [-1.0, 1.0] where
    positive = long conviction, negative = short conviction, zero = flat.
    A visual strength bar is printed alongside the numeric value.
    """
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    _header(f"SIGNAL  ·  {args.strategy}  ·  {', '.join(tickers)}")

    callbacks = _make_callbacks()
    try:
        result = engine.generate_signals(args.strategy, tickers, callbacks)
    except Exception as e:
        print(f"\n  FATAL: {e}")
        sys.exit(1)

    _section("Signal Output")
    if not result:
        print("    No signals returned.")
        return

    print(f"    {'Ticker':<10}  {'Signal':>8}  {'Direction':<8}  Strength")
    print(f"    {'-'*52}")
    for ticker, signal in result.items():
        if isinstance(signal, (int, float)):
            direction = "LONG" if signal > 0 else ("SHORT" if signal < 0 else "FLAT")
            bar = "#" * int(abs(signal) * 10)
            print(f"    {ticker:<10}  {signal:>+8.4f}  {direction:<8}  {bar}")
        else:
            print(f"    {ticker:<10}  {signal}")


# ── Shared renderer for nested result dicts ───────────────────────────────────

def _render_result_dict(data: dict, indent: int = 4) -> None:
    """
    Recursively print a result dict returned by training or signal commands.

    Nested dicts are indented by 2 extra spaces per level. Floats are
    shown to 4 decimal places; lists are printed inline; the 'cancelled'
    key is silently skipped.
    """
    pad = " " * indent
    for k, v in data.items():
        if k == "cancelled":
            continue
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _render_result_dict(v, indent + 2)
        elif isinstance(v, list):
            print(f"{pad}{k}: {v}")
        elif isinstance(v, float):
            print(f"{pad}{k:<28}  {v:.4f}")
        else:
            print(f"{pad}{k:<28}  {v}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the top-level ArgumentParser with all subcommands.

    Each subcommand maps to a cmd_* function in _DISPATCH. Subcommand
    arguments mirror the parameters of their corresponding ModelEngine
    method where possible.
    """
    parser = argparse.ArgumentParser(
        prog="research_cli.py",
        description="Research Engine CLI — mirrors GUI access via ModelEngine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  list                    Show all strategies
  features                Show all available feature IDs and their default params
  inspect  <strategy>     Show full strategy config
  init     <strategy>     Scaffold a new strategy workspace
  edit     <strategy>     Modify a strategy's manifest and sync workspace
  backtest <strategy>     Run a vectorized backtest
  train    <strategy>     Run hyperparameter optimisation / model training
  signal   <strategy>     Generate live signals
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list ────────────────────────────────────────────────────────────────────
    p = sub.add_parser("list", help="Show all strategies")
    p.add_argument("--json", action="store_true")

    # features ────────────────────────────────────────────────────────────────
    p = sub.add_parser("features", help="Show all registered features")
    p.add_argument("--json", action="store_true")

    # inspect ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("inspect", help="Show full config for a strategy")
    p.add_argument("strategy")
    p.add_argument("--json", action="store_true")

    # init ────────────────────────────────────────────────────────────────────
    p = sub.add_parser("init", help="Scaffold a new strategy workspace")
    p.add_argument("strategy", help="Name for the new strategy (valid Python identifier)")

    # edit ────────────────────────────────────────────────────────────────────
    p = sub.add_parser("edit", help="Modify a strategy manifest and sync")
    p.add_argument("strategy")
    p.add_argument("--add-feature",    metavar="ID",
                   help="Feature ID to add (e.g. RSI, MACD)")
    p.add_argument("--feature-params", metavar="k=v", nargs="+",
                   help="Params for --add-feature, space-separated k=v pairs")
    p.add_argument("--remove-feature", metavar="ID", nargs="+",
                   help="Feature ID(s) to remove (removes all entries with that ID)")
    p.add_argument("--clear-features", action="store_true",
                   help="Remove all features before applying other edits")
    p.add_argument("--set-hparam",     metavar="k=v", nargs="+",
                   help="Set/update hyperparameter(s), e.g. --set-hparam oversold=25 overbought=75")
    p.add_argument("--delete-hparam",  metavar="key", nargs="+",
                   help="Remove hyperparameter(s) by key")
    p.add_argument("--set-bound",      metavar="key=lo,hi", nargs="+",
                   help="Set parameter bound(s), e.g. --set-bound oversold=20.0,40.0")

    # backtest ────────────────────────────────────────────────────────────────
    p = sub.add_parser("backtest", help="Run a vectorized backtest")
    p.add_argument("strategy")
    p.add_argument("--tickers",  required=True,
                   help="Comma-separated ticker symbols (e.g. AAPL,MSFT)")
    p.add_argument("--interval", default="1d",
                   help="Bar interval: 1d, 1h, 15m, ... (default: 1d)")
    p.add_argument("--start",    help="Start date YYYY-MM-DD (default: 1 year ago)")
    p.add_argument("--end",      help="End date   YYYY-MM-DD (default: today)")
    p.add_argument("--capital",  type=float, default=10_000.0,
                   help="Starting capital (default: 10000)")

    # train ───────────────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="Run hyperparameter optimisation / model training")
    p.add_argument("strategy")
    p.add_argument("--tickers",  required=True)
    p.add_argument("--interval", default="1d")
    p.add_argument("--start",    help="Start date YYYY-MM-DD (default: 1 year ago)")
    p.add_argument("--end",      help="End date   YYYY-MM-DD (default: today)")

    # signal ──────────────────────────────────────────────────────────────────
    p = sub.add_parser("signal", help="Generate live signals for a strategy")
    p.add_argument("strategy")
    p.add_argument("--tickers", required=True,
                   help="Comma-separated tickers")

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

_DISPATCH = {
    "list":     cmd_list,
    "features": cmd_features,
    "inspect":  cmd_inspect,
    "init":     cmd_init,
    "edit":     cmd_edit,
    "backtest": cmd_backtest,
    "train":    cmd_train,
    "signal":   cmd_signal,
}


def main() -> None:
    """
    Entry point: parse args, instantiate ModelEngine, dispatch to a cmd_* handler.

    ModelEngine is constructed once with the same WORKSPACE_DIR and DB_PATH
    the GUI uses (defined in gui/config.py). StrategyError and ValidationError
    from the engine are caught here and printed cleanly; KeyboardInterrupt
    exits with a short message rather than a traceback.
    """
    parser = build_parser()
    args   = parser.parse_args()

    engine = ModelEngine(workspace_dir=WORKSPACE_DIR, db_path=DB_PATH)

    try:
        _DISPATCH[args.command](engine, args)
    except (StrategyError, ValidationError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
