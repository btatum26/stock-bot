import argparse
import sys
import os
import subprocess
import json
from datetime import datetime, timedelta
import pandas as pd

from engine.core.controller import ApplicationController, JobPayload, ExecutionMode
from engine.core.workspace import WorkspaceManager
from engine.core.logger import logger
from engine.core.exceptions import EngineError

def handle_init(strategy_name: str, strategies_dir: str = "src/strategies"):
    """Scaffolds a new strategy directory with default boilerplate."""
    strat_dir = os.path.join(strategies_dir, strategy_name)
    
    if os.path.exists(strat_dir):
        logger.error(f"Strategy '{strategy_name}' already exists at {strat_dir}")
        sys.exit(1)
        
    try:
        os.makedirs(strat_dir)
        
        # Define a bare-bones default configuration
        default_features = [
            {"id": "sma", "params": {"period": 50, "source": "close"}},
            {"id": "rsi", "params": {"window": 14, "source": "close"}}
        ]
        default_hparams = {"stop_loss": 0.05, "take_profit": 0.10}
        default_bounds = {"stop_loss": [0.01, 0.1]}
        
        # Bootstrap files via WorkspaceManager
        wm = WorkspaceManager(strategy_dir=strat_dir)
        wm.sync(features=default_features, hparams=default_hparams, bounds=default_bounds)
        
        print(f"\n[+] Successfully initialized strategy: '{strategy_name}'")
        print(f"    Location: {strat_dir}")
        print("\nNext Steps:")
        print("  1. Edit manifest.json to configure your features and parameters.")
        print(f"  2. Run `python main.py SYNC --strategy {strategy_name}` to update your context.")
        print("  3. Write your trading logic in model.py.")
        
    except Exception as e:
        logger.error(f"Failed to initialize strategy: {e}")
        if os.path.exists(strat_dir):
            import shutil
            shutil.rmtree(strat_dir)
        sys.exit(1)


def handle_sync(strategy_name: str, strategies_dir: str = "src/strategies"):
    """Recompiles context.py and model.py from the existing manifest.json."""
    strat_dir = os.path.join(strategies_dir, strategy_name)
    manifest_path = os.path.join(strat_dir, "manifest.json")
    
    if not os.path.exists(strat_dir):
        logger.error(f"Strategy '{strategy_name}' does not exist. Run INIT first.")
        sys.exit(1)
        
    if not os.path.exists(manifest_path):
        logger.error(f"manifest.json not found in {strat_dir}.")
        sys.exit(1)
        
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        features = manifest.get("features", [])
        hparams = manifest.get("hyperparameters", {})
        bounds = manifest.get("parameter_bounds", {})
        
        wm = WorkspaceManager(strategy_dir=strat_dir)
        wm.sync(features=features, hparams=hparams, bounds=bounds)
        
        print(f"\n[+] Successfully synced workspace for '{strategy_name}'")
        print("    context.py has been updated with your latest manifest configurations.")
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {manifest_path}. Please fix formatting errors.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to sync strategy workspace: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Research Engine CLI")
    parser.add_argument("mode", nargs="?", choices=["BACKTEST", "TRAIN", "SIGNAL", "INIT", "SYNC"], help="Execution mode")
    parser.add_argument("--strategy", help="Strategy folder name")
    parser.add_argument("--ticker", help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--interval", default="1h", help="Data interval (e.g., 1h, 1d)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--gui", action="store_true", help="Launch the Graphical User Interface")

    args = parser.parse_args()

    if args.gui:
        logger.info("Launching GUI...")
        gui_path = os.path.join("src", "gui_launcher.py")
        subprocess.Popen([sys.executable, gui_path])
        return

    if not args.mode:
        parser.print_help()
        return

    # --- Pre-Execution Tooling Modes ---
    if args.mode in ["INIT", "SYNC"]:
        if not args.strategy:
            print(f"Error: --strategy is required for {args.mode} mode.")
            sys.exit(1)
            
        if args.mode == "INIT":
            handle_init(args.strategy)
        elif args.mode == "SYNC":
            handle_sync(args.strategy)
            
        return

    # --- Standard Execution Modes ---
    if not args.strategy or not args.ticker:
        print("Error: --strategy and --ticker are required for execution modes.")
        sys.exit(1)

    # Resolve Default Dates at the CLI level
    try:
        if args.end:
            end_dt = datetime.strptime(args.end, '%Y-%m-%d')
        else:
            end_dt = datetime.now()
            
        if args.start:
            start_dt = datetime.strptime(args.start, '%Y-%m-%d')
        else:
            start_dt = end_dt - timedelta(days=365) # Default 1 year lookback
            
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)

    mode_map = {
        "BACKTEST": ExecutionMode.BACKTEST,
        "TRAIN": ExecutionMode.TRAIN,
        "SIGNAL": ExecutionMode.SIGNAL_ONLY
    }

    controller = ApplicationController()
    
    payload = {
        "strategy": args.strategy,
        "assets": [args.ticker],
        "interval": args.interval,
        "mode": mode_map[args.mode],
        "timeframe": {
            "start": start_str, # Now explicitly defined
            "end": end_str      # Now explicitly defined
        }
    }

    try:
        logger.info(f"Starting {args.mode} for {args.ticker} using {args.strategy}")
        result = controller.execute_job(payload)
        
        if args.mode == "BACKTEST":
            print("\n--- Backtest Results ---")
            for ticker, metrics in result.items():
                print(f"\nAsset: {ticker}")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
        
        elif args.mode == "TRAIN":
            print("\n--- Optimization Results ---")
            print(result)
            
        elif args.mode == "SIGNAL":
            print(f"\n--- Signal Results ---")
            for ticker, data in result.items():
                if "error" in data:
                    print(f"  {ticker}: Error - {data['error']}")
                else:
                    print(f"  {ticker}: {data['signal']} at {data['timestamp']}")

    except EngineError as e:
        logger.error(f"Engine execution failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()