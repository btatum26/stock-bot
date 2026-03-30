import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import json
import requests
import sys
from datetime import datetime

# Add project root to sys.path to allow internal imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .controller import ExecutionMode, JobPayload, Timeframe, MultiAssetMode
from .workspace import WorkspaceManager
from .config import config

class EngineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Research Engine - Control Panel")
        self.root.geometry("800x900")
        
        self.api_url = config.api_url
        self.strategies_dir = "src/strategies"
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self._setup_logging()
        self._create_widgets()
        self._refresh_strategies()
        
        self.connect_to_daemon()
        self._poll_jobs()

    def _setup_logging(self):
        """Initializes the session log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"gui_log_{timestamp}.txt")

    def _log(self, message):
        """Displays message in console and writes to log file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.console.insert(tk.END, formatted_message)
        self.console.see(tk.END)
        try:
            with open(self.log_file, "a") as f:
                f.write(formatted_message)
        except Exception:
            pass

    def connect_to_daemon(self):
        """Verifies connection to the compute daemon."""
        try:
            # Verify network connectivity
            res = requests.get(f"{self.api_url}/health", timeout=2)
            if res.status_code == 200:
                self._log(f"Connected to Daemon at {self.api_url}")
            else:
                self._show_offline_warning("Daemon responded with an error.")
                
        except Exception as e:
            self._show_offline_warning(f"Connection to Daemon failed at {self.api_url}")

    def _show_offline_warning(self, msg):
        """Helper to alert user of connectivity issues."""
        messagebox.showwarning("Daemon Disconnected", msg)
        self._log(f"Warning: {msg}")

    def _create_widgets(self):
        """Main UI layout initialization."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Strategy Selector
        selector_frame = ttk.LabelFrame(main_frame, text="Strategy Selector", padding="5")
        selector_frame.pack(fill=tk.X, pady=5)

        ttk.Label(selector_frame, text="Select Strategy:").pack(side=tk.LEFT, padx=5)
        self.strategy_var = tk.StringVar()
        self.strategy_dropdown = ttk.Combobox(selector_frame, textvariable=self.strategy_var, state="readonly")
        self.strategy_dropdown.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.strategy_dropdown.bind("<<ComboboxSelected>>", self._on_strategy_selected)

        ttk.Button(selector_frame, text="Refresh", command=self._refresh_strategies).pack(side=tk.LEFT, padx=5)
        ttk.Button(selector_frame, text="Create New", command=self._create_new_strategy_popup).pack(side=tk.LEFT, padx=5)

        # Strategy Configurator
        self.config_frame = ttk.LabelFrame(main_frame, text="Strategy Configurator", padding="5")
        self.config_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.config_canvas = tk.Canvas(self.config_frame)
        self.config_scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=self.config_canvas.yview)
        self.scrollable_config = ttk.Frame(self.config_canvas)

        self.scrollable_config.bind(
            "<Configure>",
            lambda e: self.config_canvas.configure(scrollregion=self.config_canvas.bbox("all"))
        )

        self.config_canvas.create_window((0, 0), window=self.scrollable_config, anchor="nw")
        self.config_canvas.configure(yscrollcommand=self.config_scrollbar.set)

        self.config_canvas.pack(side="left", fill="both", expand=True)
        self.config_scrollbar.pack(side="right", fill="y")

        # Execution Routing
        routing_frame = ttk.LabelFrame(main_frame, text="Execution Routing", padding="5")
        routing_frame.pack(fill=tk.X, pady=5)

        param_frame = ttk.Frame(routing_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Ticker:").pack(side=tk.LEFT, padx=2)
        self.ticker_var = tk.StringVar(value="AAPL")
        ttk.Entry(param_frame, textvariable=self.ticker_var, width=10).pack(side=tk.LEFT, padx=5)

        ttk.Label(param_frame, text="Interval:").pack(side=tk.LEFT, padx=2)
        self.interval_var = tk.StringVar(value="1h")
        self.interval_dropdown = ttk.Combobox(param_frame, textvariable=self.interval_var, values=["1m", "5m", "15m", "1h", "1d"], width=5)
        self.interval_dropdown.pack(side=tk.LEFT, padx=5)

        button_frame = ttk.Frame(routing_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Sync Data", command=self._sync_data).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Run Backtest", command=lambda: self.submit_job("BACKTEST")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Run Grid Search", command=lambda: self.submit_job("TRAIN")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Bundle", command=self._bundle_artifact).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Progress Tracker
        progress_frame = ttk.Frame(main_frame, padding="5")
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

        # Output Console
        console_frame = ttk.LabelFrame(main_frame, text="Output Console", padding="5")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=15, state="normal")
        self.console.pack(fill=tk.BOTH, expand=True)

    def _refresh_strategies(self):
        """Scan directory for available strategy folders."""
        if not os.path.exists(self.strategies_dir):
            os.makedirs(self.strategies_dir)
        
        strategies = [d for d in os.listdir(self.strategies_dir) if os.path.isdir(os.path.join(self.strategies_dir, d))]
        self.strategy_dropdown['values'] = strategies
        if strategies:
            if not self.strategy_var.get():
                self.strategy_var.set(strategies[0])
            self._on_strategy_selected()

    def _on_strategy_selected(self, event=None):
        """Load manifest when a new strategy is chosen."""
        strategy = self.strategy_var.get()
        if not strategy:
            return
        
        manifest_path = os.path.join(self.strategies_dir, strategy, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            self._populate_configurator()
        else:
            self._log(f"Warning: manifest.json not found for {strategy}")

    def _populate_configurator(self):
        """Dynamically build the configuration form from manifest data."""
        for widget in self.scrollable_config.winfo_children():
            widget.destroy()

        ttk.Label(self.scrollable_config, text="Hyperparameters", font=("", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.hp_entries = {}
        hparams = self.manifest.get("hyperparameters", self.manifest.get("parameters", {}))
        
        for key, val in hparams.items():
            frame = ttk.Frame(self.scrollable_config)
            frame.pack(fill=tk.X, padx=10, pady=2)
            ttk.Label(frame, text=key, width=20).pack(side=tk.LEFT)
            
            display_val = val.get("default", val) if isinstance(val, dict) else val
            
            var = tk.StringVar(value=str(display_val))
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.hp_entries[key] = var

        ttk.Label(self.scrollable_config, text="Features (JSON)", font=("", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.features_text = scrolledtext.ScrolledText(self.scrollable_config, height=5)
        self.features_text.pack(fill=tk.X, padx=10, pady=2)
        self.features_text.insert(tk.END, json.dumps(self.manifest.get("features", []), indent=4))

        ttk.Label(self.scrollable_config, text="Parameter Bounds (JSON)", font=("", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.bounds_text = scrolledtext.ScrolledText(self.scrollable_config, height=5)
        self.bounds_text.pack(fill=tk.X, padx=10, pady=2)
        
        bounds = self.manifest.get("parameter_bounds", {})
        self.bounds_text.insert(tk.END, json.dumps(bounds, indent=4))

        ttk.Button(self.scrollable_config, text="Save Changes", command=self._save_manifest).pack(pady=10)

    def _save_manifest(self):
        """Update manifest.json with current form values."""
        strategy = self.strategy_var.get()
        if not strategy:
            return

        try:
            new_hparams = {}
            for key, var in self.hp_entries.items():
                val = var.get()
                try:
                    if '.' in val: new_hparams[key] = float(val)
                    else: new_hparams[key] = int(val)
                except ValueError:
                    new_hparams[key] = val

            new_features = json.loads(self.features_text.get("1.0", tk.END))
            new_bounds = json.loads(self.bounds_text.get("1.0", tk.END))

            self.manifest["hyperparameters"] = new_hparams
            self.manifest["features"] = new_features
            self.manifest["parameter_bounds"] = new_bounds

            manifest_path = os.path.join(self.strategies_dir, strategy, "manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=4)
            
            self._log(f"Manifest saved for {strategy}")
            messagebox.showinfo("Success", f"Manifest for {strategy} updated.")
        except Exception as e:
            self._log(f"Error saving manifest: {e}")
            messagebox.showerror("Error", f"Failed to save manifest: {e}")

    def _sync_data(self):
        """Standardize local development environment."""
        strategy = self.strategy_var.get()
        if not strategy: return
        self._save_manifest()
        
        try:
            self._log(f"Syncing {strategy} locally...")
            self.progress_var.set(20)
            
            strat_path = os.path.join(self.strategies_dir, strategy)
            wm = WorkspaceManager(strat_path)
            
            with open(wm.manifest_path, 'r') as f:
                manifest = json.load(f)
            
            features = manifest.get('features', [])
            hparams = manifest.get('hyperparameters', {})
            if not hparams and "parameters" in manifest:
                hparams = {k: v.get("default") if isinstance(v, dict) else v for k, v in manifest["parameters"].items()}
            
            bounds = manifest.get('parameter_bounds', {})
            if not bounds and "parameters" in manifest:
                bounds = {k: [v.get("min"), v.get("max")] for k, v in manifest["parameters"].items() if isinstance(v, dict) and "min" in v}

            wm.sync(features, hparams, bounds)
            
            self.progress_var.set(100)
            self._log("Sync complete. context.py updated.")
        except Exception as e:
            self._log(f"Sync failed: {e}")
            messagebox.showerror("Sync Error", str(e))

    def submit_job(self, mode):
        """Send job request to the compute daemon."""
        if not self.api_url:
            messagebox.showerror("Error", "API URL not configured.")
            return

        strategy = self.strategy_var.get()
        ticker = self.ticker_var.get()
        interval = self.interval_var.get()
        
        if not strategy or not ticker:
            messagebox.showwarning("Input Required", "Please select a strategy and ticker.")
            return

        payload = {
            "strategy": strategy,
            "assets": [ticker],
            "interval": interval,
            "mode": mode,
            "timeframe": {"start": None, "end": None}
        }
            
        try:
            res = requests.post(f"{self.api_url}/submit", json=payload)
            if res.status_code == 200:
                job_id = res.json().get("job_id")
                self._log(f"Job Queued: {job_id}")
            else:
                self._log(f"Job submission failed: {res.text}")
        except Exception as e:
            messagebox.showerror("Network Error", str(e))

    def _poll_jobs(self):
        """Update job statuses periodically."""
        if self.api_url:
            try:
                res = requests.get(f"{self.api_url}/api/v1/jobs", timeout=2)
                if res.status_code == 200:
                    jobs = res.json()
                    if jobs:
                        latest_job = jobs[0]
                        progress = latest_job.get("progress", 0.0)
                        self.progress_var.set(progress)
            except Exception:
                pass 
                
        self.root.after(2000, self._poll_jobs)

    def _bundle_artifact(self):
        """Prepare strategy for export."""
        strategy = self.strategy_var.get()
        if not strategy: return
        
        try:
            self._log(f"Bundling {strategy}...")
            strat_path = os.path.join(self.strategies_dir, strategy)
            export_path = "exports"
            from .bundler import Bundler
            bundle_file = Bundler.export(strat_path, export_path)
            self._log(f"Artifact created: {bundle_file}")
        except Exception as e:
            self._log(f"Bundling failed: {e}")
            messagebox.showerror("Export Error", str(e))

    def _create_new_strategy_popup(self):
        """New strategy creation wizard."""
        popup = tk.Toplevel(self.root)
        popup.title("Create New Strategy")
        popup.geometry("300x150")
        
        ttk.Label(popup, text="Strategy Name:").pack(pady=5)
        name_var = tk.StringVar()
        ttk.Entry(popup, textvariable=name_var).pack(pady=5)
        
        def create():
            name = name_var.get().strip().lower().replace(" ", "_")
            if not name: return
            
            strat_path = os.path.join(self.strategies_dir, name)
            if os.path.exists(strat_path):
                messagebox.showerror("Error", "Strategy already exists.")
                return
            
            os.makedirs(strat_path)
            
            manifest = {
                "strategy_name": f"{name.replace('_', ' ').title()}",
                "description": "Baseline moving average crossover strategy.",
                "parameters": {
                    "fast_window": {"type": "int", "default": 10, "min": 5, "max": 25, "step": 1},
                    "slow_window": {"type": "int", "default": 50, "min": 30, "max": 100, "step": 5}
                },
                "features": [
                    {"id": "SMA_Fast", "params": {"window": 10}},
                    {"id": "SMA_Slow", "params": {"window": 50}}
                ]
            }
            with open(os.path.join(strat_path, "manifest.json"), 'w') as f:
                json.dump(manifest, f, indent=4)
            
            model_content = """import numpy as np
import pandas as pd
from engine.core.controller import SignalModel
from .context import Context

class Model(SignalModel):

    def train(self, df: pd.DataFrame, context: Context, params: dict) -> dict:
        \"\"\"Executes the training/optimization logic.\"\"\"
        return {}

    def generate_signals(self, df: pd.DataFrame, context: Context, params: dict, artifacts: dict) -> pd.Series:
        \"\"\"Vectorized signal generation logic.\"\"\"
        fast_ma = df[context.SMA_FAST]
        slow_ma = df[context.SMA_SLOW]

        # Logic: Fast > Slow = Long, Fast < Slow = Short
        signals = np.where(fast_ma > slow_ma, 1.0, -1.0)
        
        # Valid data mask
        is_valid = fast_ma.notna() & slow_ma.notna()
        signals = np.where(is_valid, signals, 0.0)

        return pd.Series(signals, index=df.index, dtype=np.float64)
"""
            with open(os.path.join(strat_path, "model.py"), 'w') as f:
                f.write(model_content)
            
            wm = WorkspaceManager(strat_path)
            hparams = {k: v["default"] for k, v in manifest["parameters"].items()}
            bounds = {k: [v["min"], v["max"]] for k, v in manifest["parameters"].items()}
            wm.sync(manifest["features"], hparams, bounds)

            self._refresh_strategies()
            self.strategy_var.set(name)
            self._on_strategy_selected()
            
            self._log(f"Created strategy: {name}")
            popup.destroy()

        ttk.Button(popup, text="Create", command=create).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = EngineGUI(root)
    root.mainloop()
