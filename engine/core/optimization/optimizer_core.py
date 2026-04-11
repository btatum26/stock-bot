import os
import sys
import json
import pandas as pd
import itertools
import importlib.util
from typing import Dict, Any, List, Optional
from joblib import Parallel, delayed
import optuna

from ..features.features import compute_all_features
from .local_cache import LocalCache, stage_data_to_shm, load_data_from_shm
from ..data_broker.data_broker import DataBroker

def evaluate_parameters_joblib(params: dict, features_config: list, strategy_path: str) -> dict:
    """Isolated trial execution for joblib workers."""
    import sys
    import os
    import importlib.util
    from engine.core.optimization.local_cache import load_data_from_shm
    from engine.core.features.features import compute_all_features

    df_raw = load_data_from_shm()
    
    project_root_added = False
    strategy_path_added = False
    
    try:
        # Step 2: Compute features dynamically
        df_features, _ = compute_all_features(df_raw, features_config)
        
        # Step 3: Load model.py and context.py dynamically
        model_path = os.path.join(strategy_path, "model.py")
        context_path = os.path.join(strategy_path, "context.py")
        
        project_root = os.path.abspath(os.path.join(os.getcwd()))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            project_root_added = True
        if strategy_path not in sys.path:
            sys.path.insert(0, strategy_path)
            strategy_path_added = True
            
        # Load context
        spec_ctx = importlib.util.spec_from_file_location("strategy_context", context_path)
        module_ctx = importlib.util.module_from_spec(spec_ctx)
        spec_ctx.loader.exec_module(module_ctx)
        context_class = getattr(module_ctx, 'Context', None)

        # Load model
        spec = importlib.util.spec_from_file_location("strategy_worker_model", model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Step 4: Instantiate the model
        model_instance = None
        from engine.core.controller import SignalModel
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, type) and issubclass(obj, SignalModel) and obj is not SignalModel:
                model_instance = obj()
                break
        
        if not model_instance:
            raise ImportError(f"No valid SignalModel found in {model_path}")
            
        # Step 5: Train and Generate signals
        artifacts = model_instance.train(df_features, context_class, params)
        signals = model_instance.generate_signals(df_features, context_class, params, artifacts)
        
        # T+1 execution model: signal at T, enter at T+1 open, exit at T+2 open.
        # Must match Tearsheet.calculate_metrics() to avoid lookahead bias.
        returns = df_raw["open"].pct_change().shift(-2)
        strategy_returns = (signals * returns).fillna(0)

        sharpe = 0.0
        if strategy_returns.std() != 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * (252 ** 0.5)

        return {"sharpe": float(sharpe), "params": params}
        
    except Exception as e:
        return {"sharpe": -1.0, "params": params, "error": str(e)}
    finally:
        if strategy_path_added and strategy_path in sys.path:
            sys.path.remove(strategy_path)
        if project_root_added and project_root in sys.path:
            sys.path.remove(project_root)


class OptimizerCore:
    """Master Router for HPO. Orchestrates distributed trials via Joblib/Optuna."""
    
    PERMUTATION_LIMIT = 5000 
    
    def __init__(self, strategy_path: str, dataset_ref: str, manifest: dict, 
                 ticker: str = None, interval: str = None, start: str = None, end: str = None):
        self.strategy_path = strategy_path
        self.dataset_ref = dataset_ref
        self.manifest = manifest
        self.ticker = ticker
        self.interval = interval
        self.start = start
        self.end = end
        
        self.local_cache = LocalCache()
        self.broker = DataBroker()

    def discover_params(self) -> Dict[str, Any]:
        """Runs Phase A only: hyperparameter search via grid or Optuna.

        Stages data to shared memory, executes the search, cleans up,
        and returns the best parameter set found.

        Returns:
            Dictionary mapping hyperparameter names to their optimal values.
        """
        print(f"      - Starting parameter discovery for {self.dataset_ref}...")

        df = self.broker.get_data(self.ticker, self.interval, self.start, self.end)
        self.local_cache.load_to_ram(self.dataset_ref, df)

        try:
            optimal_params = self._phase_a_discovery()
        finally:
            self.local_cache.clear_cache(self.dataset_ref)

        print(f"      - Optimal parameters found: {optimal_params}")
        return optimal_params

    def run(self):
        """Runs the full optimization + training pipeline.

        Phase A discovers optimal hyperparameters. Phase B trains the
        strategy with proper data splitting via ``LocalTrainer`` and
        persists artifacts.

        Returns:
            Dictionary containing ``optimal_params`` and training results
            (``train_metrics``, ``val_metrics``, ``split_info``).
        """
        from ..trainer import LocalTrainer

        optimal_params = self.discover_params()

        print(f"      - Running Phase B: training with data splits...")
        trainer = LocalTrainer(self.strategy_path)
        df = self.broker.get_data(self.ticker, self.interval, self.start, self.end)
        results = trainer.run(df, params=optimal_params)
        results["optimal_params"] = optimal_params
        return results

    def _phase_a_discovery(self) -> Dict[str, Any]:
        """Calculates permutations and routes to Grid Search or Optuna."""
        # Support both 'hyperparameters' and 'parameters' keys
        hparams = self.manifest.get("parameter_bounds", {})
        if not hparams and "parameters" in self.manifest:
            # Reconstruct bounds from min/max/step if available
            hparams = {}
            for k, v in self.manifest["parameters"].items():
                if isinstance(v, dict) and "min" in v and "max" in v and "step" in v:
                    hparams[k] = list(range(v["min"], v["max"] + v["step"], v["step"]))
                elif isinstance(v, dict) and "min" in v and "max" in v:
                    hparams[k] = [v["min"], v["max"]] # Fallback
                elif isinstance(v, list):
                    hparams[k] = v
        
        # If no explicit ranges, check if they passed simple lists in hyperparameters
        if not hparams:
            hparams = self.manifest.get("hyperparameters", {})
            
        keys = list(hparams.keys())
        # Ensure all values are iterable (lists)
        values = [v if isinstance(v, (list, tuple)) else [v] for v in hparams.values()]
        
        try:
            permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            total_p = len(permutations)
        except Exception:
            permutations = []
            total_p = 0
            
        if total_p == 0:
            print("      - No valid hyperparameter permutations found. Using default params.")
            return {k: v[0] if isinstance(v, list) else v for k, v in hparams.items()}
            
        if total_p <= self.PERMUTATION_LIMIT:
            print(f"      - Circuit Breaker: Tier 1 (Grid Search) selected for {total_p} permutations.")
            return self._run_grid_search(permutations)
        else:
            print(f"      - Circuit Breaker: Tier 2 (Optuna) selected for {total_p} permutations.")
            return self._run_optuna_search(hparams)

    def _run_grid_search(self, permutations: list) -> Dict[str, Any]:
        """Tier 1: Brute-force evaluation of all permutations via Joblib."""
        features_config = self.manifest.get("features", [])
        
        # Use joblib to parallelize
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_parameters_joblib)(
                p, features_config, self.strategy_path
            ) for p in permutations
        )
        
        best_trial = max(results, key=lambda x: x.get("sharpe", -1.0))
        return best_trial["params"]

    def _run_optuna_search(self, param_bounds: dict) -> Dict[str, Any]:
        """Tier 2: Bayesian Optimization via Optuna."""
        features_config = self.manifest.get("features", [])
        
        # We must disable optuna's verbose logging to prevent console spam
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = {}
            for k, bounds in param_bounds.items():
                if isinstance(bounds, list) and len(bounds) >= 2:
                    if all(isinstance(x, int) for x in bounds):
                        params[k] = trial.suggest_int(k, min(bounds), max(bounds))
                    else:
                        params[k] = trial.suggest_float(k, min(bounds), max(bounds))
                elif isinstance(bounds, list) and len(bounds) == 1:
                    params[k] = bounds[0]
                else:
                    params[k] = bounds
                    
            res = evaluate_parameters_joblib(params, features_config, self.strategy_path)
            return res.get("sharpe", -1.0)
            
        study = optuna.create_study(direction="maximize")
        # Ensure we pass catch parameters to handle trial failures gracefully
        study.optimize(objective, n_trials=100, n_jobs=-1, catch=(Exception,))
        
        return study.best_params

