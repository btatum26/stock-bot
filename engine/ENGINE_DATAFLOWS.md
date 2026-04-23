# Engine Data Flows

Visual reference for every data path through the research engine. Diagrams use Mermaid
(renders in GitHub, VS Code, and most Markdown viewers). Each diagram annotates edge
labels with the concrete Python type crossing that boundary.

---

## 1. System Overview — All Components

```mermaid
flowchart TD
    subgraph Entry["Entry Points"]
        CLI["CLI\nresearch_cli.py / main.py\nBACKTEST|TRAIN|SIGNAL|INIT|SYNC"]
        API["FastAPI\n/api/v1/jobs\nPOST JobPayloadRequest"]
        GUI["PyQt6 GUI\nModelEngine facade"]
    end

    subgraph Core["ApplicationController"]
        AC["execute_job(JobPayload)\nroutes by ExecutionMode"]
        BT["_handle_backtest"]
        TR["_handle_train"]
        SG["_handle_signal_only"]
    end

    subgraph Data["Data Layer"]
        DB["DataBroker\nget_data(ticker, interval, start, end)"]
        SQLite[("SQLite\nstocks.db\ndiagnostics.db")]
        YF["yfinance\nexternal fetch"]
    end

    subgraph Pipeline["Execution Pipeline"]
        FE["FeatureOrchestrator\ncompute_all_features()"]
        RC["RegimeOrchestrator\nbuild_context()  ← optional"]
        LB["LocalBacktester\nrun_batch()"]
        LT["LocalTrainer\nrun()"]
        TS["Tearsheet\ncalculate_metrics()"]
        SV["SignalValidator\nvalidate_and_compress()"]
    end

    subgraph Models["User Strategy"]
        MP["model.py\nSignalModel subclass"]
        CP["context.py\nContext dataclass (generated)"]
        MJ["manifest.json\nfeatures / hparams / bounds"]
    end

    subgraph Artifacts["Artifacts"]
        AJ["artifacts.joblib\nmodel + scaler + ffd metadata"]
        MLBr["MLBridge\nprepare_training/inference_matrix()"]
    end

    subgraph Regime["Regime Subsystem (optional)"]
        RO["RegimeOrchestrator"]
        RD["RegimeDetector\nvix_adx | term_structure | hmm"]
        BCPD["BayesianCPD\nnovelty scorer"]
        RCTX["RegimeContext\nproba + labels + novelty"]
    end

    subgraph Queue["Async Path"]
        RQ["Redis Queue (RQ)"]
        WK["RQ Worker\ntasks.process_job()"]
        RS[("Redis\njob state hashes")]
    end

    CLI -->|"JobPayload (dict)"| AC
    API -->|"JobPayloadRequest → enqueue"| RQ
    GUI -->|"JobPayload"| AC
    RQ -->|"job_id + payload"| WK
    WK -->|"JobPayload"| AC
    WK <-->|"QUEUED→RUNNING→DONE"| RS

    AC --> BT & TR & SG
    BT & SG -->|"ticker list"| DB
    TR -->|"ticker list"| DB
    DB <-->|"OHLCV pd.DataFrame"| SQLite
    DB <-->|"OHLCV pd.DataFrame"| YF

    BT -->|"Dict[ticker, pd.DataFrame]"| LB
    SG -->|"Dict[ticker, pd.DataFrame]"| LB
    TR -->|"Dict[ticker, pd.DataFrame]"| LT

    LB -->|"features_config list"| FE
    FE -->|"(df_full, l_max)"| LB
    LB -->|"df_clean"| RC
    RC -->|"RegimeContext"| LB

    LB -->|"df, context, params, artifacts"| MP
    MP -->|"raw pd.Series"| SV
    SV -->|"pd.Series [-1,1]"| LB
    LB -->|"Dict[ticker, pd.Series]"| TS
    TS -->|"Dict[str, scalar|pd.Series]"| BT

    LT -->|"df_train, context, params"| MP
    MP -->|"artifacts dict"| LT
    LT -->|"(X, y)"| MLBr
    MLBr -->|"scaled arrays"| MP
    LT -->|"artifacts dict"| AJ

    MJ -->|"features / hparams"| LB & LT
    CP -->|"Context instance"| MP

    RO --> RD & BCPD
    RD -->|"pd.DataFrame (T x n_states)"| RCTX
    BCPD -->|"pd.Series [0,1]"| RCTX
    RCTX -->|"regime_context kwarg"| MP
```

---

## 2. BACKTEST Path — Request to Tearsheet

```mermaid
flowchart TD
    START(["BACKTEST\nJobPayload"]) --> INC["trial_counter.increment()\nDiagnosticsDB"]
    INC --> FETCH["DataBroker.get_data()\nfor each ticker"]
    FETCH -->|"Dict[str, pd.DataFrame]\nOHLCV, DatetimeIndex"| RB["LocalBacktester\nrun_batch()"]

    RB --> LOAD["_load_user_model_and_context()\nimportlib — once per batch"]
    LOAD -->|"(model_class, context_class)"| LOOP

    subgraph LOOP["Per-Ticker Loop"]
        CF["compute_all_features()\nFeatureOrchestrator"]
        CF -->|"(df_full: pd.DataFrame,\nl_max: int)"| WP
        WP["Warmup purge\ndf_clean = df_full.iloc[l_max:]"]
        WP -->|"df_clean: pd.DataFrame"| PN
        PN["Price normalisation\nMLBridge (if training.price_normalization ≠ 'none')"]
        PN --> AU["_audit_nans()\nlog warnings only"]
        AU --> RC_OPT{"regime_aware\nin manifest?"}
        RC_OPT -->|"No"| BR
        RC_OPT -->|"Yes"| RCB["_build_regime_context(df_clean)\nRegimeOrchestrator.build_context()"]
        RCB -->|"RegimeContext"| BR

        subgraph BR["Branch on artifacts / is_ml"]
            B1["Artifacts on disk\nor passed in"]
            B2["ML, no artifacts\n80/20 temporal split"]
            B3["Rule-based\ntrain() inline"]
        end

        BR --> GS["_call_generate_signals()\ninspect.signature → 4-arg or 5-arg"]
        GS -->|"raw: pd.Series (any range)"| SV["SignalValidator\nvalidate_and_compress()"]
        SV -->|"pd.Series [-1.0, 1.0]"| LOOP_OUT["signals per ticker"]
    end

    LOOP_OUT -->|"Dict[str, pd.Series]"| TS["Tearsheet.calculate_metrics()\nfor each ticker"]
    TS -->|"Dict[str, scalar]\n+ equity_curve, trade_log stripped"| OUT(["all_metrics\nDict[ticker, Dict]"])
```

---

## 3. TRAIN Path — Request to Artifacts

```mermaid
flowchart TD
    START(["TRAIN\nJobPayload"]) --> CHK{"ENABLE_HPO\n+ bounds defined?"}

    CHK -->|"Yes (Phase A)"| HPO["OptimizerCore\n(Optuna or grid search)\nObjective: Sharpe on first ticker"]
    HPO -->|"searched_params: dict"| MERGE
    CHK -->|"No"| MERGE["merge with manifest\nhyperparameters"]
    MERGE -->|"optimal_params: dict"| TR

    TR["LocalTrainer.run()\nDict[ticker, df] or single df"]

    TR --> CF["compute_all_features()\nper ticker"]
    CF -->|"df_full, l_max"| WP["warmup purge\ndf_clean"]
    WP --> FFD["apply_ffd_to_dataframe()\nnon-stationary columns only"]
    FFD -->|"df_clean: pd.DataFrame"| SPLIT

    subgraph SPLIT["Data Splitting"]
        S1["temporal\n80/20 chronological"]
        S2["cpcv\nCPCVSplitter\nC(n_groups, k_test) folds"]
    end

    SPLIT -->|"List[Fold]: (train_idx, val_idx)"| FOLDS

    subgraph FOLDS["Per-Fold Training"]
        subgraph ML_PATH["ML (is_ml: true)"]
            BL["build_labels(df_train)\n→ pd.Series target"]
            FM["fit_model(X, y, params)\n→ artifacts dict"]
        end
        subgraph RB_PATH["Rule-based"]
            TR2["model.train(df_train)\n→ artifacts dict"]
        end
        SCALER["MLBridge.prepare_training_matrix()\nMinMaxScaler.fit(train only)\n→ (scaled_df, scaler)"]
    end

    FOLDS --> FA["Feature analysis\nfeature_importances + Pearson + MI"]
    FOLDS --> FD["Fold diagnostics\nOOS Sharpe, Spearman IS/OOS"]
    FD -->|"diagnostics.json"| DISK1[("strategies/<name>/\ndiagnostics.json")]

    FOLDS --> RETRAIN["Final retrain\non full dataset\n(all tickers pooled for ML)"]
    RETRAIN -->|"artifacts: dict"| AM["ArtifactManager.save_artifacts()"]
    AM -->|"artifacts.joblib"| DISK2[("strategies/<name>/\nartifacts.joblib")]

    FOLDS --> OUT(["results: dict\nfold_metrics, fold_diagnostics\noptimal_params"])
```

---

## 4. Feature Computation Pipeline

```mermaid
flowchart LR
    MJ["manifest.json\nfeatures: List[{id, params}]"]
    MJ -->|"features_config: List[dict]"| ORC

    subgraph ORC["FeatureOrchestrator.compute_features()"]
        VAL["validate_config()\ncheck registry membership"]
        VAL --> ITER["Iterate features_config"]
        ITER -->|"feature_id, params"| REG

        subgraph REG["FEATURE_REGISTRY\nDict[str, Type[Feature]]"]
            RSI_CLS["RSI"]
            MACD_CLS["MACD"]
            ADX_CLS["ADX"]
            MACRO_CLS["NFCI · T10Y2Y · HYSpread\nVIXCLS · DFF · ICSA\nANFCI · T10Y3M\nVIXTermStructure"]
            DOTS["... 40+ features total\nmomentum · trend · volatility\nlevels · volume · calendar\ncomparison · alternative · options · macro"]
        end

        REG -->|"feature_cls"| INST["feature_cls()\ninstantiate"]
        INST -->|"feature.compute(df, params, cache)"| CACHE

        subgraph CACHE["FeatureCache"]
            MEM["_memory: Dict[str, pd.Series]\nshared across features\navoids recomputing dependencies"]
        end

        CACHE -->|"FeatureResult\n.data: Dict[str, pd.Series]\n.levels / .zones / .heatmaps"| COLCHECK

        COLCHECK["column-count invariant check\nraise FeatureError if df mutated"]
        COLCHECK -->|"new Series only"| CONCAT["pd.concat([df, new_cols])"]
        CONCAT --> LMAX["update l_max\nmax(period, window, slow, fast, lookback)"]
    end

    ORC -->|"(df_full: pd.DataFrame,\nl_max: int)"| DOWN["Backtester / Trainer\ndf_clean = df_full.iloc[l_max:]"]
```

**Column naming convention:** `Feature.generate_column_name(feature_id, params, output_name)`
produces deterministic names: `RSI_14`, `BollingerBands_20_2.0_UPPER`, `MACD_SIGNAL`,
`T10Y2Y_level`, `T10Y2Y_roc5`, `T10Y2Y_zscore`, `VIXTermStructure`, `VIXTermStructure_zscore`.
Every run on the same manifest yields identical column names.

**Macro features note.** FRED and VIX term structure features fetch external data inside
`compute()` and align it to `df.index` via forward-fill — they require no changes to the
orchestrator. The `_level` output of every `FredFeature` subclass is declared non-stationary
via `non_stationary_outputs()`, so the ML bridge routes it through FFD before scaling.

---

## 5. Regime Detection Pipeline

```mermaid
flowchart TD
    TRIG["Backtester\nmanifest: regime_aware=true\nmanifest: regime_detector=X"]

    TRIG -->|"df_clean: pd.DataFrame\n(OHLCV + features)"| RO

    subgraph RO["RegimeOrchestrator.build_context()"]
        MF["_build_macro_features(df)"]

        subgraph MF_DETAIL["macro_features assembly"]
            YF2["yfinance.download(\n^VIX, ^VIX3M, SPY, HYG)\naligned to df.index via ffill"]
            ADX_CALC["_compute_adx(df)\nWilder ADX from df's OHLCV"]
        end

        MF --> YF2 & ADX_CALC
        YF2 & ADX_CALC -->|"macro_features: pd.DataFrame\ncolumns: vix, vix3m, vix_vix3m,\nadx, spy_ret, spy_rvol, hy_spread_chg"| DET_SEL

        DET_SEL{"Select detector\nfrom REGIME_REGISTRY"}

        subgraph DETS["Detectors"]
            VA["VixAdxRegime\nfit(macro) → no-op\npredict_proba(macro)\n→ one-hot (T x 3)\nnovelty_score() → 0"]
            TS2["TermStructureRegime\nfit(macro) → no-op\npredict_proba(macro)\n→ blended (T x 2)\nnovelty_score() → 0"]
            HMM["GaussianHMMRegime\nfit(macro) → hmmlearn.fit(X)\npredict_proba(macro)\n→ forward algorithm (T x 3)\nnovelty_score()\n→ -log P(obs|model) normalised"]
        end

        DET_SEL --> VA & TS2 & HMM
        VA & TS2 & HMM -->|"proba: pd.DataFrame\ndetector_novelty: pd.Series"| BOCPD_RUN

        BOCPD_RUN["BayesianCPD.run(vix_std)\nP(run_length < 5)\n→ novelty: pd.Series [0,1]"]

        NOVELTY_SEL{"detector_name == 'hmm'?"}
        BOCPD_RUN --> NOVELTY_SEL
        NOVELTY_SEL -->|"Yes → use BOCPD novelty"| ALIGN
        NOVELTY_SEL -->|"No → max(BOCPD, detector_novelty)"| ALIGN

        ALIGN["Align to df.index\nffill + fillna\ncompute labels = argmax(proba)"]
    end

    ALIGN -->|"RegimeContext:\ndetector_name: str\nproba: pd.DataFrame (T x n_states)\nlabels: pd.Series[int]\nnovelty: pd.Series[float]\nn_states: int\nic_weight: None (Phase 3.4 pending)"| DISPATCH

    DISPATCH["_call_generate_signals()\ninspect.signature(model.generate_signals)\nhas 'regime_context'?"]
    DISPATCH -->|"Yes → 5-arg call"| MP_AWARE["model.generate_signals(\ndf, context, params,\nartifacts, regime_context=ctx)"]
    DISPATCH -->|"No → 4-arg call"| MP_LEGACY["model.generate_signals(\ndf, context, params, artifacts)"]
```

**BOCPD internals** (Adams & MacKay 2007, Normal-Gamma conjugate):

```
At each step t:
  1. Compute Student-T predictive P(x_t | run_length=r, history) for all r ∈ [0..mrl]
  2. Update run-length posterior R[r] via:
       growth:     R_new[r+1] = R[r] * pred[r] * (1 - H)
       changepoint: R_new[0]  = Σ_r R[r] * pred[r] * H
  3. Normalise R_new
  4. novelty[t] = Σ_{r<5} R_new[r]   (P that run started < 5 bars ago)
  5. Update Normal-Gamma sufficient stats {μ, κ, α, β} for surviving hypotheses
```

---

## 6. ML Bridge — Training and Inference Matrices

```mermaid
flowchart TD
    subgraph TRAIN_PATH["Training Matrix (fit)"]
        DF_TRAIN["df_train: pd.DataFrame\n(first 80% or CPCV fold)"]
        FEAT_COLS["feature_cols: List[str]\nexclude open/high/low/close/volume"]
        DF_TRAIN & FEAT_COLS --> PM["prepare_training_matrix(\ndf, feature_cols, l_max)\n→ MinMaxScaler.fit(X_train)\nfeature_range=(-1,1)"]
        PM -->|"scaled_df: pd.DataFrame\nscaler: MinMaxScaler"| ART["artifacts dict\nartifacts['system_scaler'] = scaler\nartifacts['feature_cols'] = cols"]

        FFD_CHECK{"non_stationary_outputs()\nreturns columns?"}
        DF_TRAIN --> FFD_CHECK
        FFD_CHECK -->|"Yes"| FFD["apply_ffd_to_dataframe(\ndf, columns, d=0.4, window=10)\nLópez de Prado fixed-window FFD\npath-dependent: run per-ticker"]
        FFD --> PM
        FFD_CHECK -->|"No"| PM
    end

    subgraph INFER_PATH["Inference Matrix (transform only)"]
        DF_EVAL["df_eval: pd.DataFrame\n(full range or new bar)"]
        ART2["artifacts dict\nfrom training or disk"]
        DF_EVAL & ART2 --> PIM["prepare_inference_matrix(\ndf, feature_cols, l_max=0,\nartifacts, is_live=False)\n→ scaler.transform(X)  ← NEVER fit\nreplay FFD if ffd_columns in artifacts"]
        PIM -->|"scaled_df: pd.DataFrame"| GS2["model.generate_signals()"]
    end

    subgraph PERSIST["ArtifactManager"]
        SAVE["save_artifacts(strategy_dir, artifacts)\n→ artifacts.joblib"]
        LOAD["load_artifacts(strategy_dir)\n→ artifacts dict or None"]
    end

    ART --> SAVE
    LOAD --> ART2
```

---

## 7. Data Broker — SQLite Caching Protocol

```mermaid
flowchart TD
    REQ["DataBroker.get_data(\nticker, interval, start, end)"]

    REQ --> CLAMP{"intraday?\n1m=7d, 2-30m=60d\n60m/1h=730d"}
    CLAMP -->|"clamped start/end"| CHECK["_compute_fetch_range()\ncompare requested range\nto DB min/max dates"]

    CHECK -->|"DB empty"| FULL["fetch full range\nfrom yfinance"]
    CHECK -->|"forward gap\ncache stale"| FWD["fetch (last_cached → end)\nfrom yfinance"]
    CHECK -->|"backward gap\nolder history needed"| BWD["fetch (start → first_cached)\nfrom yfinance"]
    CHECK -->|"both gaps"| BOTH["fetch both ranges separately"]
    CHECK -->|"no gaps + within\nstaleness window"| HIT["return from SQLite\ncache hit"]

    FULL & FWD & BWD & BOTH -->|"raw pd.DataFrame"| MERGE2["INSERT OR IGNORE INTO ohlcv\n(ticker, timestamp, interval)\nbatch of 124 rows"]
    MERGE2 --> RETURN["SELECT * WHERE ticker AND interval\nAND timestamp BETWEEN start AND end\n→ pd.DataFrame, DatetimeIndex UTC-naive"]

    HIT --> RETURN

    subgraph STALENESS["Staleness windows (no fetch if within)"]
        D["daily → 4 days"]
        W["weekly → 10 days"]
        H["hourly → 2 days"]
        M15["15m → bypass always (no cache)"]
    end
```

---

## 8. API and Worker — Redis State Machine

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Redis
    participant RQ Worker
    participant Controller

    Client->>FastAPI: POST /api/v1/jobs\n{strategy, assets, interval, mode}
    FastAPI->>Redis: HSET job:{id} status=QUEUED
    FastAPI->>Redis: RPUSH rq:queue:default task_ref
    FastAPI-->>Client: 202 {job_id}

    Client->>FastAPI: GET /api/v1/jobs/{job_id}  (polling)
    FastAPI->>Redis: HGET job:{id}
    FastAPI-->>Client: {status: QUEUED}

    Redis-->>RQ Worker: dequeue task
    RQ Worker->>Redis: WATCH job:{id}
    RQ Worker->>Redis: HSET status=RUNNING
    RQ Worker->>Controller: process_job(job_id, payload)
    Controller-->>RQ Worker: results dict

    alt results < 1 MB
        RQ Worker->>Redis: MULTI\nHSET status=COMPLETED results=json\nEXEC (optimistic lock)
    else results ≥ 1 MB
        RQ Worker->>RQ Worker: write artifacts/<id>.json
        RQ Worker->>Redis: HSET artifact_value=FILE_PATH:...
    end

    Client->>FastAPI: GET /api/v1/jobs/{job_id}
    FastAPI->>Redis: HGET job:{id}
    FastAPI-->>Client: {status: COMPLETED, results: {...}}

    note over Client,Redis: Cancel path
    Client->>FastAPI: POST /api/v1/jobs/{job_id}/cancel
    FastAPI->>Redis: HSET status=CANCEL_REQUESTED
    RQ Worker->>Redis: check before/during commit
    RQ Worker->>Redis: HSET status=CANCELLED
```

**Job state machine:**

```
QUEUED ──► RUNNING ──► COMPLETED
                  └──► FAILED
                  └──► CANCELLED  (via CANCEL_REQUESTED)
```

---

## 9. Training — CPCV Fold Structure

```mermaid
flowchart LR
    subgraph DATA["Full dataset (n bars)"]
        G1["Group 1"]
        G2["Group 2"]
        G3["Group 3"]
        G4["Group 4"]
        G5["Group 5"]
        G6["Group 6"]
    end

    subgraph FOLDS["C(6,2) = 15 folds  (n_groups=6, k_test=2)"]
        F1["Fold 1\ntest: G1+G2 | train: G3-G6"]
        F2["Fold 2\ntest: G1+G3 | train: G2,G4-G6"]
        DOTS2["... 13 more folds"]
        F15["Fold 15\ntest: G5+G6 | train: G1-G4"]
    end

    subgraph PURGE["Purge + Embargo per fold"]
        PG["Remove l_max bars before\neach test block (lookback leak)"]
        EM["Remove embargo_pct×n bars after\neach test block (label autocorrel)"]
    end

    DATA --> FOLDS
    FOLDS --> PURGE
    PURGE -->|"(X_train, y_train)\n(X_val, y_val)"| FIT["fit_model(X_train, y_train)\ngenerate_signals(X_val)\nTearsheet OOS metrics"]
    FIT -->|"fold_sharpe: float\nfold_artifacts"| AGG["aggregate across folds\nmean/std OOS metrics\nSpearman IS/OOS"]
    AGG --> FINAL["Final retrain\non full pooled data\nsave artifacts.joblib"]
```

---

## 10. Key Data Types Reference

| Type | Shape / Structure | Where produced | Where consumed |
|---|---|---|---|
| `JobPayload` | Pydantic model: strategy, assets, interval, mode, timeframe | CLI / API / GUI | `ApplicationController.execute_job()` |
| `pd.DataFrame` (OHLCV) | columns: open, high, low, close, volume; DatetimeIndex UTC-naive | `DataBroker.get_data()` | `LocalBacktester`, `LocalTrainer`, `FeatureOrchestrator` |
| `pd.DataFrame` (feature-enriched) | OHLCV + N feature columns; same DatetimeIndex | `FeatureOrchestrator.compute_features()` | `LocalBacktester.run()`, `RegimeOrchestrator`, `MLBridge` |
| `(pd.DataFrame, int)` | `(df_full, l_max)` | `compute_all_features()` | Backtester warmup purge |
| `FeatureResult` | dataclass: data, levels, zones, heatmaps | `Feature.compute()` | `FeatureOrchestrator` |
| `pd.DataFrame` (macro feature columns) | FRED / yfinance series as dated columns in the feature DataFrame: `<ID>_level`, `<ID>_roc5`, `<ID>_zscore`, `VIXTermStructure`, `VIXTermStructure_zscore` | `FredFeature.compute()`, `VIXTermStructure.compute()` via `FeatureOrchestrator` | `SignalModel.generate_signals()` via `df[ctx.features.<col>]` |
| `pd.DataFrame` (regime macro) | internal regime inputs — columns: vix, adx, spy_ret, spy_rvol, hy_spread_chg, vix_vix3m; **not** in the strategy DataFrame | `RegimeOrchestrator._build_macro_features()` (direct yfinance fetch, independent of FEATURE_REGISTRY) | `RegimeDetector.fit()`, `BayesianCPD.run()` |
| `RegimeContext` | dataclass: proba (T×K), labels (T,), novelty (T,), n_states | `RegimeOrchestrator.build_context()` | `SignalModel.generate_signals()` (opt-in) |
| `pd.DataFrame` (proba) | columns = int state IDs; index = df.index; row sums to 1.0 | `RegimeDetector.predict_proba()` | `RegimeContext` |
| `pd.Series` (novelty) | float in [0,1]; index = macro.index | `BayesianCPD.run()` | `RegimeContext.novelty`, `size_multiplier` |
| `pd.Series` (raw signals) | any numeric range; user-defined index | `SignalModel.generate_signals()` | `SignalValidator.validate_and_compress()` |
| `pd.Series` (signals) | float in [-1.0, 1.0]; index = df_clean.index | `SignalValidator.validate_and_compress()` | `Tearsheet.calculate_metrics()` |
| `artifacts` dict | `{"model": est, "system_scaler": scaler, "feature_cols": [...], ...}` | `model.train()` / `model.fit_model()` | `model.generate_signals()`, `ArtifactManager.save_artifacts()` |
| `numpy.ndarray` (X) | shape `(n_samples, n_features)`, float64, scaled to [-1,1] | `MLBridge.prepare_training_matrix()` | `model.fit_model()` |
| `numpy.ndarray` (y) | shape `(n_samples,)`, int labels | `model.build_labels()` | `model.fit_model()` |
| Tearsheet `dict` | scalar metrics + equity_curve, portfolio, bh_portfolio, trade_log | `Tearsheet.calculate_metrics()` | CLI output, GUI charts |
| `ICResult` | dataclass: mean_ic, ic_ir, ic_series[horizon], quintile_returns, ... | `ICAnalyzer.compute()` | `research_cli.py ic` report, `ConditionalIC` |
| `ConditionalICResult` | dataclass: bins (sorted RegimeBin list), diagnosis, unconditional_ic | `ConditionalIC.compute()` | `research_cli.py ic-surface` report |

---

## 11. Strategy Manifest → Context → Model Data Flow

```mermaid
flowchart LR
    MJ2["manifest.json\n───────────\nfeatures: [{id, params}]\nhyperparameters: {k: v}\nparameter_bounds: {k: [lo,hi]}\nis_ml: bool\nregime_aware: bool\nregime_detector: str\ncompression_mode: str\ntraining: {split, n_groups, ...}"]

    MJ2 -->|"features list"| WS["WorkspaceManager.sync()\nJinja2 templates"]
    WS -->|"generate"| CTX["context.py (auto-generated)\n───────────\n@dataclass FeaturesContext:\n    RSI_14: str = 'RSI_14'\n    MACD_SIGNAL: str = 'MACD_SIGNAL'\n    ...\n@dataclass ParamsContext:\n    stop_loss: float = 0.05\n    ...\n@dataclass Context:\n    features: FeaturesContext\n    params: ParamsContext"]

    CTX -->|"Context()"| MP2["model.py\n───────────\nclass MyStrategy(SignalModel):\n    def generate_signals(\n        self, df, context,\n        params, artifacts,\n        regime_context=None):\n        rsi = df[context.features.RSI_14]\n        ..."]

    MJ2 -->|"hyperparameters"| MP2
    MJ2 -->|"features_config"| FE2["FeatureOrchestrator\n→ df with named columns"]
    FE2 -->|"df: pd.DataFrame"| MP2

    MP2 -->|"pd.Series"| SV2["SignalValidator\n→ pd.Series [-1,1]"]
```

---

## 12. End-to-End Type Flow Summary

```mermaid
flowchart LR
    A(["User request\nstr: strategy\nList[str]: tickers\nstr: interval\nExecutionMode"]) 
    A -->|"JobPayload"| B["ApplicationController"]
    B -->|"ticker, interval, dates"| C["DataBroker"]
    C -->|"pd.DataFrame OHLCV\n(T rows × 5 cols)"| D["FeatureOrchestrator"]
    D -->|"pd.DataFrame\n(T rows × (5+N) cols)\nint: l_max"| E["LocalBacktester"]
    E -->|"pd.DataFrame\n(T-l_max rows × (5+N) cols)"| F{"regime_aware?"}
    F -->|"Yes → pd.DataFrame"| G["RegimeOrchestrator"]
    G -->|"RegimeContext\n(proba T×K, novelty T)"| H["SignalModel\n.generate_signals()"]
    F -->|"No → pd.DataFrame"| H
    H -->|"pd.Series (raw)"| I["SignalValidator"]
    I -->|"pd.Series [-1,1]\n(T-l_max,)"| J{"mode?"}
    J -->|"BACKTEST"| K["Tearsheet\n→ Dict[str, scalar]"]
    J -->|"SIGNAL_ONLY"| L["last value\nfloat + timestamp"]
    K --> M(["Result\nDict[ticker, metrics]"])
    L --> M
```
