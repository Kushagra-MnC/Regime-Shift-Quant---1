# Regime-Shift — Macro-Aware Tactical Asset Allocation Engine

A systematic, macro-aware portfolio optimization pipeline utilizing Hidden Markov Models (HMM) for regime detection and convex optimization for dynamic asset allocation.

---

## About

This repository contains a full 7-stage quantitative research pipeline for a tactical asset allocation engine. The strategy classifies market regimes (Bull vs. Crisis) through a Gaussian HMM evaluated on a 7-feature rolling z-score panel. By dynamically soft-blending target weights based on the forward-algorithm probability of a Crisis state, the engine actively manages downside risk and transaction friction across a liquid, index-level Indian ETF universe.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Kushagra-MnC/Regime-Shift-Quant---1.git
cd Regime-Shift-Quant---1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download all data (one-time setup, ~3 minutes)
python download_data.py

# 4. Run notebooks in order — Stage 1 through Stage 7
```

> **Note:** If India VIX auto-download fails (yfinance coverage is intermittent),
> `download_data.py` prints exact manual-download instructions for nseindia.com.
> All other data sources download automatically.

---

## System Architecture

The pipeline is structured with strict isolation between data ingestion, signal
generation, and out-of-sample backtesting to mathematically eliminate look-ahead bias.

| Component | Description |
|-----------|-------------|
| **Asset Universe** | 4 NSE ETFs across distinct risk premiums: NIFTYBEES, JUNIORBEES, GOLDBEES, LIQUIDBEES |
| **Regime Detection** | 2-state Gaussian HMM (n=2; BIC-selected over n=3 and n=4) |
| **State Alignment** | Broad risk-on composite score: `+0.35×nifty_mom_z − 1.00×realized_vol_z + 0.15×trend_quality_z − 0.25×vix_z − 0.10×inr_stress − 0.05×gsec_mom_z` |
| **Signal Decoding** | `predict_proba` (forward algorithm) — causal, no future data leakage |
| **Persistence Filter** | Asymmetric: 1 month to enter Crisis, 3 months to exit |
| **Execution** | Continuous soft-blending of Bull/Crisis weights on posterior probability |
| **Optimizer** | CVXPY minimum-variance with Ledoit-Wolf shrinkage and regime-specific bounds |

---

## Design Decisions (Deviations from the Problem Statement)

Two deliberate departures from the original PS are documented here, each supported by statistical and financial evidence.

### 1 — n=2 states (Bull / Crisis) instead of Bull / Bear / Crisis

The BIC grid search returned **ΔBIC = 8.3 points** favouring n=2 over n=3. A gap above 6 is conventionally "strong" evidence (Kass & Raftery, 1995). Attempting n=3 with full covariance on 140 observations requires 89 parameters — severely over-parametrised. In practice, n=3 produced a degenerate 4-month "Bull" cluster with repeated EM non-convergence warnings, while the remaining 136 months collapsed into a single "Bear" blob. The 2-state model produces economically meaningful clusters: Bull (low vol, positive momentum) and Crisis (high vol, negative momentum).

### 2 — Minimum-Variance objective for all regimes instead of Maximum-Sharpe in Bull

Maximum Sharpe is non-convex and requires estimated expected returns (μ). Sample mean returns at monthly frequency have near-zero predictive power for next-month returns (R² < 0.02 historically). Substituting an unreliable μ estimate into the objective introduces more noise than signal.

The regime differentiation is instead achieved through **dynamic weight bounds**: Bull allows up to 65% NIFTYBEES + 35% JUNIORBEES; Crisis caps equity at 10% and floors defensives (GOLDBEES + LIQUIDBEES) at 85%. The result is the same economic intuition — aggressive in Bull, defensive in Crisis — without the instability of return estimation.

---

## Mathematical Methodology

### 7-Feature Rolling Z-Score Panel

All features use a **12-month rolling z-score** (not expanding) to ensure stressed months are measured relative to recent experience rather than all-time history:

$$
z_t = \frac{x_t - \mu_{t-252:t}}{\sigma_{t-252:t}}
$$

| # | Feature | Economic Signal |
|---|---------|----------------|
| 1 | `nifty_mom_z` | 21-day return z-score (direction) |
| 2 | `realized_vol_z` | 21-day realised vol z-score (volatility level) |
| 3 | `vix_z` | India VIX z-score (implied stress) |
| 4 | `inr_stress` | 63-day USD/INR return z-score (macro stress) |
| 5 | `gsec_mom_z` | 21-day G-Sec yield change z-score (rate momentum) |
| 6 | `trend_quality_z` | 21d return / realised vol, z-scored (risk-adjusted momentum) |
| 7 | `nifty_trend_126d_z` | 126-day Nifty trend z-score (medium-term direction) |

### Dynamic State Probability (Forward Algorithm)

The engine calculates the conditional probability of being in a specific state at time $t$ given only observations up to that point — no future data:

$$P(S_t = k \mid O_{1:t})$$

### Continuous Soft-Blend Allocation

Rather than hard-switching between regimes, the optimizer blends target weights continuously based on real-time crisis probability $p_t$:

$$w_t = (1 - p_t) \cdot w_{\text{bull}} + p_t \cdot w_{\text{crisis}}$$

### Transaction Cost Model (India-Specific)

India's asymmetric tax structure is modelled explicitly:

| ETF | Buy (one-way) | Sell (one-way) | Note |
|-----|-------------|---------------|------|
| NIFTYBEES / JUNIORBEES | 3.5 bps | 13.0 bps | STT = 10 bps on sell only |
| GOLDBEES | 2.5 bps | 2.5 bps | No equity STT on gold ETFs |
| LIQUIDBEES | 1.0 bps | 1.0 bps | Liquid — minimal spread |

Actual per-asset TC is deducted from monthly portfolio returns in the walk-forward loop (not post-hoc estimated).

### HMM Transition Matrix

*(Rows = From, Cols = To — n=2, full covariance, BIC-selected)*

| | Bull | Crisis |
|---|------|--------|
| **Bull** | 0.7960 | 0.2040 |
| **Crisis** | 0.3720 | 0.6280 |

Crisis self-loop (0.628) is lower than Bull's (0.796), meaning Crisis periods are naturally shorter-lived in the data — which motivates the asymmetric persistence filter (fast entry, slow exit).

---

## Repository Structure

```
Regime-Shift-Quant---1/
├── download_data.py                    ← Run first — downloads all data to dataset/
├── requirements.txt                    ← pip install -r requirements.txt
├── README.md
│
├── REGIME_SHIFT_Stage1_f6.ipynb        ← Stage 1: Data Foundation
├── REGIME_SHIFT_Stage2_f6.ipynb        ← Stage 2: HMM Exploration
├── REGIME_SHIFT_Stage3_f6.ipynb        ← Stage 3: HMM Production (BIC grid, full-sample fit)
├── REGIME_SHIFT_Stage4_f6.ipynb        ← Stage 4: Optimizer (CVXPY, isolated tests)
├── REGIME_SHIFT_Stage5_final_f6.ipynb  ← Stage 5: Walk-Forward Integration
├── REGIME_SHIFT_Stage6_f6.ipynb        ← Stage 6: Benchmark Analysis (9 charts)
└── REGIME_SHIFT_Stage7_f6.ipynb        ← Stage 7: Holdout 2024 (run once, sealed)
```

### Stage Summary

| Stage | Notebook | Output |
|-------|----------|--------|
| 1 | `REGIME_SHIFT_Stage1_f6.ipynb` | Data foundation — 4 ETFs, 7 features, NSE calendar, train/holdout split |
| 2 | `REGIME_SHIFT_Stage2_f6.ipynb` | HMM exploration — BIC grid, v1 walk-forward baseline |
| 3 | `REGIME_SHIFT_Stage3_f6.ipynb` | HMM production — n=2 final model, all 5 quality checks, 6/6 events ✓ |
| 4 | `REGIME_SHIFT_Stage4_f6.ipynb` | Optimizer — CVXPY min-variance, India TC model, feasibility tests |
| 5 | `REGIME_SHIFT_Stage5_final_f6.ipynb` | Walk-forward — soft blend, asymmetric persistence, net-of-TC returns |
| 6 | `REGIME_SHIFT_Stage6_f6.ipynb` | Benchmark analysis — 4 benchmarks, 9 publication charts, attribution |
| **7** | **`REGIME_SHIFT_Stage7_f6.ipynb`** | **Holdout — run once, 2024 sealed result, overfitting diagnostic** |

---

## Data Setup

All data is downloaded from free public sources via `download_data.py`. No paid API keys required.

```
dataset/
├── etf/
│   ├── NIFTYBEES.parquet     ← NSE large-cap equity ETF
│   ├── JUNIORBEES.parquet    ← NSE mid/large-cap ETF (Nifty Next 50)
│   ├── GOLDBEES.parquet      ← Gold ETF
│   └── LIQUIDBEES.parquet    ← Overnight liquid ETF (proxy for cash)
├── signal/
│   ├── NSEI.parquet          ← Nifty 50 index (signal construction)
│   ├── USDINR.parquet        ← USD/INR spot rate
│   └── INDIAVIX.parquet      ← India VIX (NSE implied volatility index)
└── macro/
    └── GSEC_10Y.parquet      ← India 10Y G-Sec yield (FRED: INDIRLTLT01STM)
```

**If India VIX download fails** (yfinance coverage is intermittent):
1. Go to: https://www.nseindia.com/products-services/indices-vix
2. Scroll to *Historical Data* tab → set From: `01-06-2009`, To: `31-12-2024`
3. Download CSV → save as `dataset/signal/INDIAVIX.csv`
4. Re-run `python download_data.py` — it auto-detects the manual file

---

## Performance Results

### Walk-Forward Backtested Performance (2015–2023, 107 months)

> These results are out-of-sample relative to each annual HMM refit but are within the strategy development period. The true out-of-sample result is the 2024 holdout below.

| Strategy | CAGR | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Win Rate |
|----------|------|----------|--------|---------|--------|--------|----------|
| **REGIME-SHIFT v2** | **11.6%** | 9.0% | **0.570** | **0.992** | **-9.5%** | **1.223** | 56% |
| NIFTYBEES B&H | 11.6% | 17.2% | 0.364 | 0.377 | -31.0% | 0.376 | 55% |
| Equal Weight | 10.4% | 8.9% | 0.448 | 0.628 | -12.5% | 0.831 | 53% |
| 60/40 India | 10.9% | 10.5% | 0.440 | 0.587 | -15.8% | 0.689 | 54% |

- Risk-free rate: 6.5% (RBI repo, flat approximation)
- Avg annual turnover: 157% · Avg round-trip TC: 1.30 bps/month · Annual TC drag: ~0.16%
- Bull timing (Nifty ↑ in Bull months): 67% · Crisis timing (Nifty ↓ in Crisis months): 45%

**REGIME-SHIFT v2 ranks #1 on Sharpe, Sortino, Calmar, and Max DD vs all four benchmarks.**
It matches B&H's raw CAGR (11.6%) while cutting max drawdown by 3.3× and delivering 3.25× better Calmar.

### Holdout Performance — Jan 2024 → Dec 2024 (Sealed, Run Once)

| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar |
|----------|------|-----|--------|---------|--------|--------|
| **REGIME-SHIFT v2** | **21.6%** | 5.4% | **2.517** | 3.405 | **-1.5%** | **14.113** |
| NIFTYBEES B&H | 11.2% | 10.7% | 0.452 | 0.539 | -7.6% | 1.470 |
| Equal Weight | 16.8% | 6.4% | 1.497 | 3.219 | -3.9% | 4.286 |
| 60/40 India | 16.4% | 6.9% | 1.327 | 2.506 | -4.7% | 3.515 |

> **2024 context:** The Crisis portfolio (10% NIFTYBEES + 45% GOLDBEES + 45% LIQUIDBEES) captured GOLDBEES's +19.7% CAGR in 2024 (geopolitical safe-haven gold rally) while the asymmetric persistence filter prevented whipsaw re-entry into equities. The holdout Sharpe of 2.517 is partly a tail-event outcome (gold bull market) and should not be taken as the strategy's forward-looking Sharpe expectation; the walk-forward 0.570 is the more conservative baseline.

---

## Alpha Attribution

```
REGIME-SHIFT CAGR    :  11.6%       Max DD    :  -9.5%
NIFTYBEES B&H CAGR   :  11.6%       B&H Max DD: -31.0%

Alpha vs B&H         :   0.0%       Sharpe advantage vs B&H : +0.206
Alpha vs Equal Weight:  +1.3%       Calmar: 1.223 vs 0.376

Regime timing alpha  :  +61.2 bps/yr
Within-regime alpha  :  -18.2 bps/yr  (min-var optimizer, conservative by design)
TC drag              :   -9.7 bps/yr

Look-ahead test      :  Real timing Sharpe +1.321 vs shuffled -0.064  ✓ PASSED
```

---

## Reproducibility Notes

- All notebooks use `random_state` seeds (HMM: `seed=42`, multi-start: seeds 42–61) for deterministic results
- Stage 7 is sealed — do not run until Stages 1–6 are complete. Running it re-uses the model parameters frozen at Stage 3; no re-fitting occurs on holdout data
- HMM convergence warnings in 2015–2016 folds are expected (training window < 60 months). Results before 2018 are labelled "warm-up" in Stage 6 charts

---

## Project Summary

**We train a Hidden Markov Model on Indian macro data to detect Bull or Crisis conditions, then use CVXPY to find the minimum-variance portfolio that automatically shifts defensive as confidence in a Crisis rises — tested rigorously with walk-forward validation and confirmed on a sealed 2024 holdout.**
# Regime-Shift Macro-Aware Tactical Asset Allocation Engine

A systematic, macro-aware portfolio optimization pipeline utilizing Hidden Markov Models (HMM) for regime detection and convex optimization for dynamic asset allocation.

## About
This repository contains a full quantitative research pipeline for a tactical asset allocation engine. The strategy avoids traditional momentum whipsaws by classifying market regimes (Bull vs. Crisis) through a Gaussian Hidden Markov Model evaluated on realized volatility. By dynamically soft-blending target weights based on the forward-algorithm probability of a crisis state, the engine actively manages downside risk and transaction friction across a highly liquid, index-level ETF universe.

**The PS suggested Max Sharpe in Bull; we use Min Variance with regime-specific equity ceiling bounds instead, since Max Sharpe is non-convex and cannot be solved directly in CVXPY.**

## System Architecture
The pipeline is structured with strict isolation between data ingestion, signal generation, and out-of-sample backtesting to mathematically eliminate look-ahead bias.

*   **Asset Universe**: Broad market index ETFs representing distinct risk premiums (NIFTYBEES, JUNIORBEES, GOLDBEES, LIQUIDBEES).
*   **Regime Detection Engine**: The project utilizes a 2-state Gaussian Hidden Markov Model (HMM).
*   **State Alignment**: States are aligned objectively using composite risk-on score (-vol_z + 0.5×mom_z) (Low Volatility = Bull, High Volatility = Crisis).
*   **Signal Decoding**: Implements `predict_proba` (the forward algorithm) rather than Viterbi decoding to ensure real-time, causal state classification without future data leakage.
*   **Asymmetric Persistence Filter**: Imposes a strict regime transition logic (1 month to enter Crisis, 3 months to exit) to prevent premature re-entry into risk assets during volatile macroeconomic recoveries.
*   **Execution Logic**: Continuous allocation soft-blending based on state probabilities to minimize transaction cost (TC) drag and portfolio turnover.

## Mathematical Methodology

### Dynamic State Probability
The engine calculates the conditional probability of being in a specific state at time $t$ given only the information observed up to that point:

$$
P(S_t = k \mid O_{1:t})
$$

### Continuous Soft-Blend Allocation
Rather than executing binary, hard-switching rebalances, the portfolio optimizer continuously blends the optimal Bull and Crisis covariance constraints based on the real-time crisis probability ($p_t$):

$$
w_t = (1 - p_t) \cdot w_{\text{bull}} + p_t \cdot w_{\text{crisis}}
$$

Where $w_t$ represents the final target weight vector at rebalance date $t$.

## Repository Structure
The workflow is broken down into modular, highly reproducible stages:

*   `01_Data_Foundation.ipynb`: Ingestion, cleaning, and formatting of ETF price data and macro features.
*   `02_HMM_Training.ipynb`: Unsupervised learning of the 2-state regime model and volatility alignment.
*   `03_Walk_Forward_Inference.ipynb`: Implementation of the forward algorithm to generate causal P(Crisis) probabilities.
*   `04_Portfolio_Optimization.ipynb`: CVXPY implementation of the soft-blend allocation vectors.
*   `05_Friction_and_Constraints.ipynb`: Application of transaction costs (TC) and the asymmetric persistence filter.
*   `06_Backtest_Engine.ipynb`: Generation of core performance metrics and comparison against baselines.
*   `07_Holdout_Sanctity_2024.ipynb`: Final, single-run evaluation on quarantined out-of-sample 2024 data to verify absence of hyperparameter overfitting.

## Data Integrity & Reproducibility
To guarantee absolute determinism in backtesting and avoid dependency on external APIs (which are prone to downtime and data revisions), this engine operates on a local, immutable dataset.

*   **Data Source**: The project utilizes locally stored Parquet files located in the `dataset/` directory.

```text
dataset/
├── etf_prices.parquet
├── macro_features.parquet
└── metadata.json
```

*   **Pipeline Independence**: No runtime data fetching is required. This ensures the engine is "air-gapped," allowing for identical result generation regardless of internet connectivity or API availability.
*   **Setup**: Users must ensure the `dataset/` directory is populated with the requisite price and macro feature files before executing the Stage 1 notebook.

## Installation

1. **Clone the repository**:
   ```bash
   https://github.com/Kushagra-MnC/Regime-Shift-Quant---1
   cd Regime-Shift-Quant---1.git
   ```

2. **Create the environment and install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure `hmmlearn`, `cvxpy`, `pandas`, and `numpy` are correctly configured).*
   hmmlearn==0.3.2
cvxpy==1.4.2
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
yfinance>=0.2.36
pandas-datareader>=0.10
pandas-market-calendars>=4.3
scipy>=1.11
clarabel

## Execution
Run the notebooks strictly in sequential order (01 through 07).

> [!NOTE]
> **Holdout Data**: Notebook `07_Holdout_Sanctity_2024.ipynb` accesses a quarantined dataset. The random seed parameters for the HMM initialization are hardcoded to ensure identical transition matrix generation upon reproduction.

### Transition Matrix
*(Rows = From, Cols = To)*

| | Bull | Crisis |
| :--- | :---: | :---: |
| **Bull** | 0.7960 | 0.2040 |
| **Crisis** | 0.3720 | 0.6280 |

*(Diagonal = probability of staying in the same regime next month)*

## Performance Tear Sheet (Out-of-Sample)
The strategy was evaluated against standard static allocation benchmarks across the holdout period.

| Strategy | CAGR | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Win Rate | Months |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **REGIME-SHIFT** | 11.6% | 9.0% | 0.570 | 0.992 | -9.5% | 1.223 | 56% | 107 |
| **NIFTYBEES B&H** | 11.6% | 17.2% | 0.364 | 0.377 | -31.0% | 0.376 | 55% | 107 |
| **Equal Weight** | 10.4% | 8.9% | 0.448 | 0.628 | -12.5% | 0.831 | 53% | 107 |
| **60/40 India** | 10.9% | 10.5% | 0.440 | 0.587 | -15.8% | 0.689 | 54% | 107 |

*   **Risk-free rate**: 6.5% (RBI repo, flat)
*   **TC drag summary**:
    *   Avg round-trip TC: 1.30 bps/month
    *   Avg buy TC: 0.35 bps/month
    *   Avg sell TC: 0.95 bps/month

  
  <img width="1313" height="914" alt="walk forward performance" src="https://github.com/user-attachments/assets/2ccce53a-3bae-4328-a7f3-239f1ca1551b" />
  
  <img width="1306" height="899" alt="image" src="https://github.com/user-attachments/assets/5f69d6f5-dea5-40f5-be78-4689c12459c3" />
  
  Alpha Attribution
  ==================================================
    REGIME-SHIFT CAGR          : 11.6%
    NIFTYBEES B&H CAGR         : 11.6%
    Equal Weight CAGR          : 10.4%
  
    Alpha vs B&H               : +0.0%
    Alpha vs Equal Weight      : +1.3%
  
    Sharpe advantage vs B&H    : +0.206
    Max DD improvement vs B&H  : -21.4%
    Calmar ratio vs B&H        : 1.223 vs 0.376
  
    Avg annual turnover        : 157%
    Avg round-trip TC          : 1.30 bps/rebalance
    Estimated annual TC drag   : 0.16%
  
    Bull timing  (Nifty ↑ in Bull months) : 67%
    Crisis timing (Nifty ↓ in Crisis months): 45%

  ======================================================================
HOLDOUT PERFORMANCE  —  Jan 2024 → Dec 2024  (12 months)

| Strategy | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar | WinRate | N |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **REGIME-SHIFT v2** | 21.6% | 5.4% | 2.517 | 3.405 | -1.5% | 14.113 | 82% | 11 |
| **NIFTYBEES B&H** | 11.2% | 10.7% | 0.452 | 0.539 | -7.6% | 1.470 | 50% | 12 |
| **Equal Weight** | 16.8% | 6.4% | 1.497 | 3.219 | -3.9% | 4.286 | 75% | 12 |
| **60/40 India** | 16.4% | 6.9% | 1.327 | 2.506 | -4.7% | 3.515 | 75% | 12 |


<img width="1609" height="1110" alt="image" src="https://github.com/user-attachments/assets/6a3145a5-5bb8-4f32-9b0e-d820d45c421f" />

  
  ### Final project summary
  | Stage | Output |
  |-------|--------|
  | 1 | Data foundation — 4 ETFs, 6 features, NSE calendar |
  | 2 | HMM exploration — feature engineering, BIC grid |
  | 3 | HMM production — n=2 states, composite alignment |
  | 4 | Optimizer — CVXPY min-variance, Ledoit-Wolf, TC model |
  | 5 | Walk-forward — soft blend, asymmetric persist |
  | 6 | Analysis — 9 publication charts, factor decomposition |
  | **7** | **Holdout — run once, results final** |

  
**Project Summary** :  **We train a Hidden Markov Model on Indian macro data to detect Bull or Crisis conditions, then use CVXPY to find the minimum-variance portfolio that automatically shifts defensive as confidence in a Crisis rises — tested rigorously with no look-ahead bias and confirmed on a sealed 2024 holdout.**
  
