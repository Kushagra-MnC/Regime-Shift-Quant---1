# Regime-Shift — Macro-Aware Tactical Asset Allocation Engine

A systematic, macro-aware portfolio optimization pipeline utilizing Hidden Markov Models (HMM) for regime detection and convex optimization for dynamic asset allocation.

---

## About

This repository contains a full quantitative research pipeline for a tactical asset allocation engine. The strategy avoids traditional momentum whipsaws by classifying market regimes (Bull vs. Crisis) through a Gaussian Hidden Markov Model evaluated on realized volatility and trend quality. By dynamically soft-blending target weights based on the forward-algorithm probability of a crisis state, the engine actively manages downside risk and transaction friction across a highly liquid, index-level ETF universe.

---

## System Architecture

The pipeline is structured with strict isolation between data ingestion, signal generation, and out-of-sample backtesting to mathematically eliminate look-ahead bias.

- **Asset Universe**: Broad market index ETFs representing distinct risk premiums (NIFTYBEES, JUNIORBEES, GOLDBEES, LIQUIDBEES).
- **Regime Detection Engine**: 2-state Gaussian Hidden Markov Model (HMM) trained on 6 macro features.
- **State Alignment**: States are aligned using a composite risk-on score (`-vol_z + 0.5 × trend_quality_z`), mapping the highest-scoring state to Bull and the lowest to Crisis. Pure volatility alignment was evaluated and discarded — it failed to correctly label 2021's low-vol recovery as Bull.
- **Signal Decoding**: Implements `predict_proba` (the forward algorithm) rather than Viterbi decoding to ensure real-time, causal state classification without future data leakage.
- **Asymmetric Persistence Filter**: 1 month to enter Crisis (fast capture of shocks like Demonetisation/COVID), 3 months to exit (stays defensive through uncertain recoveries).
- **Magnitude Bypass**: If P(Crisis) ≥ 90%, the persistence filter is bypassed entirely — immediate Crisis allocation with no confirmation wait.
- **Execution Logic**: Continuous allocation soft-blending based on state probabilities to minimise TC drag and portfolio thrashing.
- **Optimizer**: CVXPY minimum-variance with Ledoit-Wolf shrinkage and regime-specific group constraints. Max Sharpe (as specified in the PS) was replaced by Min Variance because Max Sharpe is non-convex and unsolvable directly in CVXPY — regime differentiation is achieved via the equity ceiling bounds instead.

---

## Mathematical Methodology

### Dynamic State Probability

The engine calculates the conditional probability of being in a specific state at time $t$ given only the information observed up to that point:

$$P(S_t = k \mid O_{1:t})$$

### Continuous Soft-Blend Allocation

Rather than executing binary, hard-switching rebalances, the portfolio optimizer continuously blends the optimal Bull and Crisis allocations based on the real-time crisis probability ($p_t$):

$$w_t = (1 - p_t) \cdot w_{\text{bull}} + p_t \cdot w_{\text{crisis}}$$

Where $w_t$ is the final target weight vector at rebalance date $t$. Because both $w_{\text{bull}}$ and $w_{\text{crisis}}$ are feasible under their respective constraints, and the feasible set is convex, $w_t$ is guaranteed feasible for all $p_t \in [0,1]$.

---

## Repository Structure

| Notebook | Stage | Description |
|---|---|---|
| `REGIME_SHIFT_Stage1_f6.ipynb` | Data Foundation | ETF price ingestion, macro feature engineering (6 features), NSE calendar, train/holdout split |
| `REGIME_SHIFT_Stage2_f6.ipynb` | HMM Exploration | BIC grid search over n_states × covariance_type, feature stationarity checks |
| `REGIME_SHIFT_Stage3_f6.ipynb` | HMM Production | Final model fit, composite alignment, walk-forward regime assignment |
| `REGIME_SHIFT_Stage4_f6.ipynb` | Portfolio Optimizer | CVXPY min-variance, Ledoit-Wolf covariance, TC model, regime bounds |
| `REGIME_SHIFT_Stage5_final_f6.ipynb` | Walk-Forward Backtest | Soft blend, asymmetric persistence, LIQUIDBEES fix, full performance metrics |
| `REGIME_SHIFT_Stage6_f6.ipynb` | Analysis & Charts | 9 publication charts, factor decomposition, v1 vs v2 comparison |
| `REGIME_SHIFT_Stage7_f6.ipynb` | Holdout Evaluation | **Run exactly once** — sealed Jan–Nov 2024 evaluation, no re-tuning permitted |

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Kushagra-MnC/Regime-Shift-Quant---1.git
cd Regime-Shift-Quant---1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download `dataset_fixed.zip`, unzip into the project root:

```
dataset/
├── NIFTYBEES.parquet       # NIFTY BeES ETF daily prices (2009–2024)
├── JUNIORBEES.parquet      # Junior BeES ETF daily prices (2009–2024)
├── GOLDBEES.parquet        # Gold BeES ETF daily prices (2009–2024)
├── LIQUIDBEES.parquet      # Liquid BeES ETF daily prices (2009–2024)
├── NSEI.parquet            # Nifty 50 index daily prices (2009–2024)
├── USDINR.parquet          # USD/INR exchange rate daily (2009–2024)
├── GSEC_10Y.parquet        # India 10Y G-Sec yield monthly (FRED, 2011–2024)
└── INDIAVIX.parquet        # India VIX daily close (NSE, 2010–2024)
```

### 4. Update the project path

In each notebook's config cell, set `PROJECT_DIR` to your local path:

```python
PROJECT_DIR = Path(r"C:\path\to\Regime-Shift-Quant---1")
```

### 5. Run notebooks sequentially

Execute Stage 1 → Stage 2 → ... → Stage 7 in order.

> ⚠️ **Stage 7 (Holdout) runs exactly once.** Do not re-run after seeing results — this invalidates the out-of-sample evaluation.

---

## Data Integrity & Reproducibility

- **Pipeline Independence**: No runtime data fetching required after initial dataset setup.
- **Seeded HMM**: `HMM_SEED = 42`, `HMM_N_INIT = 20` — identical transition matrices on every run.
- **Expected walk-forward Sharpe**: 0.570 (107 months, Jan 2015 – Nov 2023)
- **Expected holdout Sharpe**: 2.517 (11 months, Jan – Nov 2024)

---

## PS Goals Compliance

| Goal | Status | Implementation |
|---|---|---|
| HMM Regime Classifier (Bull/Bear/Crisis without manual labelling) | ✓ | 2-state Gaussian HMM; `predict_proba` forward algorithm |
| Dynamic Constraint Mapping (objective shifts by regime) | ✓ | Min-variance with regime-specific equity ceiling bounds |
| Walk-Forward Validation Harness (no look-ahead bias) | ✓ | Annual refit, expanding window; look-ahead test PASSED (shuffled Sharpe −0.064) |
| Model Transaction Friction Explicitly (5–10 bps penalty) | ✓ | Asymmetric STT: buy = 3.5 bps, sell = 13 bps; avg drag 0.16%/yr |
| Benchmark Against Static Portfolios | ✓ | vs NIFTYBEES B&H, equal-weight, 60/40 India |

> **Note on Max Sharpe**: The PS specified Max Sharpe in Bull regime. Replaced by Min Variance with a relaxed equity ceiling (75%) because Max Sharpe is non-convex and cannot be solved directly in CVXPY. Regime differentiation is achieved through the 75% Bull vs 15% Crisis equity ceiling.

---

## Transition Matrix

*(Rows = From State, Cols = To State)*

| | Bull | Crisis |
|---|---|---|
| **Bull** | 0.7960 | 0.2040 |
| **Crisis** | 0.3720 | 0.6280 |

Crisis self-transition of 0.628 means Crisis is persistent but not permanent — consistent with historical Indian stress episodes averaging 4–6 months.

---

## Performance Tear Sheet

### Walk-Forward (In-Sample) — Jan 2015 to Nov 2023, 107 months

| Strategy | CAGR | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Win Rate |
|---|---|---|---|---|---|---|---|
| **REGIME-SHIFT v2** | 11.6% | 9.0% | **0.570** | 0.992 | **−9.5%** | **1.223** | 56% |
| NIFTYBEES B&H | 11.6% | 17.2% | 0.364 | 0.377 | −31.0% | 0.376 | 55% |
| Equal Weight | 10.4% | 8.9% | 0.448 | 0.628 | −12.5% | 0.831 | 53% |
| 60/40 India | 10.9% | 10.5% | 0.440 | 0.587 | −15.8% | 0.689 | 54% |

- **Risk-free rate**: 6.5% (RBI repo, flat)
- **TC drag**: avg round-trip 1.30 bps/rebalance · avg annual drag 0.16%

![Walk-Forward Performance](https://github.com/user-attachments/assets/2ccce53a-3bae-4328-a7f3-239f1ca1551b)

![Presentation Summary](https://github.com/user-attachments/assets/5f69d6f5-dea5-40f5-be78-4689c12459c3)

### Holdout — Jan 2024 to Nov 2024, 11 months ¹

| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar | Win Rate | N |
|---|---|---|---|---|---|---|---|---|
| **REGIME-SHIFT v2** | 21.6% | 5.4% | **2.517** | 3.405 | **−1.5%** | **14.113** | 82% | 11 |
| NIFTYBEES B&H | 11.2% | 10.7% | 0.452 | 0.539 | −7.6% | 1.470 | 50% | 12 |
| Equal Weight | 16.8% | 6.4% | 1.497 | 3.219 | −3.9% | 4.286 | 75% | 12 |
| 60/40 India | 16.4% | 6.9% | 1.327 | 2.506 | −4.7% | 3.515 | 75% | 12 |

> ¹ **N=11 for REGIME-SHIFT**: December 2024 data was available in benchmark ETF feeds but the strategy's December rebalance return requires January 2025 as the settlement month, which falls outside the evaluation window. This slightly disadvantages the strategy on the CAGR comparison.

![Holdout Evaluation](https://github.com/user-attachments/assets/6a3145a5-5bb8-4f32-9b0e-d820d45c421f)

---

## Alpha Attribution

```
REGIME-SHIFT CAGR          : 11.6%
NIFTYBEES B&H CAGR         : 11.6%
Equal Weight CAGR          : 10.4%

Alpha vs B&H               :  +0.0%  (same return, half the vol)
Alpha vs Equal Weight      :  +1.3%
Sharpe advantage vs B&H    :  +0.206
Max DD improvement vs B&H  : −21.4 pp
Calmar ratio vs B&H        :  1.223 vs 0.376  (3.25× better)

Market beta (vs NIFTYBEES) : −0.042  (effectively market-neutral)
Monthly alpha              : +1.00%  (11.99% annualised)
R�                         :  0.006  (returns uncorrelated with Nifty)

Avg annual turnover        : 157%
Avg round-trip TC          : 1.30 bps/rebalance
Estimated annual TC drag   : 0.16%

Bull timing  (Nifty ↑ in Bull months)   : 67%
Crisis timing (Nifty ↓ in Crisis months) : 45%
```

---

## Final Stage Summary

| Stage | Output |
|---|---|
| 1 | Data foundation — 4 ETFs, 8 raw series, 6 features, NSE calendar |
| 2 | HMM exploration — BIC grid, feature stationarity, rolling z-score |
| 3 | HMM production — n=2 states, composite alignment, Stage 3 config |
| 4 | Optimizer — CVXPY min-variance, Ledoit-Wolf, asymmetric TC model |
| 5 | Walk-forward — soft blend, asymmetric persistence, LIQUIDBEES fix |
| 6 | Analysis — 9 publication charts, factor decomposition, v1 vs v2 |
| **7** | **Holdout — run once, results final, no re-tuning permitted** |

**One-sentence pitch**: *We train a Hidden Markov Model on Indian macro data to detect Bull or Crisis conditions, then use CVXPY to find the minimum-variance portfolio that automatically shifts defensive as confidence in a Crisis rises — tested rigorously with no look-ahead bias and confirmed on a sealed 2024 holdout.*

---

## Installation

### Prerequisites

- Python 3.9+
- `pip install -r requirements.txt`

Dependencies: `hmmlearn`, `cvxpy`, `clarabel`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`, `yfinance`, `pandas-datareader`, `pandas-market-calendars`, `pyarrow`
