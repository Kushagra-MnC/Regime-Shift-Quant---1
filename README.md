# Regime-Shift Macro-Aware Tactical Asset Allocation Engine

A systematic, macro-aware portfolio optimization pipeline utilizing Hidden Markov Models (HMM) for regime detection and convex optimization for dynamic asset allocation.

## About
This repository contains a full quantitative research pipeline for a tactical asset allocation engine. The strategy avoids traditional momentum whipsaws by classifying market regimes (Bull vs. Crisis) through a Gaussian Hidden Markov Model evaluated on realized volatility. By dynamically soft-blending target weights based on the forward-algorithm probability of a crisis state, the engine actively manages downside risk and transaction friction across a highly liquid, index-level ETF universe.

## System Architecture
The pipeline is structured with strict isolation between data ingestion, signal generation, and out-of-sample backtesting to mathematically eliminate look-ahead bias.

*   **Asset Universe**: Broad market index ETFs representing distinct risk premiums (NIFTYBEES, JUNIORBEES, GOLDBEES, LIQUIDBEES).
*   **Regime Detection Engine**: The project utilizes a 2-state Gaussian Hidden Markov Model (HMM).
*   **State Alignment**: States are aligned objectively using mean realized volatility (Low Volatility = Bull, High Volatility = Crisis).
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
   git clone https://github.com/Kushagra-MnC/regime-shift-allocation.git
   cd regime-shift-allocation
   ```

2. **Create the environment and install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure `hmmlearn`, `cvxpy`, `pandas`, and `numpy` are correctly configured).*

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
  
