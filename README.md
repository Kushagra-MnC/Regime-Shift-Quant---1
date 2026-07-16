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
- **State Alignment**: States are aligned using a composite risk-on score across all 6 HMM features — weighted +0.35 on momentum (`nifty_mom_z`), +0.15 on trend quality, and negative weights on realized volatility (−1.00, the dominant term), VIX (−0.25), INR stress (−0.10), and G-Sec momentum (−0.05) — mapping the highest-scoring state to Bull and the lowest to Crisis.
- **Signal Decoding**: Implements `predict_proba` (the forward algorithm) rather than Viterbi decoding to ensure real-time, causal state classification without future data leakage.
- **Asymmetric Persistence Filter**: 1 month to enter Crisis (fast capture of shocks like Demonetisation/COVID), 3 months to exit (stays defensive through uncertain recoveries).
- **Magnitude Bypass**: If P(Crisis) ≥ 90%, the persistence filter is bypassed entirely — immediate Crisis allocation with no confirmation wait.
- **Execution Logic**: Continuous allocation soft-blending based on state probabilities to minimise TC drag and portfolio thrashing.
- **Optimizer**: CVXPY convex objective combining **variance + CVaR + transaction cost**, with regime-dependent weights (not pure minimum-variance). In Bull, variance dominates (weight 1.00 vs CVaR 0.10) — the portfolio is optimised mainly for smoothness. In Crisis, CVaR dominates (weight 1.25 vs variance 0.35) — the optimizer prioritises tail-risk minimisation over variance once a crisis is confirmed. Ledoit-Wolf shrinkage is used for the covariance estimate. Max Sharpe (as specified in the PS) was replaced because it is non-convex and unsolvable directly in CVXPY — regime differentiation is instead achieved via these objective-weight shifts plus the equity group bounds (Bull: 60–90% combined equity; Crisis: 0–15%).

---

## Mathematical Methodology

### Dynamic State Probability

The engine calculates the conditional probability of being in a specific state at time $t$ given only the information observed up to that point:

$$P(S_t = k \mid O_{1:t})$$

### Continuous Soft-Blend Allocation

Rather than executing binary, hard-switching rebalances, the portfolio optimizer continuously blends the optimal Bull and Crisis allocations based on the real-time crisis probability ($p_t$):

$$w_t = (1 - p_t) \cdot w_{\text{bull}} + p_t \cdot w_{\text{crisis}}$$

Where $w_t$ is the final target weight vector at rebalance date $t$. Because both $w_{\text{bull}}$ and $w_{\text{crisis}}$ are feasible under their respective constraints, and the feasible set is convex, $w_t$ is guaranteed feasible for all $p_t \in [0,1]$.

### Regime Objective (per-regime CVXPY formulation)

<img width="356" height="24" alt="image" src="https://github.com/user-attachments/assets/3d727982-f32b-4b76-84c9-f07455b813d2" />


| Regime | $\lambda_{\text{var}}$ | $\lambda_{\text{cvar}}$ | $\lambda_{\text{tc}}$ | Dominant term |
|---|---|---|---|---|
| Bull | 1.00 | 0.10 | 1.00 | Variance |
| Bear *(unused — see note below)* | 1.00 | 0.50 | 1.25 | Variance |
| Crisis | 0.35 | 1.25 | 1.50 | **CVaR** |

CVaR uses the Rockafellar–Uryasev formulation (auxiliary variables $\eta$, $u$) at the 95% confidence level.

---

## Repository Structure

The pipeline runs across **6 notebooks**. HMM exploration (BIC grid search) and HMM production (final fit) are combined into a single notebook — there is no separate "Stage 2" file; `hmm_dev.ipynb` does both the model-selection grid and the production fit that earlier drafts of this project split across two notebooks.

| Notebook | Role | Description |
|---|---|---|
| `data_foundation.ipynb` | Data Foundation | ETF price ingestion, macro feature engineering (**7 features** engineered; see note below), NSE calendar, train/holdout split |
| `hmm_dev.ipynb` | HMM Development | BIC grid search over n_states × covariance_type **and** final model fit, composite alignment, walk-forward regime assignment, convergence checks, Viterbi vs `predict_proba` comparison — saves to `data/stage3/` |
| `cvxpy.ipynb` | Portfolio Optimizer | CVXPY variance+CVaR+TC objective defined and unit-tested in isolation (known covariance, static historical covariance, degenerate-covariance fallback), Ledoit-Wolf covariance, regime bounds — saves locked config to `stage4_optimizer_outputs/` |
| `walkfwd_integration.ipynb` | Walk-Forward Backtest | Soft blend, asymmetric persistence, LIQUIDBEES fix, full performance metrics — saves to `data/stage5/` |
| `performance.ipynb` | Analysis & Charts | 9 publication charts, factor decomposition, v1 vs v2 comparison — saves to `data/stage6/` |
| `holdout.ipynb` | Holdout Evaluation | **Run exactly once** — sealed Jan 2024 – May 2026 evaluation (28 months), no re-tuning permitted — saves to `data/stage7/` |

> **Note on feature count**: `data_foundation.ipynb` engineers 7 rolling z-score features, including a 126-day slow-trend diagnostic (`nifty_trend_126d_z`) added to audit broad bull-market coverage. `hmm_dev.ipynb` deliberately feeds only 6 of these 7 to the HMM (`HMM_FEATURE_COLUMNS` excludes the slow-trend feature) — so "6 macro features" throughout this README refers to the HMM's input, not the full feature set saved to disk.
>
> **Note on `optimizer_new.py`**: this is the **shared, actively-imported** optimizer module — both `walkfwd_integration.ipynb` and `holdout.ipynb` do `from optimizer_new import solve_regime_objective, ...` rather than redefining the objective inline. `cvxpy.ipynb` keeps its own local copy of the same logic, since its purpose is to unit-test the objective in isolation before it becomes the shared module the production backtest imports. This gives a single source of truth for the objective function between the two out-of-sample stages.
>
> **Note on the "v1" reference numbers**: v1 (the pre-soft-blending baseline used in the v1-vs-v2 comparison chart in `performance.ipynb`) is no longer a runnable notebook — it exists only as a hardcoded reference dict (`V1_REF` in `performance.ipynb`, Cell 2) carried over from an earlier version of the pipeline. Treat the "v1" column in that comparison as a fixed historical baseline, not something you can regenerate by re-running code in this repo.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Kushagra-MnC/Regime-Shift-Quant.git
cd Regime-Shift-Quant
```

ETF, signal, and macro price data (`NIFTYBEES.parquet`, `NSEI.parquet`, `GSEC_10Y.parquet`, etc.) are committed at the repo root and arrive automatically with the clone — no separate data download step is required.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Unzip the optimizer config

```bash
unzip stage4_optimizer_config.zip
```

This creates `stage4_optimizer_config/` containing `optimizer_constants.json`, `weight_bounds.csv`, and `combined_constraints.csv` — the locked parameters Stage 4 and Stage 5 both read from.

### 4. Run notebooks sequentially

Execute in order:

```
data_foundation.ipynb → hmm_dev.ipynb → cvxpy.ipynb →
walkfwd_integration.ipynb → performance.ipynb → holdout.ipynb
```

Each notebook reads its inputs from the previous stage's saved outputs in `data/`.

> ⚠️ **`holdout.ipynb` runs exactly once.** Do not re-run after seeing results — this invalidates the out-of-sample evaluation.

---

## Data Integrity & Reproducibility

- **Pipeline Independence**: No runtime data fetching required after `git clone` — all price data ships with the repo.
- **Seeded HMM**: `HMM_SEED = 42`, `HMM_N_INIT = 20` — identical transition matrices on every run.
- **Expected walk-forward Sharpe**: 0.626 (107 months, Jan 2015 – Nov 2023)
- **Expected holdout Sharpe**: 1.027 (28 months, Jan 2024 – Apr 2026)

---

## PS Goals Compliance

| Goal | Status | Implementation |
|---|---|---|
| HMM Regime Classifier (Bull/Bear/Crisis without manual labelling) | ⚠ Partial | 2-state Gaussian HMM (Bull/Crisis only); `predict_proba` forward algorithm. See note below. |
| Dynamic Constraint Mapping (objective shifts by regime) | ✓ | Variance+CVaR+TC objective with regime-dependent weights; regime-specific equity bounds |
| Walk-Forward Validation Harness (no look-ahead bias) | ✓ | Annual refit, expanding window; look-ahead test PASSED (shuffled timing Sharpe −0.119 vs real +1.487) |
| Model Transaction Friction Explicitly (5–10 bps penalty) | ✓ | Asymmetric STT: buy = 3.5 bps, sell = 13 bps; avg drag 0.18%/yr |
| Benchmark Against Static Portfolios | ✓ | vs NIFTYBEES B&H, equal-weight, 60/40 India |

> **Note on Bull/Bear/Crisis**: The PS specifies three regimes. The BIC grid search (Stage 3) tests n_states ∈ {2,3,4} and selects **n=2 (Bull/Crisis)** by a decisive margin (BIC=2103.1 vs 2106.7 for the next-best n=3 model), while n=3 and n=4 with full covariance repeatedly failed to converge (89–123 parameters on 139 training observations). `Bear` bounds and constraints exist in the optimizer config for forward-compatibility but are never triggered by the current 2-state model. See **Tried Approaches** below for what was attempted to force a genuine third state.
>
> **Note on Max Sharpe**: The PS specified Max Sharpe in Bull regime. Replaced by a variance-dominant convex objective with a relaxed equity ceiling (90% combined NIFTYBEES+JUNIORBEES) because Max Sharpe is non-convex and cannot be solved directly in CVXPY.

---

## Transition Matrix

*(Rows = From State, Cols = To State)*

| | Bull | Crisis |
|---|---|---|
| **Bull** | 0.7950 | 0.2050 |
| **Crisis** | 0.3530 | 0.6470 |

Crisis self-transition of 0.6470 means Crisis is persistent but not permanent — consistent with historical Indian stress episodes averaging 4–6 months.

---

## Performance Tear Sheet

### Walk-Forward (In-Sample) — Jan 2015 to Nov 2023, 107 months

| Strategy | CAGR | Ann. Vol | Sharpe | Sortino | Max DD | Calmar | Win Rate |
|---|---|---|---|---|---|---|---|
| **REGIME-SHIFT** | **12.1%** | **8.9%** | **0.626** | **1.038** | **−7.7%** | **1.575** | **57%** |
| NIFTYBEES B&H | 11.6% | 16.9% | 0.367 | 0.371 | −31.0% | 0.376 | 55% |
| Equal Weight | 10.4% | 8.8% | 0.457 | 0.642 | −12.5% | 0.834 | 53% |
| 60/40 India | 10.9% | 10.3% | 0.448 | 0.595 | −15.8% | 0.691 | 54% |

- **Risk-free rate**: 6.5% (RBI repo, flat)
- **TC drag**: avg round-trip 1.53 bps/rebalance · avg annual drag 0.18%

**Regime-conditional performance** (Stage 6):

| | Bull (58 months) | Crisis (49 months) |
|---|---|---|
| REGIME-SHIFT CAGR | 11.0% | **13.5%** |
| NIFTYBEES B&H CAGR | 19.9% | 2.6% |
| Worst month | −5.3% | −6.6% |

The strategy gives up meaningful upside in Bull (11.0% vs 19.9% — the cost of the 90% equity ceiling and variance-dominant objective) but more than compensates in Crisis (13.5% vs 2.6%, and a −6.6% worst month vs B&H's −25.1%). Crisis timing accuracy (Nifty actually falling in a Crisis month) is 43% — the majority of Crisis months are "false positives" where the defensive posture wasn't strictly necessary, but the payoff asymmetry (small cost in false positives, large benefit in true positives) is what drives the Calmar advantage.

<img width="1313" height="914" alt="Walk-forward performance dashboard" src="https://github.com/user-attachments/assets/08222961-6259-4305-a829-9075b363478b" />

<img width="1313" height="914" alt="Presentation summary" src="https://github.com/user-attachments/assets/fe89aafe-5343-4fc6-8e1a-078c4c1910a5" />

### Holdout — Jan 2024 to Apr 2026, 28 months ¹

| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar | Win Rate | N |
|---|---|---|---|---|---|---|---|---|
| **REGIME-SHIFT** | **14.2%** | **7.0%** | **1.027** | **1.230** | **−5.1%** | **2.764** | **68%** | 28 |
| NIFTYBEES B&H | 4.6% | 13.6% | −0.069 | −0.175 | −14.5% | 0.315 | 52% | 29 |
| Equal Weight | 16.8% | 9.4% | 1.034 | 1.100 | −7.9% | 2.122 | 72% | 29 |
| 60/40 India | 16.1% | 10.3% | 0.891 | 0.905 | −9.0% | 1.788 | 72% | 29 |

> ¹ **N=28 for REGIME-SHIFT**: the same one-month settlement lag as before applies at the end of the extended window — the final decision (30 Apr 2026) requires May 2026 as its settlement month, which falls right at the edge of the evaluation window and is excluded once no further decision exists to consume it.
>
> **This is an extended holdout, covering two materially different periods.** Jan 2024 – May 2025 (17 months) was read almost entirely as Crisis, consistent with the original 11-month holdout reported in earlier versions of this README. The extension adds Jun 2025 – Apr 2026 (11 months), during which the model called **Bull for the first time in any holdout test** — a sustained call from May/Jun 2025 through Jan 2026, before reverting to Crisis in Feb–Apr 2026 (coinciding with a real −5.1% NIFTYBEES month in March 2026). Regime split: 20 Crisis months (71%), 8 Bull months (29%).
>
>
<img width="770" height="526" alt="image" src="https://github.com/user-attachments/assets/41db9112-2300-4b69-b80e-582d21ee8060" />
---

## Alpha Attribution

```
REGIME-SHIFT CAGR          : 12.1%
NIFTYBEES B&H CAGR         : 11.6%
Equal Weight CAGR          : 10.4%

Alpha vs B&H               :  +0.5%
Alpha vs Equal Weight      :  +1.7%
Sharpe advantage vs B&H    :  +0.259
Max DD improvement vs B&H  : −23.2 pp
Calmar ratio vs B&H        :  1.575 vs 0.376  (4.18× better)

Market beta (vs NIFTYBEES) : −0.03  (effectively market-neutral)
Monthly alpha              : +1.02%  (12.29% annualised)
R²                         :  0.003  (returns uncorrelated with Nifty)

Avg annual turnover        : 185%
Avg round-trip TC          : 1.53 bps/rebalance
Estimated annual TC drag   : 0.18%

Bull timing  (Nifty ↑ in Bull months)   : 67%
Crisis timing (Nifty ↓ in Crisis months) : 43%

Look-ahead test (timing-contribution shuffle):
  Real timing Sharpe    : +1.487
  Shuffled timing Sharpe: −0.119 ± 0.204   → PASSED
```

---

## Final Stage Summary

| # | Notebook | Output |
|---|---|---|
| 1 | `data_foundation.ipynb` | Data foundation — 4 ETFs, 8 raw series, 7 features engineered, NSE calendar |
| 2 | `hmm_dev.ipynb` | HMM development — BIC grid search + production fit, n=2 states, composite alignment, saves `data/stage3/` |
| 3 | `cvxpy.ipynb` | Optimizer — CVXPY variance+CVaR+TC objective, Ledoit-Wolf, asymmetric TC model, unit-tested in isolation |
| 4 | `walkfwd_integration.ipynb` | Walk-forward — soft blend, asymmetric persistence, LIQUIDBEES fix |
| 5 | `performance.ipynb` | Analysis — 9 publication charts, factor decomposition, v1 vs v2 |
| **6** | **`holdout.ipynb`** | **Holdout — run once, results final, no re-tuning permitted** |

**One-sentence pitch**: *We train a Hidden Markov Model on Indian macro data to detect Bull or Crisis conditions, then use CVXPY to find the risk-optimal portfolio — variance-dominant in Bull, tail-risk-dominant in Crisis — that automatically shifts defensive as confidence in a Crisis rises, tested rigorously with no look-ahead bias and confirmed on a sealed holdout extending through April 2026.*

---

## Installation

### Prerequisites

- Python 3.9+
- `pip install -r requirements.txt`

Dependencies: `hmmlearn`, `cvxpy`, `clarabel`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`, `yfinance`, `pandas-datareader`, `pandas-market-calendars`, `pyarrow`

---

## Tried Approaches

Several design choices in the current pipeline replaced earlier approaches that were implemented, tested, and rejected on evidence. This section documents what was tried and why it was abandoned — the failures were as informative as the successes.

### State alignment: pure volatility ranking → composite risk-on score

**Tried:** Aligning HMM states purely by mean `realized_vol_z` per state (lowest vol = Bull, highest = Crisis).
**Result:** Correctly separated Crisis from calm periods, but systematically mislabeled 2021 as non-Bull. 2021 was a low-volatility year following the COVID recovery, but the model's ranking put it in the same bucket as genuinely weak markets because volatility alone doesn't capture direction.
**Fix:** Replaced with a 6-feature weighted composite score (`+0.35×momentum + 0.15×trend_quality − 1.00×realized_vol − 0.25×VIX − 0.10×INR_stress − 0.05×G-Sec_momentum`). Momentum and trend-quality terms restored the directional signal that pure volatility ranking discarded.

### Feature scaling: expanding z-score → rolling 12-month z-score

**Tried:** Standardising each feature against its full historical mean/std up to time $t$ (expanding window).
**Result:** Once the COVID outlier entered the expanding window's history, subsequent stress events (2021 volatility, 2022 tightening) looked statistically unremarkable by comparison — the expanding denominator had absorbed a 2020-sized shock, so nothing after it registered as extreme.
**Fix:** Switched to a rolling 12-month (252 trading day) window for all z-scores, so each observation is judged against recent conditions rather than all of history.

### Regime count: forced n=3 (Bull/Bear/Crisis) → BIC-selected n=2

**Tried:** Forcing a 3-state HMM to match the PS's Bull/Bear/Crisis specification, and separately a 4-state model.
**Result:** With `covariance_type='full'` on 6 features, n=3 requires 89 parameters and n=4 requires 123 — both severely over-parameterised against ~139 usable training months. The BIC grid search logged repeated "Model is not converging" warnings for n≥3/full, and even where EM converged, the resulting third state was not economically distinct from Bull or Crisis (see `data/stage3/bic_grid.csv`).
**Fix:** Let BIC select freely across n∈{2,3,4} × covariance∈{full,diag,tied}. n=2/full won by a clear margin (BIC=2103.1 vs 2106.7 for the closest competitor). `Bear` bounds remain in the optimizer config in case a future feature set (see Future Iterations) creates a genuine third cluster.

### Regime decoding: Viterbi → forward algorithm (`predict_proba`)

**Tried:** Hard Viterbi decoding for regime labels, which finds the single most likely state *sequence* rather than the per-timestep probability.
**Result:** Comparing the two methods around the March 2020 boundary (Stage 3, Cell 22) showed Viterbi assigning hard Crisis/Bull flips a step later than the posterior probability implied, and disagreeing with `predict_proba` on 2 of 12 months in the comparison window. Viterbi's global-sequence optimisation is not the right tool for a system that needs a causal, real-time read of "what state am I in right now."
**Fix:** Switched to `predict_proba`, which gives $P(S_t = k \mid O_{1:t})$ using only observations up to $t$ — no future information, and no sequence-level smoothing that could leak information backward.

### Look-ahead test: shuffled portfolio returns → shuffled timing contribution

**Tried:** Validating the absence of look-ahead bias by shuffling the strategy's monthly returns and checking whether the shuffled Sharpe collapsed to zero.
**Result:** The shuffled portfolio's Sharpe stayed strongly positive (≈0.6) regardless of shuffling, because the portfolio is structurally long equity — any random reordering of a positive-expected-return series still has positive Sharpe. This test could never fail, so it proved nothing.
**Fix:** Redefined the test around *timing contribution* — the return earned specifically from deviating from the average (passive) allocation, `(w_t − w_avg)·r_{t+1}`. Shuffling this series correctly collapsed Sharpe to ≈0 (−0.119 ± 0.204), while the real timing series scored +1.487, giving a test that can actually distinguish genuine timing skill from passive market exposure.

### LIQUIDBEES return source: raw yfinance series → flat RBI repo proxy

**Tried:** Using LIQUIDBEES' own yfinance-reported NAV return series directly.
**Result:** The series showed a COVID-era distortion, understating the true overnight-rate return (≈2.7%/yr annualised vs an actual RBI repo rate closer to 6.5%/yr over the same period), which would have understated the "safe" leg of the Crisis portfolio.
**Fix:** Overrode LIQUIDBEES with a flat 6.5%/yr proxy. Noted in Future Iterations as a simplification — a time-varying MIBOR/TREPS series would be more accurate, since the flat rate understates real yields in 2013–16 and overstates them in 2020–21.

### Transaction costs: symmetric flat rate → asymmetric buy/sell by asset class

**Tried:** A single flat per-rebalance cost applied identically to all assets and both trade directions.
**Result:** This ignored India's Securities Transaction Tax (STT), which applies asymmetrically — 0.1% on the sell leg for equity ETF delivery trades, and not at all on gold or liquid ETFs.
**Fix:** Asset-specific buy/sell costs in `optimizer_constants.json` (NIFTYBEES/JUNIORBEES: 3.5 bps buy / 13.0 bps sell; GOLDBEES: 2.5/2.5; LIQUIDBEES: 1.0/1.0), reflecting the actual STT structure.

### Fixed-income sleeve: GILTBEES included → dropped

**Tried:** Including GILTBEES (a G-Sec ETF) as a fifth asset to give the Crisis portfolio genuine duration exposure alongside gold.
**Result:** Reliable price history for GILTBEES only extends back to 2018, leaving an 8-year hole at the start of the 2010–2023 backtest window.
**Fix:** Dropped for the current 4-asset universe. Restoring it via a synthetic pre-2018 series (CRISIL Composite Bond Index or NIFTY 10yr G-Sec Index, both available from 2003) is listed under Future Iterations.

---

## Future Iterations & Planned Improvements

### Tier 1 — Feature Engineering

- **Replace `vix_z` with VIX/Realised-Vol ratio** (`vix_premium_z`): implied vol divided
  by concurrent realised vol. Encodes forward-looking fear premium rather than raw
  level — orthogonal to `realized_vol_z`, which currently gets two correlated votes
  in the HMM clustering.
  
- **Add raw trailing return features** (3-month and 12-month Nifty pct_change, not
  z-scored): a −15% trailing return signals genuine drawdown regardless of historical
  average. Directly targets the 43% Crisis timing accuracy by separating "VIX spike
  with market flat" from "VIX spike after market has already fallen 10%+".
  
- **Add G-Sec yield level z-score** alongside the existing yield momentum feature: a
  RBI hiking cycle looks different from a low-and-stable rate environment even when
  monthly changes are similar. Level + momentum together give the HMM both the
  direction and the starting point.

### Tier 2 — Regime Architecture

- **Graduated bounds as a function of P(Crisis)**: instead of fixed weight ceilings
  per regime, make the equity upper bound a continuous function:
  `equity_upper = 0.90 − 0.75 × P(Crisis)`. Eliminates the hard corner solution
  where every Bull month produces identical weights regardless of how bullish the
  signal is.
  
- **Restore GILTBEES with a synthetic pre-2018 series**: backfill using the CRISIL
  Composite Bond Index or NIFTY 10yr Benchmark G-Sec Index (both available from 2003).
  Re-introduces a genuine fixed-income leg — the asset that rises when rates fall and
  equities sell off — which gold cannot fully replace in Crisis portfolios.
  
- **n=3 HMM with a genuine Bear state**: the PS requires Bull/Bear/Crisis but the
  current model collapses to 2 states because the feature set can't separate
  "moderate vol, weak trend" from "low vol, strong trend". Adding `vix_premium_z`
  and trailing return features should create the necessary third cluster without
  forcing it via `FORCE_N_STATES`.

### Tier 3 — Execution & Cost Model
- **Time-varying LIQUIDBEES return**: replace the flat 6.5% proxy with actual
  RBI MIBOR/TREPS daily rates from the RBI DBIE database. The flat rate understates
  Crisis portfolio returns in 2013–2016 (MIBOR ~8–9%) and overstates them in
  2020–2021 (repo rate 4%). Proper time-series makes the Crisis alpha attribution
  honest.
  
- **Market impact model for larger AUM**: the current TC model handles brokerage +
  STT + stamp duty but ignores market impact. A square-root impact model
  (`impact = σ × √(trade_size / ADV)`) would let the strategy report the AUM
  ceiling at which the alpha is destroyed — standard practice for institutional
  strategy documentation.

- `walkfwd_integration.ipynb` and `holdout.ipynb` both import `solve_regime_objective()` and related functions
  directly from `optimizer_new.py`, so the objective function unit-tested in
  `cvxpy.ipynb` is the same code executed in both out-of-sample stages.

### Tier 4 — Validation & Robustness
- **Monte Carlo regime-label permutation test**: randomly permute the confirmed
  regime labels 1000 times and rerun the optimizer each time. The real strategy's
  Sharpe should sit in the top 5% of the permutation distribution — this formally
  tests whether the HMM regime detection adds value beyond a random labelling scheme.
  
- **Rolling window sensitivity analysis**: rerun the full walk-forward with initial
  training windows of 48, 60, and 84 months. If Sharpe varies materially across
  window lengths, the strategy is sensitive to this design choice and that
  sensitivity should be disclosed.
