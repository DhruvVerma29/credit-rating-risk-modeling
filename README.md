# Credit Portfolio Risk Model: Markov Chains & Monte Carlo VaR

An object-oriented Python model that measures corporate credit risk using **continuous-time Markov chains** and **Monte Carlo simulation**. It converts a one-year rating transition matrix into a continuous-time generator, projects default probabilities over any horizon, and computes portfolio Expected Credit Loss (ECL), Value at Risk (VaR) and Unexpected Loss for a loan book of any size, including 100,000 loans.

---

## Features

- **Rating transition modelling:** cleans a one-year transition matrix and removes the unrated (NR) state, re-normalising each row to sum to 1
- **Generator matrix:** derives the continuous-time generator `Q` from the matrix logarithm, with off-diagonals set non-negative and diagonals set so every row sums to zero
- **Any-horizon transition probabilities:** `P(t) = exp(Q·t)` for fractional or multi-year horizons
- **Expected Credit Loss:** `ECL = PD × LGD × EAD` for every loan
- **Monte Carlo VaR:** simulates final rating states for the whole portfolio, then reports VaR at a chosen confidence level and Unexpected Loss
- **Synthetic portfolio generator:** builds a realistic corporate loan book with log-normal exposures and Beta-distributed LGDs

---

## Project structure

```
.
├── Credit_Risk.py            # CreditPortfolio class and main script
├── portfolio_generator.py    # Synthetic loan book generator
├── Transition_rates.csv      # One-year rating transition matrix (%)
└── README.md
```

---

## Input data

### Transition matrix (`Transition_rates.csv`)
A one-year rating migration matrix in percent. Rows are the starting rating and columns are the ending rating.

- **Rating states:** AAA, AA, A, BBB, BB, B, CCC/C, D (default)
- **NR column:** share of entities that became unrated; removed and re-normalised by the model
- **D row:** default is an absorbing state

### Loan book
A DataFrame with one row per loan:

| Column | Description |
|---|---|
| `Loan_ID` | Loan identifier |
| `Rating` | Current rating |
| `EAD` | Exposure at default |
| `LGD` | Loss given default (fraction) |

---

## Methodology

### 1. Matrix preparation
The NR column is dropped and each row is divided by its sum, giving a proper one-year transition probability matrix over the rated states and default.

### 2. Generator matrix
The generator is computed with the matrix logarithm:

```
Q = log(P)
```

Off-diagonal entries below zero are set to 0, and each diagonal entry is set to the negative sum of its row's off-diagonals, so that `Q` is a valid generator.

### 3. Transition probabilities at any horizon

```
P(t) = exp(Q · t)
```

The `D` column of `P(t)` gives the cumulative probability of default (PD) for each starting rating over `t` years.

### 4. Expected Credit Loss

```
ECL = PD(t) × LGD × EAD
```

Each loan's PD is looked up from its current rating, and ECL is summed across the portfolio.

### 5. Monte Carlo Value at Risk
1. Build the cumulative distribution of final states for each starting rating from `P(t)`.
2. For each of `n_simulations` paths, draw a uniform random number per loan and map it to a final rating state through the CDF.
3. Loans that land in the default state contribute `EAD × LGD` to the path's portfolio loss.
4. VaR is the chosen percentile of the simulated loss distribution.
5. **Unexpected Loss = VaR − Expected Credit Loss.**

### 6. Synthetic portfolio (`portfolio_generator.py`)

| Attribute | Distribution |
|---|---|
| Rating | AAA 3%, AA 7%, A 25%, BBB 40%, BB 15%, B 8%, CCC/C 2% |
| EAD | Log-normal (mean 14.5, sigma 1.0), clipped to $250K–$50M |
| LGD | Beta(4, 5) |

A fixed seed makes the portfolio reproducible.

---

## Getting started

### 1. Install dependencies

```bash
pip install numpy pandas scipy
```

### 2. Run

```bash
python Credit_Risk.py
```

This generates a 100,000-loan portfolio, runs 10,000 Monte Carlo paths over a 5-year horizon, and prints the results.

---

## Usage

```python
import pandas as pd
from Credit_Risk import CreditPortfolio
from portfolio_generator import generate_corporate_portfolio

raw_matrix = pd.read_csv("Transition_rates.csv", index_col=0)
loan_book = generate_corporate_portfolio(num_loans=100000)

model = CreditPortfolio(raw_matrix, loan_book)

# Transition probabilities for any horizon (in years)
P_3y = model.get_transition_matrix(years=3)

# Loan-level PD and ECL
ecl_table = model.calculate_expected_loss(years=5)

# Portfolio risk metrics
metrics = model.calculate_value_at_risk(years=5, n_simulations=10000, confidence_level=99.0)
print(metrics)
```

### Output

`calculate_value_at_risk` returns a dictionary:

| Key | Description |
|---|---|
| `Total_EAD` | Total exposure of the portfolio |
| `Expected_Credit_Loss` | Sum of loan-level ECL |
| `VaR_<confidence>` | Loss at the chosen percentile of the simulated distribution |
| `Unexpected_Loss` | VaR minus Expected Credit Loss |

---

## Tech stack

- **Python**
- **NumPy / pandas** – data handling and simulation
- **SciPy** – matrix logarithm and matrix exponential

---

## Key concepts

- **Credit rating migration** and absorbing default states
- **Continuous-time Markov chains** and generator matrices
- **Matrix exponential** for transition probabilities over arbitrary horizons
- **PD, LGD, EAD** and Expected Credit Loss
- **Monte Carlo simulation** of portfolio loss distributions
- **Value at Risk** and Unexpected Loss as measures of credit capital
