# Options Pricing

American and European option pricing via Least-Squares Monte Carlo (LSM) and Black-Scholes. Interactive Streamlit demo.

**Live demo:** [lsm-pricing-options.streamlit.app](https://lsm-pricing-options.streamlit.app/) &nbsp;|&nbsp; **Paper:** [Longstaff, Schwartz 2001](https://doi.org/10.1093/rfs/14.1.113)

---

## What this does

Four tabs in one app:

| Tab | Method | Option type | Speed |
|---|---|---|---|
| LSM | Least-Squares Monte Carlo | American (early exercise) | Seconds |
| Black-Scholes | Closed-form analytical | European (expiry only) | Instant |
| Comparison | Both side by side | Shows early exercise premium | Instant after LSM run |
| Greeks | Black-Scholes sensitivities | Delta, Gamma, Vega, Theta, Rho | Instant |

---

## LSM Algorithm

American options can be exercised at any point before expiry, which makes them harder to price than European options. The Longstaff-Schwartz (2001) Least-Squares Monte Carlo method solves this by:

1. Simulating M price paths using Geometric Brownian Motion
2. Working **backwards** from expiry — at each timestep, using OLS regression to estimate the continuation value (expected value of waiting)
3. Exercising early on paths where the immediate payoff exceeds the estimated continuation value
4. Averaging the discounted cash flows across all paths

The regression uses orthogonal polynomial basis functions (Laguerre, Chebyshev, or Legendre — selectable in the demo).

**Final price:**
```
option_price = mean( V[:, t=1] ) * exp(-r * dt)
```

---

## Black-Scholes

Closed-form solution assuming log-normal asset prices, constant volatility, and no early exercise. Valid for European options.

```
d1 = ( ln(S0/K) + (r + σ²/2) * T ) / ( σ * √T )
d2 = d1 - σ * √T

Put:  P = K * e^(-rT) * N(-d2) - S0 * N(-d1)
Call: C = S0 * N(d1) - K * e^(-rT) * N(d2)
```

---

## Quick start

```bash
git clone https://github.com/aleksejcupic/options-pricing
cd options-pricing
pip install -r requirements.txt
streamlit run app.py
```

---

## Parameters

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| Initial price | S0 | 1.00 | Starting asset price |
| Strike price | K | 1.00 | Agreed exercise price |
| Time to expiry | T | 1 yr | Duration of the contract |
| Risk free rate | r | 6% | Baseline return rate |
| Volatility | σ | 30% | Annualised std dev of returns |
| Time steps | N | 50 | Exercise points (LSM only) |
| Paths | M | 10,000 | Simulated GBM paths (LSM only) |
| Basis | | Laguerre | Polynomial family for OLS regression (LSM only) |

---

## References

Longstaff, F. A. and Schwartz, E. S. (2001). Valuing American options by simulation: a simple least squares approach. *The Review of Financial Studies*, 14(1), 113 to 147.

Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637 to 654.

---

## Author

Aleksej Cupic  
[aleksejcupic.com](https://aleksejcupic.com)

Original LSM implementation: Boston College CSCI2244 (May 2022), Aleksej Cupic, Gio Canales, Komi Alasse.  
Completed and expanded (2026): Aleksej Cupic.
