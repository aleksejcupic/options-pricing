"""
Options Pricing — LSM + Black-Scholes
Run locally:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="Options Pricing", layout="wide")
st.title("Options Pricing")
st.caption(
    "Least-Squares Monte Carlo (Longstaff-Schwartz 2001) and Black-Scholes (1973). "
    "American options via LSM, European options via Black-Scholes."
)

# ── Shared session state ──────────────────────────────────────────────────────
if "lsm_result" not in st.session_state:
    st.session_state.lsm_result = None


# ── Sidebar parameters ────────────────────────────────────────────────────────
def param(label, min_val, max_val, default, step, fmt=None, key=None, help=None):
    if key not in st.session_state:
        st.session_state[key] = default

    def _from_slider():
        st.session_state[key] = st.session_state[f"{key}_s"]

    def _from_input():
        st.session_state[key] = st.session_state[f"{key}_n"]

    col_s, col_n = st.sidebar.columns([3, 2])
    col_s.slider(label, min_val, max_val, value=st.session_state[key], step=step,
                 key=f"{key}_s", on_change=_from_slider, label_visibility="collapsed")
    col_n.number_input(label, min_val, max_val, value=st.session_state[key], step=step,
                       format=fmt, key=f"{key}_n", on_change=_from_input,
                       label_visibility="collapsed")
    st.sidebar.caption(f"**{label.strip()}**")
    if help:
        st.sidebar.caption(f"*{help}*")
    return float(st.session_state[key])


st.sidebar.header("Parameters")

option_type = st.sidebar.radio(
    "Option type", ["Put", "Call"], horizontal=True,
    help="Put options profit when the asset falls below the strike. "
         "Call options profit when it rises above.",
)
basis = st.sidebar.radio(
    "Basis functions", ["Laguerre", "Chebyshev", "Legendre"], horizontal=True,
    help="Polynomial family used in the LSM regression. Laguerre is the Longstaff-Schwartz default. "
         "Chebyshev and Legendre are orthogonal alternatives that produce similar results.",
)

st.sidebar.markdown("")
S0    = param("S0  Initial stock price ($)", 0.50, 5.00, 1.00, 0.01, "%.2f", "S0",
              help="Starting price of the underlying asset.")
K     = param("K   Strike price ($)", 0.50, 5.00, 1.00, 0.01, "%.2f", "K",
              help="Price at which the option can be exercised. "
                   "Puts profit below K, calls profit above K.")
sigma = param("Volatility (σ)", 0.05, 1.00, 0.30, 0.01, "%.2f", "sigma",
              help="Annualised standard deviation of the asset's returns.")
r     = param("Risk free rate (r)", 0.00, 0.20, 0.06, 0.01, "%.3f", "r",
              help="Baseline return rate used for GBM drift and discounting. "
                   "Typically the 3-month treasury yield.")
T     = param("Time to expiry (years)", 0.25, 5.00, 1.00, 0.25, "%.2f", "T",
              help="Duration of the contract in years.")

st.sidebar.markdown("")
N = int(param("Time steps (N)", 10, 200, 50, 5, "%d", "N",
              help="Discrete exercise points between now and expiry. "
                   "More steps better approximates continuous exercise."))
M = int(param("Simulated paths (M)", 500, 50000, 10000, 500, "%d", "M",
              help="Number of GBM price paths. "
                   "Standard error scales as 1/√M."))
seed = st.sidebar.number_input("Random seed", 0, 9999, 42, 1,
                                help="Fix for reproducible results.")


# ── Basis function builder ────────────────────────────────────────────────────
def build_poly(x_vals, basis_name):
    b = basis_name.lower()
    if b == "laguerre":
        return pd.DataFrame({
            0: np.exp(-x_vals / 2),
            1: np.exp(-x_vals / 2) * (1 - x_vals),
            2: np.exp(-x_vals / 2) * (1 - 2 * x_vals + x_vals ** 2 / 2),
        })
    if b == "chebyshev":
        return pd.DataFrame({0: np.ones_like(x_vals), 1: x_vals, 2: 2 * x_vals ** 2 - 1})
    return pd.DataFrame({0: np.ones_like(x_vals), 1: x_vals, 2: (3 * x_vals ** 2 - 1) / 2})


# ── LSM engine ────────────────────────────────────────────────────────────────
def run_lsm(s0, k, t_exp, rate, vol, n_steps, n_paths, rng_seed, opt_type, basis_name):
    rng = np.random.default_rng(int(rng_seed))
    dt  = t_exp / n_steps
    df  = np.exp(-rate * dt)
    is_put = opt_type == "Put"

    def _payoff(p):
        return np.maximum(k - p, 0) if is_put else np.maximum(p - k, 0)

    def _itm(p):
        return p < k if is_put else p > k

    s = np.zeros((n_paths, n_steps + 1))
    s[:, 0] = s0
    for step in range(1, n_steps + 1):
        dz = rng.standard_normal(n_paths) * np.sqrt(dt)
        s[:, step] = s[:, step - 1] * np.exp((rate - 0.5 * vol ** 2) * dt + vol * dz)

    v = np.zeros((n_paths, n_steps + 1))
    v[:, n_steps] = _payoff(s[:, n_steps])

    mid = n_steps // 2
    reg = {}
    for step in range(n_steps - 1, 0, -1):
        itm = _itm(s[:, step])
        if itm.sum() < 3:
            v[:, step] = v[:, step + 1] * df
            continue
        x = s[itm, step]
        y = v[itm, step + 1] * df
        poly         = build_poly(x, basis_name)
        coef         = sm.OLS(y, poly).fit().params
        continuation = poly.to_numpy() @ coef
        exercise     = _payoff(x)
        v[itm, step]  = np.where(exercise > continuation, exercise, v[itm, step + 1] * df)
        v[~itm, step] = v[~itm, step + 1] * df
        if step == mid:
            reg = {"x": x, "y": y, "continuation": continuation,
                   "exercise": exercise, "coef": coef, "t": step}

    price = float(np.mean(v[:, 1]) * df)
    return s, price, reg


# ── Black-Scholes engine ──────────────────────────────────────────────────────
def bs_price(s0, k, t_exp, rate, vol, opt_type):
    if t_exp <= 0:
        return max(k - s0, 0) if opt_type == "Put" else max(s0 - k, 0)
    d1 = (np.log(s0 / k) + (rate + 0.5 * vol ** 2) * t_exp) / (vol * np.sqrt(t_exp))
    d2 = d1 - vol * np.sqrt(t_exp)
    if opt_type == "Put":
        return k * np.exp(-rate * t_exp) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    return s0 * norm.cdf(d1) - k * np.exp(-rate * t_exp) * norm.cdf(d2)


def bs_greeks(s0, k, t_exp, rate, vol, opt_type):
    """Returns (delta, gamma, vega, theta, rho)."""
    if t_exp <= 0:
        return (0.0,) * 5
    d1   = (np.log(s0 / k) + (rate + 0.5 * vol ** 2) * t_exp) / (vol * np.sqrt(t_exp))
    d2   = d1 - vol * np.sqrt(t_exp)
    pdf1 = norm.pdf(d1)
    gamma = pdf1 / (s0 * vol * np.sqrt(t_exp))
    vega  = s0 * pdf1 * np.sqrt(t_exp) / 100          # per 1% move in vol
    if opt_type == "Put":
        delta = norm.cdf(d1) - 1
        theta = ((-s0 * pdf1 * vol / (2 * np.sqrt(t_exp)))
                 + rate * k * np.exp(-rate * t_exp) * norm.cdf(-d2)) / 365
        rho   = -k * t_exp * np.exp(-rate * t_exp) * norm.cdf(-d2) / 100
    else:
        delta = norm.cdf(d1)
        theta = ((-s0 * pdf1 * vol / (2 * np.sqrt(t_exp)))
                 - rate * k * np.exp(-rate * t_exp) * norm.cdf(d2)) / 365
        rho   = k * t_exp * np.exp(-rate * t_exp) * norm.cdf(d2) / 100
    return delta, gamma, vega, theta, rho


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_lsm, tab_bs, tab_cmp, tab_greeks = st.tabs([
    "LSM  (American)", "Black-Scholes  (European)", "Comparison", "Greeks",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — LSM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_lsm:
    st.subheader("Least-Squares Monte Carlo — American Option")
    st.markdown(
        "Simulates thousands of GBM price paths then works **backwards** from expiry, "
        "using OLS regression to decide at each step whether early exercise beats waiting."
    )
    if st.button("Price Option (LSM)", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            s_paths, price_lsm, reg = run_lsm(
                S0, K, T, r, sigma, N, M, seed, option_type, basis
            )
        st.session_state.lsm_result = {
            "price": price_lsm, "paths": s_paths, "reg": reg,
            "params": (S0, K, T, r, sigma, N, M, option_type, basis),
        }

    if st.session_state.lsm_result:
        res = st.session_state.lsm_result
        col_p, col_info = st.columns([1, 2])
        with col_p:
            st.metric(f"American {option_type} Price (LSM)", f"${res['price']:.4f}")
        with col_info:
            p = res["params"]
            st.markdown(
                f"**S0** {p[0]:.2f}  **K** {p[1]:.2f}  **σ** {p[4]:.0%}  "
                f"**r** {p[3]:.1%}  **T** {p[2]:.2f}yr  "
                f"**N** {p[5]}  **M** {p[6]:,}  **basis** {p[8]}"
            )

        st.markdown("---")
        left, right = st.columns(2)

        with left:
            st.subheader("Simulated Price Paths")
            fig1, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(res["paths"][:50, :].T, alpha=0.6, lw=0.9)
            ax1.axhline(K, color="crimson", ls="--", lw=1.8, label=f"Strike ${K:.2f}")
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Asset Price ($)")
            ax1.set_title(f"GBM Paths (50 of {M:,} shown)")
            ax1.legend(loc="upper right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with right:
            reg = res["reg"]
            if reg:
                st.subheader(f"LSM Regression at t = {reg['t']}")
                x_plt = reg["x"]; y_plt = reg["y"]
                cont = reg["continuation"]; ex = reg["exercise"]; coef = reg["coef"]
                x_line = np.linspace(x_plt.min(), x_plt.max(), 300)
                y_line = build_poly(x_line, basis).to_numpy() @ coef
                ex_line = (np.maximum(K - x_line, 0) if option_type == "Put"
                           else np.maximum(x_line - K, 0))
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                ax2.scatter(x_plt, y_plt, s=6, alpha=0.3, color="orange",
                            label="Hold value (simulated)")
                ax2.scatter(x_plt, cont, s=10, alpha=0.6, color="royalblue", marker="x",
                            label="LSM continuation estimate")
                ax2.plot(x_line, y_line, color="royalblue", lw=2,
                         label=f"LSM curve ({basis})")
                ax2.plot(x_line, ex_line, color="crimson", lw=2, ls="--",
                         label="Exercise value")
                ax2.set_xlabel(f"Asset Price at t = {reg['t']}")
                ax2.set_ylabel("Option Value")
                ax2.legend(fontsize=8, loc="upper right")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
    else:
        st.info("Set parameters in the sidebar and click **Price Option (LSM)**.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — BLACK-SCHOLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_bs:
    st.subheader("Black-Scholes — European Option")
    st.markdown(
        "Analytical closed-form formula (Black, Scholes 1973). Assumes log-normal prices, "
        "constant volatility, and **no early exercise** — valid for European options only."
    )

    price_bs = bs_price(S0, K, T, r, sigma, option_type)
    st.metric(f"European {option_type} Price (BS)", f"${price_bs:.4f}")

    st.markdown("---")
    st.markdown("**Formula**")
    st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}")
    if option_type == "Put":
        st.latex(r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)")
    else:
        st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")

    st.markdown("---")
    # Price vs. spot chart
    spots = np.linspace(max(0.1, S0 * 0.5), S0 * 2.0, 200)
    prices_curve = [bs_price(s, K, T, r, sigma, option_type) for s in spots]
    intrinsic    = [(max(K - s, 0) if option_type == "Put" else max(s - K, 0)) for s in spots]
    fig_bs, ax_bs = plt.subplots(figsize=(8, 4))
    ax_bs.plot(spots, prices_curve, color="royalblue", lw=2, label="BS price")
    ax_bs.plot(spots, intrinsic,    color="crimson",   lw=1.5, ls="--", label="Intrinsic value")
    ax_bs.axvline(S0, color="grey", ls=":", lw=1, label=f"Current S0 = {S0:.2f}")
    ax_bs.axvline(K,  color="black", ls="--", lw=1, label=f"Strike K = {K:.2f}")
    ax_bs.set_xlabel("Asset Price")
    ax_bs.set_ylabel("Option Price")
    ax_bs.set_title(f"Black-Scholes {option_type} Price vs. Asset Price")
    ax_bs.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_bs)
    plt.close(fig_bs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_cmp:
    st.subheader("LSM vs Black-Scholes Comparison")
    st.markdown(
        "Black-Scholes prices **European** options (no early exercise). "
        "LSM prices **American** options (early exercise allowed). "
        "The difference is the **early exercise premium** — the value of being able to exit early."
    )

    price_bs_cmp = bs_price(S0, K, T, r, sigma, option_type)

    if st.session_state.lsm_result:
        price_lsm_cmp = st.session_state.lsm_result["price"]
        premium = price_lsm_cmp - price_bs_cmp

        col1, col2, col3 = st.columns(3)
        col1.metric(f"LSM American {option_type}", f"${price_lsm_cmp:.4f}")
        col2.metric(f"BS European {option_type}",  f"${price_bs_cmp:.4f}")
        col3.metric("Early Exercise Premium", f"${premium:.4f}",
                    help="How much more the American option is worth due to early exercise rights. "
                         "Always >= 0 since American options can do everything European ones can.")

        if premium < 0:
            st.warning(
                "The LSM estimate is below the BS price. "
                "This can happen with small M — increase paths for a more accurate result."
            )
        else:
            pct = premium / price_bs_cmp * 100 if price_bs_cmp > 0 else 0
            st.info(
                f"Early exercise adds **{pct:.1f}%** to the option's value "
                f"(${premium:.4f} on top of the ${price_bs_cmp:.4f} European price)."
            )

        st.markdown("---")
        # Price vs. spot comparison chart
        spots_cmp = np.linspace(max(0.1, S0 * 0.5), S0 * 1.8, 200)
        bs_curve  = [bs_price(s, K, T, r, sigma, option_type) for s in spots_cmp]
        fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
        ax_cmp.plot(spots_cmp, bs_curve, color="royalblue", lw=2,
                    label="BS European (analytical)")
        ax_cmp.axhline(price_lsm_cmp, color="orange", lw=2, ls="--",
                       label=f"LSM American (at S0={S0:.2f}) = ${price_lsm_cmp:.4f}")
        ax_cmp.axvline(S0, color="grey", ls=":", lw=1)
        ax_cmp.axvline(K,  color="black", ls="--", lw=1, label=f"Strike K = {K:.2f}")
        ax_cmp.set_xlabel("Asset Price")
        ax_cmp.set_ylabel("Option Price")
        ax_cmp.set_title(f"American vs European {option_type} Price")
        ax_cmp.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

    else:
        st.metric(f"BS European {option_type}", f"${price_bs_cmp:.4f}")
        st.info("Run LSM in the **LSM (American)** tab to see the full comparison.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — GREEKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_greeks:
    st.subheader("Greeks — Option Sensitivities (Black-Scholes)")
    st.markdown(
        "The Greeks measure how the option price changes as each input moves. "
        "Computed analytically from the Black-Scholes formula."
    )

    delta, gamma, vega, theta, rho = bs_greeks(S0, K, T, r, sigma, option_type)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Delta δ",  f"{delta:.4f}",
              help="Change in option price per $1 change in the asset price.")
    c2.metric("Gamma γ",  f"{gamma:.4f}",
              help="Rate of change of delta per $1 change in the asset price.")
    c3.metric("Vega ν",   f"{vega:.4f}",
              help="Change in option price per 1% change in volatility.")
    c4.metric("Theta θ",  f"{theta:.4f}",
              help="Daily time decay — how much the option loses per day.")
    c5.metric("Rho ρ",    f"{rho:.4f}",
              help="Change in option price per 1% change in the risk-free rate.")

    st.markdown("---")

    spots_g = np.linspace(max(0.1, S0 * 0.4), S0 * 1.8, 300)
    greek_vals = [bs_greeks(s, K, T, r, sigma, option_type) for s in spots_g]
    deltas, gammas, vegas, thetas, rhos = zip(*greek_vals)

    greek_data = {
        "Delta δ":  (deltas,  "Sensitivity to asset price"),
        "Gamma γ":  (gammas,  "Rate of change of delta"),
        "Vega ν":   (vegas,   "Sensitivity to volatility (per 1% vol move)"),
        "Theta θ":  (thetas,  "Daily time decay"),
        "Rho ρ":    (rhos,    "Sensitivity to interest rate (per 1% rate move)"),
    }

    cols = st.columns(2)
    for idx, (name, (values, desc)) in enumerate(greek_data.items()):
        fig_g, ax_g = plt.subplots(figsize=(5.5, 3))
        ax_g.plot(spots_g, values, color="royalblue", lw=2)
        ax_g.axhline(0, color="lightgrey", lw=0.8)
        ax_g.axvline(S0, color="grey", ls=":", lw=1, label=f"S0 = {S0:.2f}")
        ax_g.axvline(K,  color="black", ls="--", lw=1, label=f"K = {K:.2f}")
        ax_g.set_xlabel("Asset Price")
        ax_g.set_ylabel(name)
        ax_g.set_title(f"{name} — {desc}")
        ax_g.legend(fontsize=8)
        plt.tight_layout()
        cols[idx % 2].pyplot(fig_g)
        plt.close(fig_g)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "---\n"
    "Aleksej Cupic  |  "
    "[GitHub](https://github.com/aleksejcupic/options-pricing)  |  "
    "[aleksejcupic.com](https://aleksejcupic.com)"
)
