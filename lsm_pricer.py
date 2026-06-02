"""
American Option Pricing via Least-Squares Monte Carlo
Longstaff, Schwartz, Valuing American Options by Simulation (2001)

Supports both put and call options.

Usage:
    python lsm_pricer.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ── Parameters ────────────────────────────────────────────────────────────────
OPTION = "put"       # "put" or "call"
BASIS  = "laguerre"  # "laguerre", "chebyshev", or "legendre"
S0     = 1.00   # initial stock price
K      = 1.00   # strike price
T      = 1      # time to expiration (years)
r      = 0.06   # risk free rate
sigma  = 0.30   # volatility
N      = 50     # number of time steps
M      = 10000  # number of simulated paths
SEED   = 42

assert OPTION in ("put", "call"),                     "OPTION must be 'put' or 'call'"
assert BASIS  in ("laguerre", "chebyshev", "legendre"), "BASIS must be 'laguerre', 'chebyshev', or 'legendre'"
rng = np.random.default_rng(SEED)

dt = T / N
df = np.exp(-r * dt)  # per-step discount factor


def build_poly(X):
    """Three-term basis function matrix for the OLS regression."""
    if BASIS == "laguerre":
        return pd.DataFrame({
            0: np.exp(-X / 2),
            1: np.exp(-X / 2) * (1 - X),
            2: np.exp(-X / 2) * (1 - 2 * X + X ** 2 / 2),
        })
    if BASIS == "chebyshev":
        return pd.DataFrame({
            0: np.ones_like(X),
            1: X,
            2: 2 * X ** 2 - 1,
        })
    # legendre
    return pd.DataFrame({
        0: np.ones_like(X),
        1: X,
        2: (3 * X ** 2 - 1) / 2,
    })


def payoff(prices):
    """Intrinsic value of the option given asset prices."""
    if OPTION == "put":
        return np.maximum(K - prices, 0)
    return np.maximum(prices - K, 0)


def itm_mask(prices):
    """True for paths that are currently in the money."""
    if OPTION == "put":
        return prices < K
    return prices > K


# ── 1. Simulate price paths (geometric Brownian motion) ──────────────────────
S = np.zeros((M, N + 1))
S[:, 0] = S0
for t in range(1, N + 1):
    dZ = rng.standard_normal(size=M) * np.sqrt(dt)
    S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * dZ)

# ── Plot 1: simulated price paths ─────────────────────────────────────────────
label = OPTION.capitalize()
fig, ax = plt.subplots(figsize=(10, 5.5))

# Show 50 individual paths so each line is visible
ax.plot(S[:50, :].T, alpha=0.6, lw=0.9)
ax.axhline(K, color="crimson", ls="--", lw=1.8, label=f"Strike  ${K:.2f}")

ax.set_xlabel("Time Step", fontsize=10)
ax.set_ylabel("Asset Price ($)", fontsize=10)
ax.set_title(
    f"Simulated GBM Price Paths  ({label})\n"
    f"50 sample paths shown     N = {N} steps     σ = {sigma:.0%}     r = {r:.0%}",
    fontsize=10, pad=10,
)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="lightgrey")

plt.tight_layout()
plt.savefig("images/gbm_paths.png", dpi=150)
plt.close()
print("Saved gbm_paths.png")

# ── 2. Initialise value matrix ────────────────────────────────────────────────
# V[m, t] = option value for path m at time t under optimal exercise
V = np.zeros((M, N + 1))
V[:, N] = payoff(S[:, N])

# ── 3. Backward induction (Longstaff-Schwartz LSM) ───────────────────────────
mid = N // 2
regression_data = {}

for t in range(N - 1, 0, -1):
    itm = itm_mask(S[:, t])
    if itm.sum() < 3:
        V[:, t] = V[:, t + 1] * df
        continue

    X = S[itm, t]              # ITM stock prices at time t
    Y = V[itm, t + 1] * df    # discounted continuation values

    poly = build_poly(X)

    coef         = sm.OLS(Y, poly).fit().params
    continuation = poly.to_numpy() @ coef
    exercise     = payoff(X)

    V[itm, t]  = np.where(exercise > continuation, exercise, V[itm, t + 1] * df)
    V[~itm, t] = V[~itm, t + 1] * df

    if t == mid:
        regression_data = {"X": X, "Y": Y, "continuation": continuation,
                           "exercise": exercise, "coef": coef}

# ── 4. Option price ───────────────────────────────────────────────────────────
option_price = np.mean(V[:, 1]) * df
print(f"\nAmerican {label} Option Price (LSM): ${option_price:.4f}")
print(f"Parameters: S0={S0}, K={K}, T={T}yr, r={r:.0%}, sigma={sigma:.0%}, N={N}, M={M:,}, basis={BASIS}")

# ── Plot 2: LSM regression at midpoint timestep ───────────────────────────────
if regression_data:
    X_plt = regression_data["X"]
    Y_plt = regression_data["Y"]
    cont  = regression_data["continuation"]
    ex    = regression_data["exercise"]
    coef  = regression_data["coef"]

    x_line = np.linspace(X_plt.min(), X_plt.max(), 300)
    y_line = build_poly(x_line).to_numpy() @ coef     # LSM continuation curve
    ex_line = payoff(x_line)                           # exercise value as a line

    # Find where exercise value and continuation curve cross (the critical price S*)
    diff = ex_line - y_line
    crossings = np.nonzero(np.diff(np.sign(diff)))[0]
    s_star = None
    if len(crossings) > 0:
        i = crossings[0]
        # linear interpolation for a precise crossing point
        s_star = x_line[i] - diff[i] * (x_line[i + 1] - x_line[i]) / (diff[i + 1] - diff[i])

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 6.5), layout="constrained")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[3.0, 1.2], wspace=0.03)
    ax      = fig.add_subplot(gs[0])   # main chart
    ax_side = fig.add_subplot(gs[1])   # info panel
    ax_side.axis("off")

    x_min, x_max = X_plt.min(), X_plt.max()

    # Background zone shading
    if s_star is not None:
        if OPTION == "put":
            ax.axvspan(x_min,  s_star, color="tomato",         alpha=0.08, zorder=0)
            ax.axvspan(s_star, x_max,  color="mediumseagreen", alpha=0.08, zorder=0)
            ax.text(x_min + (s_star - x_min) * 0.5, 0.02, "EXERCISE NOW",
                    ha="center", va="bottom", fontsize=8.5,
                    color="firebrick", fontweight="bold",
                    transform=ax.get_xaxis_transform())
            ax.text(s_star + (x_max - s_star) * 0.5, 0.02, "HOLD",
                    ha="center", va="bottom", fontsize=8.5,
                    color="seagreen", fontweight="bold",
                    transform=ax.get_xaxis_transform())
        else:
            ax.axvspan(x_min,  s_star, color="mediumseagreen", alpha=0.08, zorder=0)
            ax.axvspan(s_star, x_max,  color="tomato",         alpha=0.08, zorder=0)
            ax.text(x_min + (s_star - x_min) * 0.5, 0.02, "HOLD",
                    ha="center", va="bottom", fontsize=8.5,
                    color="seagreen", fontweight="bold",
                    transform=ax.get_xaxis_transform())
            ax.text(s_star + (x_max - s_star) * 0.5, 0.02, "EXERCISE NOW",
                    ha="center", va="bottom", fontsize=8.5,
                    color="firebrick", fontweight="bold",
                    transform=ax.get_xaxis_transform())

    # Scatter + lines  (two-line labels: title \n description, no dashes)
    ax.scatter(X_plt, Y_plt, s=8,  alpha=0.35, color="orange",    zorder=2,
               label="Simulated hold value\nDiscounted payoff if you waited")
    ax.scatter(X_plt, cont,  s=14, alpha=0.65, color="royalblue", zorder=3, marker="x",
               label="LSM continuation estimate\nModel's per-path hold estimate")
    ax.plot(x_line, y_line,  color="royalblue", lw=2.2, zorder=4,
            label=f"LSM continuation curve\nFitted {BASIS.capitalize()} regression")
    ax.plot(x_line, ex_line, color="crimson",   lw=2,   zorder=4, ls="--",
            label="Exercise value\nIntrinsic payoff if exercised now")

    # S* threshold
    if s_star is not None:
        ax.axvline(s_star, color="grey", lw=1.2, ls=":", zorder=3)
        ax.annotate(
            f"S* = {s_star:.3f}\nexercise threshold",
            xy=(s_star, (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.5),
            xytext=(s_star + (x_max - s_star) * 0.32,
                    (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.68),
            fontsize=8, color="dimgrey",
            arrowprops={"arrowstyle": "->", "color": "dimgrey", "lw": 1},
            ha="center",
        )

    ax.set_xlabel(f"Asset Price at t = {mid}", fontsize=10)
    ax.set_ylabel("Option Value", fontsize=10)
    ax.set_title(
        f"LSM Decision Boundary at t = {mid}  ({label}, step {mid} of {N})\n"
        f"At each step: if exercise value > LSM curve, exercise. Otherwise hold.",
        fontsize=10, pad=10,
    )

    # Info box at top of right panel
    info = (
        f"American {label}\n"
        f"Price (LSM)\n"
        f"\n"
        f"  ${option_price:.4f}\n"
        f"\n"
        f"  S₀  {S0}\n"
        f"  K   {K}\n"
        f"  T   {T} yr\n"
        f"  r   {r:.1%}\n"
        f"  σ   {sigma:.0%}\n"
        f"  basis {BASIS[:3].upper()}\n"
        f"  N   {N} steps\n"
        f"  M   {M:,} paths\n"
        f"\n"
        f"  Price = mean\n"
        f"  discounted cash\n"
        f"  flow over all paths"
    )
    ax_side.text(
        0.06, 0.98, info,
        transform=ax_side.transAxes,
        fontsize=8.5, va="top", ha="left",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "white",
              "alpha": 0.95, "edgecolor": "lightgrey"},
    )

    # Legend at bottom of right panel (handles come from ax)
    handles, labels_leg = ax.get_legend_handles_labels()
    ax_side.legend(
        handles, labels_leg,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        fontsize=8,
        framealpha=0.0,
        edgecolor="lightgrey",
        handlelength=1.6,
        borderpad=0.4,
        labelspacing=0.9,
    )

    plt.savefig("images/lsm_regression.png", dpi=150)
    plt.close()
    print("Saved lsm_regression.png")
