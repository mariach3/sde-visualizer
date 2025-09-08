# Volatility & SDE Explorer
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

# --------------------------------------------------
# App config (only once)
# --------------------------------------------------
st.set_page_config(page_title="Volatility & SDE Explorer", layout="wide", page_icon="📈")

# === Repo-local defaults ===
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA_FILE = "aapl_options_subset_2020-08.csv.gz"

# ==================================================
# ===============  NAV / ROUTER  ===================
# ==================================================

NAV_LABELS = ["🏠 Home",
              "📚 Theory & Notes",
              "🧪 SDE Visualiser",
              "⚡ Performance & Benchmark",
              "📊 Vol Smile Explorer"]

# ---- initialize state once
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = NAV_LABELS[0]  # default to Home

# ---- consume any override set by buttons on the previous run
# (Important: do this BEFORE we instantiate the radio)
if "__page_override" in st.session_state:
    ov = st.session_state.pop("__page_override")
    if ov in NAV_LABELS:
        st.session_state["nav_page"] = ov

# Small helper the cards can use
def _navigate_to(label: str):
    """Set an override the router will consume on the next run."""
    st.session_state["__page_override"] = label

# ==================================================
# ===============  SHARED UTILITIES  ===============
# ==================================================

@st.cache_data(show_spinner=False)
def load_clean(path: str) -> pd.DataFrame:
    """Load and lightly clean the options CSV used in the Vol Smile Explorer page."""
    df = pd.read_csv(path, skipinitialspace=True, on_bad_lines="skip")
    df.columns = [c.strip().lstrip("[").rstrip("]") for c in df.columns]
    for col in ["QUOTE_READTIME", "QUOTE_DATE", "EXPIRE_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    numeric_cols = [
        "QUOTE_UNIXTIME", "QUOTE_TIME_HOURS", "UNDERLYING_LAST",
        "DTE", "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV",
        "C_VOLUME", "C_LAST", "C_SIZE", "C_BID", "C_ASK",
        "STRIKE", "P_BID", "P_ASK", "P_SIZE", "P_LAST",
        "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO", "P_IV",
        "P_VOLUME", "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["C_SIZE", "P_SIZE"]:
        if col in df.columns:
            df = df.drop(columns=col)
    return df


def _subset(df: pd.DataFrame, date, expiry) -> pd.DataFrame:
    d = pd.Timestamp(date).normalize()
    e = pd.Timestamp(expiry).normalize()
    lhs_date   = df["QUOTE_DATE"].dt.normalize()
    lhs_expiry = df["EXPIRE_DATE"].dt.normalize()
    sub = df.loc[(lhs_date == d) & (lhs_expiry == e)].copy()
    return sub


def _atm_iv(sub: pd.DataFrame, S0: float) -> float:
    if "C_IV" not in sub.columns:
        raise ValueError("C_IV column missing for market IVs.")
    mny = sub["STRIKE"] / S0
    near = sub.loc[mny.between(0.95, 1.05) & sub["C_IV"].notna(), "C_IV"]
    return float(np.median(near)) if not near.empty else float(np.nanmedian(sub["C_IV"]))


def set_state(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def help_block(title: str, body_md: str, presets: Optional[List[Dict]] = None):
    with st.expander(title):
        st.markdown(body_md)
        if presets:
            cols = st.columns(min(4, len(presets)))
            for i, p in enumerate(presets):
                with cols[i % len(cols)]:
                    st.button(p.get("label", "Use preset"), on_click=p.get("on_click"))


def _inject_academic_css():
    """Subtle 'paper' look + hero + cards."""
    st.markdown(
        """
        <style>
        .serif, .serif * { font-family: 'Georgia','Times New Roman',serif !important; }
        .box { padding: 0.75rem 1rem; border-radius: 10px; margin: 0.75rem 0; }
        .theorem { background: #f7faff; border-left: 5px solid #3b82f6; }
        .definition { background: #f6fff8; border-left: 5px solid #10b981; }
        .remark { background: #fffaf0; border-left: 5px solid #f59e0b; }
        .eqcap { font-size: 0.9rem; color:#4b5563; margin-top: -0.5rem; }

        /* Hero */
        .hero {
          background: linear-gradient(180deg,#f5f7fa, #ffffff);
          padding: 2.5rem 1.25rem;
          border-radius: 18px;
          text-align: center;
          box-shadow: 0 6px 16px rgba(0,0,0,0.06);
          margin-bottom: 1.25rem;
        }
        .hero h1 { margin: 0 0 .5rem 0; font-size: 2rem; }
        .hero h3 { margin: 0; font-weight: 400; color:#555; }

        /* Section cards */
        .card {
          border-radius: 16px;
          padding: 1rem 1.2rem;
          border: 1px solid #eef0f3;
          background: #fff;
          box-shadow: 0 2px 8px rgba(0,0,0,0.04);
          height: 100%;
        }
        .card h4 { margin: .1rem 0 .4rem 0; }
        .card p { margin: 0 0 .6rem 0; color: #555; font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

def _cover_welcome_block():
    st.markdown(
        """
        <div class="hero">
          <h1>📘 Volatility & SDE Explorer</h1>
          <h3>Four guided areas: Theory, Visualiser, Performance, and Real-Data Smiles</h3>
          <p style="color:#666; max-width:780px; margin: 0.75rem auto 0; font-size:1.05rem;">
            Start here. Pick a section below to dive into the content. You can always come back to this cover from the sidebar.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def _quick_start_block():
    with st.expander("📖 Quick Start (what to do first)", expanded=False):
        st.markdown(
            "- Skim **Brownian Motion** and **GBM** to ground the basics.\n"
            "- Jump to **Black–Scholes** for the PDE & closed form.\n"
            "- Read **Mean Reversion & OU** before the **Heston** section.\n"
            "- Try the **SDE Visualiser** for paths & distributions.\n"
            "- Use **Vol Smile Explorer** to compare models to **real data**."
        )

# -----------------------------------------------
# Image helper (for Performance page)
# -----------------------------------------------
def _find_asset(filename: str) -> Optional[Path]:
    try:
        here = Path(__file__).parent
    except NameError:
        here = Path.cwd()
    candidates = [
        here / filename,
        here / "assets" / filename,
        Path.cwd() / filename,
        Path.cwd() / "assets" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def show_image_safe(filename: str, caption: str = ""):
    img_path = _find_asset(filename)
    if img_path is not None:
        try:
            from PIL import Image  # Pillow
            with Image.open(img_path) as im:
                st.image(im, use_container_width=True, caption=caption)
                return
        except Exception as e:
            st.warning(f"Found image at {img_path}, but couldn’t open it: {e}. Showing a fallback chart below.")
    else:
        st.info(f"Image '{filename}' not found in common locations (./, ./assets). Showing a fallback chart below.")

    # Fallback chart
    N = np.array([50, 100, 200, 400, 800, 1600])
    t_gbm = 0.002 * N
    t_hes = 0.006 * N
    fig, ax = plt.subplots()
    ax.plot(N, t_gbm, label="GBM (illustrative)")
    ax.plot(N, t_hes, label="Heston (illustrative)")
    ax.set_xlabel("Time steps N")
    ax.set_ylabel("Runtime (relative units)")
    ax.set_title("GBM vs Heston timing vs time steps — fallback view")
    ax.grid(True); ax.legend()
    st.pyplot(fig)
    st.caption("Place the real PNG as './assets/benchmark_sde_timing.png' to display it here.")

# ==================================================
# ==================== HOME ========================
# ==================================================
def page_home():
    _inject_academic_css()
    _cover_welcome_block()
    #_quick_start_block()

    st.markdown("#### Choose a section")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card"><h4>📚 Theory & Notes</h4><p>Concise notes: Brownian motion, GBM, Black–Scholes, OU, Heston, calibration and more.</p>', unsafe_allow_html=True)
        st.button("Open Theory & Notes ▶", key="home_open_theory",
                  on_click=_navigate_to, args=("📚 Theory & Notes",))
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><h4>🧪 SDE Visualiser</h4><p>Interactively simulate GBM and Heston, see fan charts, variance dynamics, and return distributions.</p>', unsafe_allow_html=True)
        st.button("Open Visualiser ▶", key="home_open_vis",
                  on_click=_navigate_to, args=("🧪 SDE Visualiser",))
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card"><h4>⚡ Performance & Benchmark</h4><p>Quick timing experiments for GBM vs Heston; scaling intuition and a reference chart.</p>', unsafe_allow_html=True)
        st.button("Open Performance ▶", key="home_open_perf",
                  on_click=_navigate_to, args=("⚡ Performance & Benchmark",))
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><h4>📊 Vol Smile Explorer</h4><p>Load AAPL option data; compare market IVs against BS (flat σ) and calibrated Heston.</p>', unsafe_allow_html=True)
        st.button("Open Vol Smile ▶", key="home_open_smile",
                  on_click=_navigate_to, args=("📊 Vol Smile Explorer",))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Tip: Use the sidebar at any time. Home is fully separate from the other pages.")

# ==================================================
# =====  THEORY PAGE (cards → full section view) ===
# ==================================================

THEORY_SECTIONS = [
    ("brownian",     "Standard Brownian Motion", "Foundation of continuous-time randomness.", "🟦"),
    ("ito",          "Itô's Lemma",              "Chain rule for stochastic processes.",      "🧮"),
    ("gbm_def",      "Geometric Brownian Motion","Definition, solution & log-normality.",     "📈"),
    ("gbm_var",      "GBM: different variances", "Same noise, different σ to compare paths.", "📊"),
    ("bs_theory",    "Black–Scholes (theory)",   "GBM under Q, PDE and call formula.",        "🧠"),
    ("bs_calc",      "BS mini-calculator",       "Quick prices for C/P with d₁, d₂.",         "⚙️"),
    ("limitations",  "Limitations & Smile",      "Why BS is too rigid; smile examples.",      "🎯"),
    ("meanrev",      "Mean-Reverting Process",   "OU intuition & link to Heston.",            "↩️"),
    ("ou_explorer",  "OU Explorer",              "Simulate mean reversion interactively.",    "🧪"),
    ("heston",       "Heston model",             "Dynamics, CF pricing & intuition.",         "🌀"),
    ("heston_calib", "Heston calibration",       "Objective, LM update & tips.",              "🧷"),
    ("glossary",     "Glossary",                 "Key terms at a glance.",                    "🔎"),
]

def _set_theory_view(section_id: Optional[str]):
    st.session_state["theory_view"] = section_id

def _theory_cards_view():
    st.markdown("#### Jump to a section")
    rows = (len(THEORY_SECTIONS) + 2) // 3
    idx = 0
    for _ in range(rows):
        c1, c2, c3 = st.columns(3)
        for col in (c1, c2, c3):
            if idx >= len(THEORY_SECTIONS):
                col.empty()
                continue
            sid, title, desc, emoji = THEORY_SECTIONS[idx]
            with col:
                st.markdown(f'<div class="card"><h4>{emoji} {title}</h4><p>{desc}</p>', unsafe_allow_html=True)
                st.button("Open", key=f"open_{sid}", on_click=_set_theory_view, args=(sid,))
                st.markdown("</div>", unsafe_allow_html=True)
            idx += 1

def _back_to_theory_home():
    cols = st.columns([0.18, 0.82])
    with cols[0]:
        # No st.rerun() here — Streamlit will rerun automatically after click
        st.button("⬅️ Back to Theory Home", on_click=_set_theory_view, args=(None,))
    with cols[1]:
        pass
    st.markdown("---")

# ---- Section renderers  ----
def _section_brownian():
    st.header("Standard Brownian Motion")
    st.markdown('<div class="box definition">Definition (Standard Brownian Motion).</div>', unsafe_allow_html=True)
    st.latex(r"\text{A standard Brownian motion is a random process:} \quad W = \{W_t : t \in [0,\infty)\}, \quad W_t \in \mathbb{R} \quad \text{such that:}")
    st.latex(r"W_0 = 0 \quad \text{(with probability 1)}")
    st.latex(r"\text{W has stationary increments: } W_t - W_s \sim W_{t-s}, \quad \forall s<t")
    st.latex(r"\text{W has independent increments: } W_{t_1},\, W_{t_2}-W_{t_{1}},\, \ldots,\, W_{t_n}-W_{t_{n-1}} \text{ are independent}")
    st.latex(r"W_t \sim \mathcal{N}(0,t), \quad \forall t>0")
    st.latex(r"\mathbb{P}\!\left(t \mapsto W_t \text{ is continuous on } [0,\infty)\right) = 1")

    st.write("**Interactive illustration — Simulating Brownian motion sample paths:**")
    st.markdown("""
**How to use this panel:**  
- Adjust the sliders to change the **time horizon (T)**, **number of steps (N)**, and **number of paths (M)**.  
- Use the **seed** box to reproduce the exact same random paths, or change it to generate new ones.  
- Tick **increments histogram** to see that ΔW is normally distributed with variance Δt.  
""")
    c1, c2, c3 = st.columns(3)
    with c1:
        T = st.slider("Horizon T", 0.25, 5.0, 1.0, 0.25, key="bm_T")
        M = st.slider("Paths M", 1, 30, 5, 1, key="bm_M")
    with c2:
        N = st.slider("Steps N", 100, 4000, 800, 100, key="bm_N")
        seed = st.number_input("Seed", min_value=0, value=0, step=1, key="bm_seed")
    with c3:
        show_hist = st.checkbox("Increments histogram", value=True, key="bm_hist")

    rng = np.random.default_rng(int(seed))
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    dW = rng.normal(0.0, np.sqrt(dt), size=(M, N))
    W = np.concatenate([np.zeros((M, 1)), np.cumsum(dW, axis=1)], axis=1)

    fig_paths, axp = plt.subplots()
    for i in range(M):
        axp.plot(t, W[i], lw=1.5, alpha=0.9)
    axp.set_title(f"Standard Brownian motion — {M} path(s), T={T:g}, N={N}")
    axp.set_xlabel("t"); axp.set_ylabel("W(t)")
    axp.grid(True)
    st.pyplot(fig_paths)
    st.caption("Properties: W₀=0, stationary & independent increments, W(t) ~ N(0, t).")

    if show_hist:
        fig_h, axh = plt.subplots()
        inc = dW.reshape(-1)
        axh.hist(inc, bins=50, density=True, alpha=0.85)
        xg = np.linspace(inc.min(), inc.max(), 400)
        axh.plot(xg, norm.pdf(xg, 0.0, np.sqrt(dt)), linestyle="--", label="N(0, dt)")
        axh.set_title("Distribution of increments ΔW ~ N(0, dt)")
        axh.set_xlabel("ΔW"); axh.set_ylabel("Density")
        axh.legend(); axh.grid(True)
        st.pyplot(fig_h)
        st.caption(f"Empirical mean ≈ {float(inc.mean()):.4f}, variance ≈ {float(inc.var()):.5f} (theory: Var=dt={dt:.5f}).")

def _section_ito():
    st.header("Itô's Lemma")
    st.markdown('<div class="box remark">Itô process.</div>', unsafe_allow_html=True)
    st.write("An Itô process is a stochastic process that can be expressed as a sum of a drift term and a diffusion term driven by Brownian motion. Formally an Itô process can be written as:")
    st.latex(r"dX_t = a(X_t,t)\,dt + b(X_t,t)\,dW_t")
    st.latex(r"\text{where } a(X_t,t) \text{ is the drift, } b(X_t,t) \text{ the diffusion, and } dW_t \text{ the Brownian increment.}")
    st.info(r"**Itô’s Lemma.** If $X_t$ is an Itô process with drift $a$ and diffusion $b$, then for smooth $f$:")
    st.latex(r"df(X_t,t) = \left(f_t + a f_x + \tfrac12 b^2 f_{xx}\right)dt + b f_x\, dW_t")
    st.caption(r"Here $f_t=\partial f/\partial t$, $f_x=\partial f/\partial x$, $f_{xx}=\partial^2 f/\partial x^2$.")

def _section_gbm_def():
    st.header("Geometric Brownian Motion (definition & solution)")
    st.info(r"""
**Definition (Geometric Brownian Motion).** In mathematical finance, an asset price $\{S_t\}_{t \ge 0}$ is said to follow a
Geometric Brownian Motion (GBM) if it satisfies:
""")
    st.latex(r"dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t,\qquad S_0>0,\ \mu\in\mathbb{R},\ \sigma>0.")
    st.markdown(r"Below we show the explicit solution of the GBM SDE and log-normality of $S_t$:")    
    st.markdown(r"**(1) Setup.** Let $f(s)=\ln s$. Then $f_s=1/s$, $f_{ss}=-1/s^2$, $f_t\equiv 0$.")
    st.markdown(r"**(2) Apply Itô to** $f(S_t,t)=\ln S_t$ with $a=\mu S_t$ and $b=\sigma S_t$:")
    st.latex(
        r"""
\begin{aligned}
d\ln S_t
&= \Big(0 + (\mu S_t)\cdot \tfrac{1}{S_t} + \tfrac12 (\sigma S_t)^2 \cdot \big(-\tfrac{1}{S_t^2}\big)\Big)dt
+ (\sigma S_t)\cdot \tfrac{1}{S_t}\,dW_t\\
&= \left(\mu - \tfrac12 \sigma^2\right)dt + \sigma\,dW_t.
\end{aligned}
""")
    st.markdown(r"**(3) Integrate** 0→t and use $W_0=0$:")
    st.latex(
        r"""
\ln S_t - \ln S_0
= \int_0^t \left(\mu - \tfrac12 \sigma^2\right) ds + \int_0^t \sigma\, dW_s
= \left(\mu - \tfrac12 \sigma^2\right)t + \sigma W_t.
""")
    st.markdown(r"**(4) Exponentiate:**  $S_t = S_0 \exp\!\big((\mu - \tfrac12 \sigma^2)t + \sigma W_t\big)$.")
    st.markdown(r"**(5) Distribution.** Since $W_t\sim\mathcal N(0,t)$,")
    st.latex(r"\ln S_t \sim \mathcal N\!\Big(\ln S_0 + (\mu-\tfrac12\sigma^2)t,\ \sigma^2 t\Big)")
    st.markdown(r"so $S_t$ is **log-normal**.")
    st.markdown(
        r"_Risk-neutral with dividends._ Under $\mathbb{Q}$ with dividend yield $q$, "
        r"$dS_t=(r-q)S_t\,dt+\sigma S_t\,dW_t^{\mathbb Q}$, hence"
    )
    st.latex(r"S_t = S_0 \exp\!\big((r-q-\tfrac12\sigma^2)t + \sigma W_t^{\mathbb Q}\big).")

def _section_gbm_var():
    st.header("GBM: Realizations with different variances")
    st.markdown("Set μ=1 and plot one GBM path for each σ on the **same Brownian path**.")
    c1, c2, c3 = st.columns(3)
    with c1:
        S0 = st.number_input("Initial level S₀", value=50.0, step=10.0, key="rep_S0")
        mu = st.number_input("Drift μ", value=1.0, step=0.1, key="rep_mu")
        T  = st.number_input("Horizon T (years)", value=2.6, step=0.1, key="rep_T")
    with c2:
        N  = st.slider("Steps N", 200, 4000, 2000, 100, key="rep_N")
        seed = st.number_input("Seed", min_value=0, value=42, step=1, key="rep_seed")
    with c3:
        s1 = st.number_input("σ₁", value=0.8,  step=0.1, key="rep_sigma1")
        s2 = st.number_input("σ₂", value=1.2,  step=0.1, key="rep_sigma2")
        s3 = st.number_input("σ₃", value=1.6,  step=0.1, key="rep_sigma3")

    rng = np.random.default_rng(int(seed))
    dt = T / N
    t  = np.linspace(0.0, T, N + 1)
    dW = rng.normal(0.0, np.sqrt(dt), size=N)
    W  = np.concatenate([[0.0], np.cumsum(dW)])  # shape (N+1,)

    def gbm_path(S0, mu, sigma, t, W):
        return S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    S_1 = gbm_path(S0, mu, s1, t, W)
    S_2 = gbm_path(S0, mu, s2, t, W)
    S_3 = gbm_path(S0, mu, s3, t, W)

    fig, ax = plt.subplots()
    ax.plot(t, S_1, label=fr"$\sigma = {s1:.1f}$")
    ax.plot(t, S_2, label=fr"$\sigma = {s2:.1f}$")
    ax.plot(t, S_3, label=fr"$\sigma = {s3:.1f}$")
    ax.set_title(fr"Realizations of GBM with different variances, $\mu = {mu:.2f}$")
    ax.set_xlabel("t"); ax.set_ylabel("x")
    ax.grid(True); ax.legend(loc="upper left")
    st.pyplot(fig)
    st.caption("Same Brownian path for all; only σ changes.")

def _section_bs_theory():
    st.header("Black–Scholes (Theory, PDE & Solution)")
    st.markdown('<div class="box definition">Risk-neutral dynamics (GBM with dividends).</div>', unsafe_allow_html=True)
    st.latex(r"dS_t = (r-q)S_t\,dt + \sigma S_t\, dW_t^{\mathbb Q}")
    st.markdown('<div class="box assumptions">Model assumptions:</div>', unsafe_allow_html=True)
    st.markdown(r"""
1. Stock price follows a GBM with constant drift and volatility.  
2. No arbitrage opportunities.  
3. Constant interest rate \(r\).  
4. Continuous trading, no transaction costs or taxes.  
5. European-style exercise only.  
""")
    st.markdown('<div class="box theorem">Black–Scholes PDE (European payoffs).</div>', unsafe_allow_html=True)
    st.latex(r"V_t + (r-q)S V_S + \tfrac12 \sigma^2 S^2 V_{SS} - r V = 0,\qquad V(T,S)=\text{payoff}(S).")
    st.markdown('<div class="box formula">Closed-form European call:</div>', unsafe_allow_html=True)
    st.latex(
        r"""
C = S_0 e^{-q\tau}\Phi(d_1) - K e^{-r\tau}\Phi(d_2),\qquad
d_1 = \frac{\ln(S_0e^{-q\tau}/K) + \tfrac12\sigma^2\tau}{\sigma\sqrt{\tau}},\quad
d_2 = d_1 - \sigma\sqrt{\tau}.
""")
    st.markdown('<div class="box note">Volatility in practice:</div>', unsafe_allow_html=True)
    st.markdown(r"""
    * **Historical volatility**: estimated from past returns (standard deviation of historical log-returns).
    * **Implied volatility**: the σ that matches observed option prices (by inverting the BS formula).""")
    st.markdown('<div class="box warning">Limitations:</div>', unsafe_allow_html=True)
    st.markdown(r"""
* Assumes constant σ and normally distributed returns.  
* Cannot reproduce volatility smiles/skews.  
* Tends to underprice out-of-the-money options.  
* Performance is worse in crises/periods of market turbulence when volatility clusters.
""")

def _section_bs_calc():
    st.header("Black–Scholes mini-calculator")
    from math import log, sqrt, exp
    from scipy.stats import norm as _N
    S0 = st.number_input("Spot S₀", value=100.0); K = st.number_input("Strike K", value=100.0)
    r  = st.number_input("Risk-free r", value=0.02); q = st.number_input("Dividend q", value=0.00)
    vol= st.number_input("Vol σ", value=0.20); tau = st.number_input("Time to maturity τ (yrs)", value=0.5)
    if st.button("Compute BS"):
        d1 = (log(S0/K)+(r-q+0.5*vol**2)*tau)/(vol*sqrt(tau))
        d2 = d1 - vol*sqrt(tau)
        C  = S0*exp(-q*tau)*_N.cdf(d1) - K*exp(-r*tau)*_N.cdf(d2)
        P  = K*exp(-r*tau)*_N.cdf(-d2) - S0*exp(-q*tau)*_N.cdf(-d1)
        st.write(f"Price (Call) = **{C:.4f}**, Price (Put) = **{P:.4f}** | d1={d1:.3f}, d2={d2:.3f}")

def _section_limitations():
    st.header("Limitations & Volatility Smile")
    st.markdown(r"""Where Black–Scholes falls short: 
- **Constant volatility** ⇒ flat IV surface; real markets show **smile/skew** especially at short maturities.
- **Log-normal returns** ⇒ zero skewness/kurtosis; real returns have **negative skew** and **fat tails**.
- **Volatility dynamics**: BS has i.i.d. log-returns; data shows **volatility clustering**.
- **Term structure**: short maturities curve/tilt more; curvature typically **decreases with maturity**.
""")
    m = np.linspace(0.6, 1.4, 161)
    logm = np.log(m)
    def smile(logm, base, beta, kappa):
        return base + beta*logm + kappa*(logm**2)
    iv_short = smile(logm, base=0.24, beta=-0.18, kappa=0.55)
    iv_med   = smile(logm, base=0.22, beta=-0.12, kappa=0.40)
    iv_long  = smile(logm, base=0.20, beta=-0.07, kappa=0.28)
    fig, ax = plt.subplots()
    ax.plot(m, iv_short, label="1M")
    ax.plot(m, iv_med,   label="6M")
    ax.plot(m, iv_long,  label="2Y")
    atm_idx = int(np.argmin(np.abs(m - 1.0)))
    ax.axhline(iv_med[atm_idx], linestyle="--", alpha=0.8, label="Black–Scholes: flat σ (ATM)")
    ax.set_xlabel("Moneyness  $K/F$"); ax.set_ylabel("Implied volatility")
    ax.set_title("Volatility smile vs Black–Scholes flat σ")
    ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig)
    st.caption("Shorter maturities curve/tilt more; deviation from dashed line shows why constant-σ BS is rigid.")

def _section_meanrev():
    st.header("Mean-Reverting Process (definition)")
    st.markdown('<div class="box definition">Mean-reverting process.</div>', unsafe_allow_html=True)
    st.markdown(
        "A process is *mean-reverting* if its drift pulls it back towards a long-run level "
        "$\\theta$: above $\\theta$ the drift is negative; below $\\theta$ it is positive. "
        "A famous example is the **Ornstein–Uhlenbeck (OU) process**."
    )
    st.latex(r"dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t,\qquad \kappa>0,\ \sigma>0.")
    st.markdown(
        "- $\\kappa$ (**speed**): larger $\\kappa$ → faster pull towards $\\theta$.\n"
        "- $\\theta$ (**long-run level**).\n"
        "- $\\sigma$ (**noise scale**)."
    )
    st.caption(r"As $t\to\infty$: $X_t \sim \mathcal N(\theta,\ \sigma^2/(2\kappa))$.")
    st.markdown("**Link to Heston variance (CIR type):**")
    st.latex(r"dv_t = \kappa(\theta - v_t)\,dt + \sigma_v \sqrt{v_t}\,dW_t,\qquad v_t \ge 0.")
    st.caption("Heston uses mean-reverting variance $v_t$ (with leverage $\\rho<0$) to generate equity skew.")

def _section_ou_explorer():
    st.header("Ornstein–Uhlenbeck Explorer")
    st.markdown("""
**How to use**  
- Slide **κ**: higher κ pulls X(t) back to **θ** faster and shrinks stationary variance.  
- Adjust **θ** and **σ**.  
- Use **Seed** to reproduce the path; **Common noise** shows mean reversion clearly.
""")
    c1, c2, c3 = st.columns(3)
    with c1:
        T = st.slider("Horizon T", 0.5, 10.0, 2.0, 0.5, key="ou_T")
        N = st.slider("Steps N", 100, 4000, 800, 100, key="ou_N")
        seed = st.number_input("Seed", min_value=0, value=1, step=1, key="ou_seed")
    with c2:
        kappa = st.slider("Mean reversion κ", 0.1, 8.0, 2.0, 0.1, key="ou_kappa")
        theta = st.slider("Long-run mean θ", -2.0, 2.0, 0.0, 0.1, key="ou_theta")
        sigma = st.slider("Volatility σ", 0.05, 2.0, 0.6, 0.05, key="ou_sigma")
    with c3:
        x0 = st.slider("Initial value x₀", -3.0, 3.0, 1.5, 0.1, key="ou_x0")
        M  = st.slider("Number of paths M", 1, 50, 8, 1, key="ou_M")
        common_noise = st.checkbox("Use same noise for all paths", value=False, key="ou_common")

    rng = np.random.default_rng(int(seed))
    dt = T / N
    t = np.linspace(0.0, T, N+1)
    if common_noise:
        Z = rng.normal(size=N)
        Z = np.tile(Z, (M, 1))
    else:
        Z = rng.normal(size=(M, N))
    X = np.empty((M, N+1))
    X[:, 0] = x0
    for i in range(1, N+1):
        X[:, i] = X[:, i-1] + kappa*(theta - X[:, i-1])*dt + sigma*np.sqrt(dt)*Z[:, i-1]
    m_t = theta + (x0 - theta) * np.exp(-kappa * t)
    fig_ou, ax = plt.subplots()
    ax.plot(t, X.T, lw=1.0, alpha=0.6)
    ax.plot(t, m_t, ls="--", lw=2.0, label="E[Xₜ]")
    ax.axhline(theta, ls=":", alpha=0.9, label="θ (long-run mean)")
    ax.set_title(f"Ornstein–Uhlenbeck sample paths (M={M})")
    ax.set_xlabel("t"); ax.set_ylabel("X(t)")
    ax.grid(True); ax.legend()
    st.pyplot(fig_ou)
    st.caption("OU mean-reverts to θ at speed κ. σ sets shock size.")

def _section_heston():
    st.header("Heston Model — dynamics, intuition & pricing")
    st.markdown('<div class="box definition">Risk-neutral dynamics.</div>', unsafe_allow_html=True)
    st.latex(
        r"""
\begin{aligned}
dS_t &= \mu\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{(1)},\\
dv_t &= \kappa(\theta - v_t)\,dt + \sigma_v \sqrt{v_t}\,dW_t^{(2)}, \qquad
d\langle W^{(1)},W^{(2)}\rangle_t=\rho\,dt.
\end{aligned}
""")
    st.markdown(
        "- $v_t$ is the **instantaneous variance** (CIR process).\n"
        "- $\\rho$ captures the **leverage effect** (equities: typically negative → left skew).\n"
        "- Under $\\mathbb{Q}$, set $\\mu=r-q$."
    )
    st.markdown('<div class="box remark">Parameter roles (quick intuition).</div>', unsafe_allow_html=True)
    st.markdown(
        "- $v_0$: initial variance.\n"
        "- $\\kappa$: mean-reversion speed.\n"
        "- $\\theta$: long-run variance.\n"
        "- $\\sigma_v$: vol-of-vol (smile curvature & tails).\n"
        "- $\\rho$: spot–vol correlation (skew).\n"
        "- **Feller:** $2\\kappa\\theta \\ge \\sigma_v^2$ keeps $v_t>0$."
    )
    st.markdown('<div class="box theorem">European option pricing by Fourier transform.</div>', unsafe_allow_html=True)
    st.latex(
        r"""
C_0 = S_0 e^{-q\tau} P_1 - K e^{-r\tau} P_2,\qquad
P_j = \tfrac{1}{2} + \tfrac{1}{\pi}\int_0^\infty 
\Re\!\left[\frac{e^{-i\varphi \ln K}\,f_j(\varphi)}{i\varphi}\right] d\varphi.
""")
    st.markdown("Using the ‘Little Heston Trap’ branch for numerical stability on long maturities.")
    help_block(
        "Heston parameters interpretation:",
        "- **κ (kappa)** — Higher κ ⇒ faster reversion of variance to θ.\n"
        "- **σᵥ (vol-of-vol)** — Larger σᵥ ⇒ heavier tails and stronger smile curvature.\n"
        "- **ρ (rho)** — More negative ρ ⇒ stronger downside skew.\n"
    )




def _section_heston_calib():
    st.header("Calibration of the Heston Model")

    # Calibration problem
    st.markdown('<div class="box definition">Calibration problem.</div>', unsafe_allow_html=True)
    st.markdown(
        "We estimate parameters $\\Theta=(\\kappa,\\theta,\\sigma,\\rho,v_0)$ "
        "so that model option prices (or implied vols) match observed market data."
    )
    st.latex(r"\min_{\Theta}\; \tfrac{1}{2}\sum_{i=1}^N \big(P_i^{\text{mkt}} - P_i^{\text{model}}(\Theta)\big)^2")
    st.caption("Objective: minimise squared pricing errors (other metrics like MAE, RMSE, or MAPE can also be used).")

    # Error metrics
    st.markdown("**Common error metrics:** mean absolute error (MAE), root mean squared error (RMSE), "
                "and mean absolute percentage error (MAPE).")

    # LM update
    st.markdown('<div class="box theorem">Levenberg–Marquardt (LM) update.</div>', unsafe_allow_html=True)
    st.latex(r"\Delta\Theta = - (J^\top J + \mu I)^{-1} J^\top r")
    st.caption("LM blends gradient descent (stable) with Gauss–Newton (fast near optimum).")

    # Constraints
    st.markdown(
        "**Constraints:** $\\kappa,\\theta,\\sigma>0$, $-1<\\rho<1$, $v_0\\ge0$. "
        "**Feller condition:** $2\\kappa\\theta\\ge\\sigma^2$."
    )

    st.markdown(
        "In practice, calibration uses numerical optimisation (e.g. `scipy.optimize.least_squares`) "
        "on market option data. Results depend on data quality and initial guesses, and may be "
        "vega-weighted to emphasise contracts most sensitive to volatility."
    )



def _section_glossary():
    st.header("Glossary")
    glossary = {
        "moneyness": "How far in/out of the money an option is. Ratio K/F or K/S₀; >1 = ITM put, <1 = ITM call.",
        "implied volatility": "The volatility σ that makes the model price equal the observed market price.",
        "historical volatility": "Volatility from past returns, often annualised stdev of log-returns.",
        "martingale": "Under Q, discounted prices have zero drift.",
        "risk-neutral measure": "Measure under which discounted assets are martingales.",
        "numeraire": "Benchmark asset (e.g. bond). Choice changes the martingale measure.",
        "characteristic function": "Fourier transform of a distribution; core to Heston pricing.",
        "feller condition": "2κθ ≥ σ²ᵥ ensures vₜ stays positive (CIR property).",
        "term structure": "How vol/parameters vary with maturity.",
        "leverage effect": "ρ < 0: prices fall ↔ vol rises → left skew.",
        "volatility smile/skew": "Shape of IV vs strike; absent in BS, natural in Heston.",
        "mean reversion": "Tendency to revert to θ; OU is a classic example.",
        "OU process": "dXₜ = κ(θ − Xₜ)dt + σ dWₜ.",
        "CIR process": "dvₜ = κ(θ − vₜ)dt + σ√vₜ dWₜ.",
        "little heston trap": "Numerical branch choice to avoid instabilities.",
        "calibration": "Fitting parameters to market data (minimise RMSE).",
    }
    for k, v in glossary.items():
        st.markdown(f"**{k.capitalize()}** — {v}")

SECTION_RENDER = {
    "brownian": _section_brownian,
    "ito": _section_ito,
    "gbm_def": _section_gbm_def,
    "gbm_var": _section_gbm_var,
    "bs_theory": _section_bs_theory,
    "bs_calc": _section_bs_calc,
    "limitations": _section_limitations,
    "meanrev": _section_meanrev,
    "ou_explorer": _section_ou_explorer,
    "heston": _section_heston,
    "heston_calib": _section_heston_calib,
    "glossary": _section_glossary,
}

def page_theory():
    _inject_academic_css()
    view = st.session_state.get("theory_view", None)

    if view is None:
        _cover_welcome_block()
        _quick_start_block()
        _theory_cards_view()
    else:
        _back_to_theory_home()
        title = next((t for sid, t, _, _ in THEORY_SECTIONS if sid == view), None)
        if title:
            st.subheader(f"{title}")
        SECTION_RENDER.get(view, lambda: st.info("Section not found."))()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("Open SDE Visualiser ▶", on_click=_navigate_to, args=("🧪 SDE Visualiser",))
    with c2:
        st.button("Open Performance ▶", on_click=_navigate_to, args=("⚡ Performance & Benchmark",))
    with c3:
        st.button("Open Vol Smile ▶", on_click=_navigate_to, args=("📊 Vol Smile Explorer",))

# ==================================================
# ============  PAGE 1: SDE VISUALISER  ============
# ==================================================

def simulate_gbm_paths(S0, T, mu, sigma, M, N):
    dt = T / N
    t_local = np.linspace(0, T, N)
    paths = np.zeros((M, N))
    for i in range(M):
        W = np.cumsum(np.random.randn(N)) * np.sqrt(dt)
        paths[i] = S0 * np.exp((mu - 0.5 * sigma**2) * t_local + sigma * W)
    return paths


def simulate_heston_paths(S0, T, r, v0, kappa, theta, sigma_v, rho, M, N, return_variance: bool = False):
    """Euler scheme for Heston. If return_variance=True, also returns the variance paths array (M x N)."""
    dt = T / N
    paths = np.zeros((M, N))
    vars_ = np.zeros((M, N)) if return_variance else None
    for j in range(M):
        S = np.zeros(N)
        v = np.zeros(N)
        S[0], v[0] = S0, max(v0, 1e-12)
        for i in range(1, N):
            z1 = np.random.normal()
            z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * np.random.normal()
            v_prev = max(v[i-1], 1e-12)
            v[i] = np.abs(v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(v_prev) * np.sqrt(dt) * z2)
            S[i] = S[i-1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev) * np.sqrt(dt) * z1)
        paths[j] = S
        if return_variance:
            vars_[j] = v
    return (paths, vars_) if return_variance else paths


def page_sde_visualiser():
    st.title("Stochastic Differential Equation Visualiser (Black–Scholes & Heston)")

    # Sidebar parameters
    with st.sidebar:
        st.header("🧮 Global Parameters")
        S0 = st.number_input("Initial Asset Price (S₀)", value=st.session_state.get("S0", 100.0), key="S0")
        K = st.number_input("Strike Price (K)", value=st.session_state.get("K", 100.0), key="K")
        r = st.slider("Risk-Free Rate (r)", 0.0, 0.1, float(st.session_state.get("r", 0.03)), key="r")
        T = st.slider("Time Horizon (Years)", 0.5, 5.0, float(st.session_state.get("T", 1.0)), key="T")
        N = st.slider("Time Steps", 100, 1000, int(st.session_state.get("N", 250)), key="N")
        M = st.slider("Simulations (Monte Carlo)", 10, 2000, int(st.session_state.get("M", 250)), key="M")
        t = np.linspace(0, T, N)

    with st.sidebar.expander("Black–Scholes Parameters"):
        mu = st.slider("Drift (μ)", -0.1, 0.2, float(st.session_state.get("mu", 0.05)), key="mu")
        sigma = st.slider("Volatility (σ)", 0.01, 1.0, float(st.session_state.get("sigma", 0.2)), key="sigma")

    with st.sidebar.expander("Heston Parameters"):
        v0 = st.slider("Initial Variance (v₀)", 0.01, 0.5, float(st.session_state.get("v0", 0.04)), key="v0")
        kappa = st.slider("Mean Reversion (κ)", 0.1, 5.0, float(st.session_state.get("kappa", 2.0)), key="kappa")
        theta = st.slider("Long-Term Variance (θ)", 0.01, 0.5, float(st.session_state.get("theta", 0.04)), key="theta")
        sigma_v = st.slider("Volatility of Volatility (σᵥ)", 0.01, 1.0, float(st.session_state.get("sigma_v", 0.3)), key="sigma_v")
        rho = st.slider("Correlation (ρ)", -1.0, 1.0, float(st.session_state.get("rho", -0.7)), key="rho")

    # Simulations
    gbm_paths = simulate_gbm_paths(S0, T, mu, sigma, M, N)
    heston_paths, heston_vars = simulate_heston_paths(S0, T, r, v0, kappa, theta, sigma_v, rho, M, N, return_variance=True)

    # --- Normalize prices by S0 so S0 ≡ 1 ---
    _eps = 1e-12
    gbm_norm = gbm_paths / max(S0, _eps)
    heston_norm = heston_paths / max(S0, _eps)

    # Layout: two columns
    col1, col2 = st.columns(2)

        # Left column — side-by-side paths, no histograms/quantiles/variance
    # Left column (simplified sample paths only)
    with col1:
        st.subheader("📈 Simulated Price Paths (normalized to S₀)")
        logy = st.checkbox("Use log y-scale", value=False)

        fig, ax = plt.subplots()
        show = min(12, M)  # number of sample paths to show
        idx = np.random.choice(M, show, replace=False)

        # Plot GBM paths (blue, light)
        for i in idx:
            ax.plot(t, gbm_norm[i], linewidth=1, alpha=0.7, color="steelblue")

        # Plot Heston paths (orange, light)
        for i in idx:
            ax.plot(t, heston_norm[i], linewidth=1, alpha=0.7, color="darkorange")

        ax.set_title(f"{show} random sample paths (GBM and Heston)")
        ax.set_xlabel("Time")
        ax.set_ylabel("S(t) / S₀")
        if logy:
            ax.set_yscale("log")
        ax.grid(True)
        ax.legend(["GBM paths", "Heston paths"])
        st.pyplot(fig)

        help_block(
            "How to read the plot",
            (
                "- Each line is one simulated path of the asset price.\n"
                "- Paths are normalised so they all start at 1.\n"
                "- The blue lines are generated under the GBM model, the orange lines under the Heston model.\n"
                "- The plot shows how simulated prices can evolve randomly over time."
            )
        )

        st.caption("All prices are scaled so $S₀=1$ for easy comparison.")




    # Right column: return distribution plot
    with col2:
        st.subheader("📊 Return Distribution vs Normal Distribution")

        # Terminal log-returns relative to S0
        returns_gbm = np.log(np.maximum(gbm_paths[:, -1], 1e-12) / max(S0, 1e-12))
        returns_heston = np.log(np.maximum(heston_paths[:, -1], 1e-12) / max(S0, 1e-12))

        fig2, ax2 = plt.subplots()
        ax2.hist(returns_gbm, bins=30, alpha=0.5, label="GBM Terminal Log-Returns", density=True)
        ax2.hist(returns_heston, bins=30, alpha=0.5, label="Heston Terminal Log-Returns", density=True)

        x_min = float(min(returns_gbm.min(), returns_heston.min()))
        x_max = float(max(returns_gbm.max(), returns_heston.max()))
        x = np.linspace(x_min, x_max, 200)
        ax2.plot(x, norm.pdf(x, np.mean(returns_gbm), np.std(returns_gbm)), linestyle="--", label="Normal (Black-Scholes)")
        ax2.plot(x, norm.pdf(x, np.mean(returns_heston), np.std(returns_heston)), linestyle="--", label="Heston")

        ax2.set_xlabel("Log Returns")
        ax2.set_ylabel("Density")
        ax2.set_title("Comparison of Terminal Return Distributions (GBM vs Heston)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        help_block(
            "Help — Reading the Return Distributions",
            (
                "- GBM (Black-Scholes) → log-returns follow a Normal distribution by design; histogram should match dashed curve.\n"
                "- Heston → stochastic volatility creates fat tails, meaning more extreme returns than Normal predicts.\n"
                "- Why it matters: fat tails are common in real markets and imply higher risk of large price moves.\n"
                "- How to read: if histogram tails are taller than the dashed curve, the model is producing heavier tails.\n"
                "- Experiment: increase σᵥ or |ρ| to make fat tails more pronounced.\n\n"
            ),
            presets=[
                {"label": "Fatter tails: σᵥ=1.0, ρ=-0.9", "on_click": lambda: set_state(sigma_v=1.0, rho=-0.9)},
                {"label": "Near-GBM: σᵥ=0.05, ρ=0.0", "on_click": lambda: set_state(sigma_v=0.05, rho=0.0)},
            ]
        )

# ==================================================
# =======  PAGE 2: PERFORMANCE & BENCHMARK  ========
# ==================================================
def page_performance():
    st.title("⚡ Performance & Benchmark")
    st.caption("Interactive timing plus a reference chart for larger simulation sizes.")

    # Two columns: left = interactive, right = help/remark with precomputed image
    left, right = st.columns([1.2, 0.8])

    with left:
        st.subheader("Interactive Performance Benchmark")
        bm_M = st.slider("Number of Paths (M)", 10, 5000, st.session_state.get("bm_M", 100), step=10, key="bm_M")
        bm_N = st.slider("Number of Time Steps (N)", 10, 5000, st.session_state.get("bm_N", 100), step=10, key="bm_N")

        
        S0 = st.session_state.get("S0", 100.0)
        r = st.session_state.get("r", 0.03)
        mu = st.session_state.get("mu", 0.05)
        sigma = st.session_state.get("sigma", 0.2)
        T = st.session_state.get("T", 1.0)
        v0 = st.session_state.get("v0", 0.04)
        kappa = st.session_state.get("kappa", 2.0)
        theta = st.session_state.get("theta", 0.04)
        sigma_v = st.session_state.get("sigma_v", 0.3)
        rho = st.session_state.get("rho", -0.7)

        if st.button("Run Performance Test"):
            start = time.perf_counter()
            _ = simulate_gbm_paths(S0, T, mu, sigma, M=bm_M, N=bm_N)
            t_gbm = time.perf_counter() - start

            start = time.perf_counter()
            _ = simulate_heston_paths(S0, T, r, v0, kappa, theta, sigma_v, rho, M=bm_M, N=bm_N)
            t_heston = time.perf_counter() - start

            st.success(f"GBM (M={bm_M}, N={bm_N}) took {t_gbm:.3f}s | Heston took {t_heston:.3f}s")
            st.caption("Note: timings vary by hardware and current load; Heston is typically several× slower.")

        help_block(
            "Help — What affects runtime?",
            (
                "- Complexity grows roughly O(M·N).\n"
                "- Heston costs more per step (variance dynamics + correlation).\n"
                "- Use fewer steps for quick exploration; increase M last for stability.\n"
            ),
            presets=[
                {"label": "Fast demo: M=100, N=100", "on_click": lambda: set_state(bm_M=100, bm_N=100)},
                {"label": "Stress: M=2000, N=2000", "on_click": lambda: set_state(bm_M=2000, bm_N=2000)},
            ]
        )

    with right:
        st.subheader("Reference (Precomputed)")
        with st.expander("Show expected scaling remarks & chart"):
            st.markdown(
"This plot shows how runtime grows for GBM and Heston as the number of steps (N) increases. It’s a quick way to get a sense of how long bigger simulations might take."
            )
            
            show_image_safe(
                "benchmark_sde_timing.png",
                caption="GBM vs Heston timing vs time steps (if file present)."
            )
            

# ---------- Heston calibration diagnostics (plots) ----------
def _heston_calib_diagnostics(sub_iv: pd.DataFrame,
                              iv_col: str,
                              S0: float,
                              heston_points: Optional[List[Tuple[float, float]]],
                              heston_params: Optional[Dict],
                              sel_date,
                              sel_expiry):
    import matplotlib.pyplot as _plt
    import numpy as _np
    import pandas as _pd
    import streamlit as _st

    if not heston_points or heston_params is None:
        _st.info("No Heston diagnostics to show (missing params or points).")
        return

    # Build aligned dataframe of market vs model IVs for same strikes
    mkt = (sub_iv
           .loc[sub_iv[iv_col].notna() & _np.isfinite(sub_iv[iv_col]), ["STRIKE", iv_col, "UNDERLYING_LAST"]]
           .groupby("STRIKE")[iv_col].median()
           .rename("iv_mkt")
           .reset_index())
    mdl = _pd.DataFrame(heston_points, columns=["STRIKE", "iv_heston"])
    df_res = mkt.merge(mdl, on="STRIKE", how="inner")
    if df_res.empty:
        _st.info("Could not align market and model IVs on strike.")
        return

    df_res["moneyness"] = df_res["STRIKE"] / float(S0)
    df_res["residual"]  = df_res["iv_mkt"] - df_res["iv_heston"]

    # Layout
    _st.markdown("### Heston Calibration — Diagnostics")
    t1, t2, t3, t4 = _st.tabs(["Residuals vs Strike", "Residuals vs Moneyness", "Residuals Histogram", "Parameter Panel"])

    with t1:
        fig, ax = _plt.subplots()
        ax.axhline(0.0, ls="--", alpha=0.7)
        ax.scatter(df_res["STRIKE"], df_res["residual"], s=18)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Market IV − Heston IV")
        ax.set_title(f"Residuals vs Strike\n{pd.Timestamp(sel_date).date()} → {pd.Timestamp(sel_expiry).date()}")
        ax.grid(True)
        _st.pyplot(fig)

    with t2:
        fig, ax = _plt.subplots()
        ax.axhline(0.0, ls="--", alpha=0.7)
        ax.scatter(df_res["moneyness"], df_res["residual"], s=18)
        ax.set_xlabel("Moneyness  K / S₀")
        ax.set_ylabel("Market IV − Heston IV")
        ax.set_title("Residuals vs Moneyness (K/S₀)")
        ax.grid(True)
        _st.pyplot(fig)
        _st.caption("Pattern tips: U-shape residuals → curvature mismatch; tilted line → skew mismatch; clustering by K/S₀ → local misfit.")

    with t3:
        fig, ax = _plt.subplots()
        ax.hist(df_res["residual"], bins=24, density=True, alpha=0.85)
        ax.set_xlabel("Residual (IV units)")
        ax.set_ylabel("Density")
        ax.set_title("Residuals Histogram")
        ax.grid(True)
        _st.pyplot(fig)

        mae = float(_np.mean(_np.abs(df_res["residual"])))
        rmse = float(_np.sqrt(_np.mean(df_res["residual"]**2)))
        bias = float(_np.mean(df_res["residual"]))
        _st.markdown(
            f"**Diagnostics:** MAE = `{mae:.4f}` | RMSE = `{rmse:.4f}` | Bias = `{bias:.4f}` (positive = Heston under-fits IVs)"
        )

    with t4:
        # Params bar + Feller check
        kappa = float(heston_params.get("kappa", _np.nan))
        theta = float(heston_params.get("theta", _np.nan))
        sigma = float(heston_params.get("sigma", _np.nan))
        rho   = float(heston_params.get("rho", _np.nan))
        v0    = float(heston_params.get("v0", _np.nan))

        c1, c2 = _st.columns([1.1, 0.9])
        with c1:
            fig, ax = _plt.subplots()
            names = [r"$\kappa$", r"$\theta$", r"$\sigma_v$", r"$\rho$", r"$v_0$"]
            vals  = [kappa, theta, sigma, rho, v0]
            ax.bar(names, vals)
            ax.set_title("Calibrated Parameters")
            ax.grid(True, axis="y", alpha=0.3)
            _st.pyplot(fig)

        with c2:
            feller_lhs = 2.0 * kappa * theta
            feller_rhs = sigma**2
            ok = feller_lhs >= feller_rhs
            _st.markdown("**Feller condition**  \n"
                         r"$2\kappa\theta \;\ge\; \sigma_v^2$ "
                         f"→ **{'satisfied ✅' if ok else 'violated ⚠️'}**")
            _st.write(f"Computed: 2κθ = `{feller_lhs:.6f}` vs σᵥ² = `{feller_rhs:.6f}`")
            _st.caption("If violated, variance can hit 0; pricing is still well-defined but simulation may need care (e.g., QE or full-truncation).")

        _st.markdown(
            "- **ρ (leverage):** more negative → stronger downside skew.\n"
            "- **σᵥ (vol-of-vol):** bigger → fatter tails / stronger smile curvature.\n"
            "- **κ, θ:** control mean-reversion level and speed of variance.\n"
            "- **v₀:** ATM level; a mismatch vs θ often shows up as residuals near K≈S₀."
        )

# ==================================================
# ==============  PAGE 3: VOL SMILE  ===============
# ==================================================
def page_vol_smile():
    st.title("Volatility Smile Explorer")
    st.caption("This tab uses **real AAPL options data** to compare observed market implied volatility with two models: Black–Scholes (flat σ at-the-money) and the Heston stochastic-volatility model. Heston parameters are **calibrated to the selected trade date and expiry** using the market IVs shown below (QuantLib, if available).")

    # Sidebar 
    st.sidebar.subheader("📁 Data")

    # Ensure ./data exists
    DEFAULT_DATA_DIR.mkdir(exist_ok=True)

    # List local data files in ./data (both .csv and .csv.gz)
    local_files = sorted([p.name for p in DEFAULT_DATA_DIR.glob("*.csv*")])

    if not local_files:
        st.sidebar.info("Add a small dataset to the app’s ./data/ folder (e.g., aapl_2020_aug_subset.csv.gz).")

    chosen_local = st.sidebar.selectbox(
        "Choose a local data file (./data)",
        options=["(none)"] + local_files,
        index=(["(none)"] + local_files).index(DEFAULT_DATA_FILE) if DEFAULT_DATA_FILE in local_files else 0,
    )

    st.sidebar.caption("Or enter a full path or a URL (optional):")
    manual_path_or_url = st.sidebar.text_input(
        "Full path / URL",
        value="",
        placeholder="https://... or /full/path/to/file.csv(.gz)"
    )

    # Fallback text inputs (kept for flexibility)
    dataset_dir = st.sidebar.text_input(
        "Dataset folder (used only if no file above)",
        value=str(DEFAULT_DATA_DIR),
        key="dataset_dir"
    )
    filename = st.sidebar.text_input(
        "CSV filename (used only if no file above)",
        value=st.session_state.get("csv_filename", DEFAULT_DATA_FILE if DEFAULT_DATA_FILE in local_files else ""),
        key="csv_filename"
    )

    
    if manual_path_or_url.strip():
        csv_path = manual_path_or_url.strip()
    elif chosen_local != "(none)":
        csv_path = str(DEFAULT_DATA_DIR / chosen_local)
    else:
        csv_path = os.path.join(dataset_dir, filename or "")

    st.sidebar.subheader("🎛️ Filters")
    side = st.sidebar.selectbox("Side", ["both", "C", "P"], index=["both", "C", "P"].index(st.session_state.get("vs_side", "both")), key="vs_side")
    r = st.sidebar.number_input("Risk-free rate r", value=float(st.session_state.get("vs_r", 0.02)), step=0.005, format="%.4f", key="vs_r")
    q = st.sidebar.number_input("Dividend yield q", value=float(st.session_state.get("vs_q", 0.00)), step=0.005, format="%.4f", key="vs_q")
    run_heston = st.sidebar.checkbox("Include Heston fit (QuantLib)", value=st.session_state.get("vs_heston", True), key="vs_heston")

    # Load
    try:
        df = load_clean(csv_path)
    except Exception as e:
        st.error(f"Failed to load CSV from '{csv_path}': {e}")
        st.stop()

    # Validate required columns
    if "QUOTE_DATE" not in df.columns or "EXPIRE_DATE" not in df.columns:
        st.error("Required columns QUOTE_DATE and EXPIRE_DATE are missing.")
        st.stop()

    # Build date/expiry choices
    qd = pd.to_datetime(df["QUOTE_DATE"], errors="coerce").dt.normalize()
    ed = pd.to_datetime(df["EXPIRE_DATE"], errors="coerce").dt.normalize()

    date_options = [pd.Timestamp(x) for x in sorted(qd.dropna().unique().tolist())]
    if not date_options:
        st.error("No valid QUOTE_DATE values in the file.")
        st.stop()

    date_label = lambda d: pd.Timestamp(d).date().isoformat()
    sel_date = st.sidebar.selectbox("Trade date", date_options, format_func=date_label, index=0, key="vs_date")

    expiries_for_day = [pd.Timestamp(x) for x in sorted(ed[qd == sel_date].dropna().unique().tolist())]
    if not expiries_for_day:
        st.warning("No expiries found for that date; choose a different trade date.")
        st.stop()

    expiry_label = lambda d: pd.Timestamp(d).date().isoformat()
    sel_expiry = st.sidebar.selectbox("Expiry date", expiries_for_day, format_func=expiry_label, index=0, key="vs_expiry")

    # Layout: three columns
    col1, col2, col3 = st.columns([1.1, 1.1, 0.9])

    # ---- Plot 1: Market Smile ----
    with col1:
        st.subheader("Market Volatility Smile")
        sub = _subset(df, sel_date, sel_expiry)
        if sub.empty:
            st.warning("No rows for that date/expiry.")
        elif "STRIKE" not in sub.columns:
            st.error("STRIKE column missing.")
        else:
            fig1, ax1 = plt.subplots()
            plotted = False
            if side in ("C", "both") and "C_IV" in sub.columns:
                ax1.scatter(sub["STRIKE"], sub["C_IV"], s=12, label="Call IV")
                plotted = True
            if side in ("P", "both") and "P_IV" in sub.columns:
                ax1.scatter(sub["STRIKE"], sub["P_IV"], s=12, marker="x", label="Put IV")
                plotted = True
            if plotted:
                ax1.set_xlabel("Strike")
                ax1.set_ylabel("Implied Volatility")
                label_side = {"both": "Call/Put", "C": "Call", "P": "Put"}[side]
                ax1.set_title(f"{label_side} Smile on {pd.Timestamp(sel_date).date()} expiring {pd.Timestamp(sel_expiry).date()}")
                ax1.legend()
                fig1.tight_layout()
                st.pyplot(fig1)

                help_block(
                    "Help — How to read the Market Smile",
                    (
                        "- **Real AAPL data**: each point is a market IV at a strike for the chosen date & expiry.\n"
                        "- A U-shape indicates smile/skew. Black–Scholes would be flat.\n"
                        "- Compare calls vs puts (markers) to spot skew.\n"
                ))
            else:
                st.info("Nothing to plot (missing C_IV/P_IV columns).")

    # ---- Plot 2: Overlay ----
    with col2:
        st.subheader("Overlay: Market vs BS vs Heston")
        if sub.empty or "STRIKE" not in sub.columns or "UNDERLYING_LAST" not in sub.columns:
            st.info("Not enough data to build overlay plot.")
        else:
            iv_col = "C_IV" if side in ("C", "both") else "P_IV"
            if iv_col not in sub.columns:
                st.info(f"Column {iv_col} not found.")
            else:
                sub_iv = sub.loc[sub[iv_col].notna() & np.isfinite(sub[iv_col])]
                if sub_iv.empty:
                    st.info("No finite market IVs to plot.")
                else:
                    S0 = float(np.nanmedian(sub_iv["UNDERLYING_LAST"]))
                    # BS flat-σ at ATM
                    try:
                        sigma_bs = _atm_iv(sub_iv, S0)
                    except Exception as e:
                        st.warning(f"ATM IV estimate failed: {e}")
                        sigma_bs = float(np.nanmedian(sub_iv[iv_col]))

                    x = sub_iv["STRIKE"].values
                    y = sub_iv[iv_col].values

                    # Optional Heston (QuantLib)
                    heston_points = None
                    heston_params = None
                    heston_rmse = np.nan

                    if run_heston:
                        with st.spinner("Calibrating Heston (QuantLib)…"):
                            try:
                                import QuantLib as ql
                                calendar = ql.TARGET()
                                dc = ql.Actual365Fixed()
                                d0 = pd.Timestamp(sel_date)
                                e0 = pd.Timestamp(sel_expiry)
                                todays_date = ql.Date(d0.day, d0.month, d0.year)
                                exercise_date = ql.Date(e0.day, e0.month, e0.year)
                                ql.Settings.instance().evaluationDate = todays_date

                                spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
                                r_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, float(r), dc))
                                q_ts = ql.YieldTermStructureHandle(ql.FlatForward(todays_date, float(q), dc))

                                days_to_expiry = exercise_date - todays_date
                                maturity = ql.Period(int(days_to_expiry), ql.Days)

                                helpers = []
                                for _, row in sub_iv.iterrows():
                                    K = float(row["STRIKE"])
                                    vol = float(row[iv_col])
                                    if not np.isfinite(vol) or vol <= 0:
                                        continue
                                    quote = ql.QuoteHandle(ql.SimpleQuote(vol))
                                    helpers.append(ql.HestonModelHelper(maturity, calendar, float(S0), K, quote, r_ts, q_ts))
                                if helpers:
                                    v0    = max(sigma_bs**2, 1e-6)
                                    kappa = 1.5; theta = v0; sigma = 0.5; rho = -0.5
                                    process = ql.HestonProcess(r_ts, q_ts, spot_handle, v0, kappa, theta, sigma, rho)
                                    model   = ql.HestonModel(process)
                                    engine_for_helpers = ql.AnalyticHestonEngine(model)
                                    for h in helpers:
                                        h.setPricingEngine(engine_for_helpers)
                                    om = ql.LevenbergMarquardt()
                                    model.calibrate(helpers, om, ql.EndCriteria(400, 50, 1e-8, 1e-8, 1e-8))

                                    # Invert to BS IVs on same strikes
                                    init_vol = ql.BlackVolTermStructureHandle(
                                        ql.BlackConstantVol(todays_date, calendar, max(sigma_bs, 1e-4), dc)
                                    )
                                    bs_process = ql.BlackScholesMertonProcess(spot_handle, q_ts, r_ts, init_vol)
                                    opt_engine = ql.AnalyticHestonEngine(model)
                                    opt_type = ql.Option.Call if side in ("C", "both") else ql.Option.Put

                                    pts = []
                                    for _, row in sub_iv.iterrows():
                                        K = float(row["STRIKE"])
                                        payoff   = ql.PlainVanillaPayoff(opt_type, K)
                                        exercise = ql.EuropeanExercise(exercise_date)
                                        opt = ql.VanillaOption(payoff, exercise)
                                        opt.setPricingEngine(opt_engine)
                                        price = opt.NPV()
                                        try:
                                            iv = opt.impliedVolatility(price, bs_process, 1e-7, 500, 1e-6, 4.0)
                                        except RuntimeError:
                                            iv = np.nan
                                        if np.isfinite(iv) and iv > 0:
                                            pts.append((K, float(iv)))
                                    if pts:
                                        heston_points = sorted(pts, key=lambda p: p[0])

                                    try:
                                        kappa_cal, theta_cal, sigma_cal, rho_cal, v0_cal = [float(x) for x in model.params()]
                                        heston_params = dict(
                                            v0=v0_cal, kappa=kappa_cal, theta=theta_cal, sigma=sigma_cal, rho=rho_cal
                                        )
                                        if heston_points:
                                            mkt_by_k = sub_iv.groupby("STRIKE")[iv_col].median().to_dict()
                                            market_ivs = []
                                            model_ivs = []
                                            for K, iv_model in heston_points:
                                                if K in mkt_by_k and np.isfinite(mkt_by_k[K]) and mkt_by_k[K] > 0:
                                                    market_ivs.append(float(mkt_by_k[K]))
                                                    model_ivs.append(float(iv_model))
                                            if market_ivs:
                                                market_ivs = np.array(market_ivs)
                                                model_ivs = np.array(model_ivs)
                                                heston_rmse = float(np.sqrt(np.mean((market_ivs - model_ivs) ** 2)))
                                    except Exception as e:
                                        st.info(f"Heston calibration insights unavailable: {e}")

                            except ImportError:
                                st.info("QuantLib not installed. Skipping Heston. (pip install QuantLib-Python)")
                            except Exception as e:
                                st.info(f"Heston calibration skipped: {e}")

                    # Plot overlay
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(x, y, s=12, label=f"Market {'Call' if side in ('C','both') else 'Put'} IV")
                    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                    ax2.plot(x_line, np.full_like(x_line, sigma_bs), linestyle="--", label=f"BS fit (σ_ATM≈{sigma_bs:.3f})")
                    if heston_points:
                        xh = np.array([p[0] for p in heston_points])
                        yh = np.array([p[1] for p in heston_points])
                        ax2.plot(xh, yh, label="Heston fit")

                    rmse_str = f" | Heston RMSE={heston_rmse:.4f}" if np.isfinite(heston_rmse) else ""
                    ax2.set_xlabel("Strike"); ax2.set_ylabel("Implied Volatility")
                    ax2.set_title(
                        f"Market vs BS vs Heston{rmse_str}\n{pd.Timestamp(sel_date).date()} → {pd.Timestamp(sel_expiry).date()}"
                    )
                    ax2.legend()
                    fig2.tight_layout()
                    st.pyplot(fig2)

                    # diagnostics (tabs) under the overlay
                    _heston_calib_diagnostics(sub_iv, iv_col, S0, heston_points, heston_params, sel_date, sel_expiry)

            help_body = (
                "**What you see**\n\n"
                "• Blue dots: market IVs.\n\n"
            
                "• Dashed line: flat BS σ at ATM.\n\n"

                "• Heston curve (if enabled): model IVs from calibrated parameters.\n\n\n"

                "**Practical notes**\n\n"
                "• Use risk-free r and dividend q to reflect the trade date.\n\n"
                "• RMSE is in absolute IV units (e.g., 0.02 = 2 vol points).\n"
            )
            presets = [
                {"label": "Preset: Calls, r=2%, q=0%", "on_click": lambda: set_state(vs_side="C", vs_r=0.02, vs_q=0.00)},
                {"label": "Preset: Puts, r=0.5%, q=0.5%", "on_click": lambda: set_state(vs_side="P", vs_r=0.005, vs_q=0.005)},
                {"label": "Preset: Both sides, r=3%, q=1%", "on_click": lambda: set_state(vs_side="both", vs_r=0.03, vs_q=0.01)},
            ]
            help_block("Help — Interpreting Overlay & Calibration", help_body, presets)

            if 'heston_params' in locals() and heston_params is not None:
                st.markdown("### Heston Calibration Insights / Params")
                st.write(f"**v₀** (Initial variance): {heston_params['v0']:.6f}")
                st.write(f"**κ** (Mean reversion speed): {heston_params['kappa']:.6f}")
                st.write(f"**θ** (Long-run variance): {heston_params['theta']:.6f}")
                st.write(f"**σ** (Vol-of-vol): {heston_params['sigma']:.6f}")
                st.write(f"**ρ** (Spot/Vol correlation): {heston_params['rho']:.6f}")
                if np.isfinite(heston_rmse):
                    st.write(f"**Calibration RMSE** (Market IV vs Heston IV): {heston_rmse:.6f}")
                else:
                    st.write("**Calibration RMSE**: n/a")
                st.caption(
                    f"Calibrated on {len(heston_points) if heston_points else 0} model points; RMSE is absolute IV units."
                )

    # ---- Help & Guided Example (right column) ----
    with col3:
        st.subheader("📘 Help & Guided Example")
        with st.expander("Show guide"):
            st.write("This guide walks through a minimal example so you can interpret the two plots.")
            example_date = pd.Timestamp("2019-01-02")
            example_expiry = pd.Timestamp("2019-01-04")
            st.markdown(
                f"**Example setup**  \n"
                f"• Trade date = {example_date.date()}  \n"
                f"• Expiry date = {example_expiry.date()}  \n"
                f"• Side = Call (C)  \n"
                f"• r = 0.02, q = 0.00"
            )
            example_sub = _subset(df, example_date, example_expiry)
            if example_sub.empty:
                st.info("Example data not found in this dataset — pick any visible date/expiry on the left.")
            else:
                fig_ex1, ax_ex1 = plt.subplots()
                if "C_IV" in example_sub.columns:
                    ax_ex1.scatter(example_sub["STRIKE"], example_sub["C_IV"], s=12, label="Call IV")
                ax_ex1.set_xlabel("Strike"); ax_ex1.set_ylabel("Implied Volatility"); ax_ex1.legend()
                ax_ex1.set_title("Market Volatility Smile (Example)")
                st.pyplot(fig_ex1)
                st.markdown("**Reading tips:** The U-shape is the smile; BS assumes flat σ, so deviations show skew/smile.")

                S0 = float(np.nanmedian(example_sub["UNDERLYING_LAST"])) if "UNDERLYING_LAST" in example_sub.columns else np.nan
                try:
                    sigma_bs = _atm_iv(example_sub, S0)
                except Exception:
                    sigma_bs = float(np.nanmedian(example_sub.get("C_IV", pd.Series([np.nan]))))

                x = example_sub["STRIKE"].values
                y = example_sub.get("C_IV", pd.Series(np.full_like(x, np.nan))).values
                fig_ex2, ax_ex2 = plt.subplots()
                ax_ex2.scatter(x, y, s=12, label="Market Call IV")
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                if np.isfinite(sigma_bs):
                    ax_ex2.plot(x_line, np.full_like(x_line, sigma_bs), linestyle="--", label=f"BS fit (σ≈{sigma_bs:.3f})")
                ax_ex2.set_xlabel("Strike"); ax_ex2.set_ylabel("Implied Volatility"); ax_ex2.legend()
                ax_ex2.set_title("Market vs BS (Example)")
                st.pyplot(fig_ex2)

    st.caption("Tip: toggle the ‘Include Heston fit’ switch if QuantLib isn’t installed, or tweak r and q to see sensitivity.")


# ==================================================
# =================== SIDEBAR NAV ==================
# ==================================================
with st.sidebar:
    st.markdown("### Navigate")
    page = st.radio("",
                    NAV_LABELS,
                    key="nav_page",
                    label_visibility="collapsed")

# ==================================================
# ==================== ROUTER ======================
# ==================================================
if page.startswith("🏠"):
    page_home()
elif page.startswith("📚"):
    page_theory()
elif page.startswith("🧪"):
    page_sde_visualiser()
elif page.startswith("⚡"):
    page_performance()
else:
    page_vol_smile()
