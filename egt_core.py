"""
Model code for the pharma compliance paper.

The system is

    xdot = x(1-x) (a0 + ay*y + az*z)    ethical firms
    ydot = y(1-y) (b0 + bx*x)           active regulators
    zdot = z(1-z) (c0 + cx*x)           active public

with a0 = -C_base, ay = S_pol + alpha - k^3, az = gamma,
b0 = B_r - C_r, bx = -B_r, c0 = B_p - C_p, cx = -B_p.

The notebook and the paper both pull their numbers from here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import io

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize

try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # running inside a notebook
    _HERE = Path.cwd()

# Copies of the two data files, used if data/ isn't next to this file
# (e.g. when the notebook is uploaded to Colab by itself).
_EMBEDDED_INDICES = """Year,Evolutionary_t,Ethical_Firms_x,Regulators_y,Public_z,Friction_k
2010,0,0.65,0.20,0.15,0.60
2012,2,0.72,0.30,0.18,0.65
2014,4,0.78,0.45,0.25,0.70
2016,6,0.85,0.60,0.35,0.75
2018,8,0.80,0.75,0.50,0.80
2020,10,0.60,0.85,0.65,0.88
2022,12,0.40,0.92,0.80,0.95
2024,14,0.25,0.88,0.90,1.05
"""
_EMBEDDED_FIRMS = r"""firm,strategy_zone,rnd_pct_of_revenue,rnd_basis,rnd_source,total_penalties_musd,penalty_basis,penalty_source,revenue_2019_musd,revenue_basis,source_access_date
Teva,S2,6.0,"R&D 1,010 / revenues 16,887 (FY2019 Form 10-K)",https://www.sec.gov/Archives/edgar/data/818686/000119312520044221/d852939d10k.htm,2326,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,16887.0,revenues FY2019 (Form 10-K),2026-10-14
Abbott,S2,7.6,"R&D 2,440 / net sales 31,904 (FY2019 Form 10-K)",https://www.sec.gov/Archives/edgar/data/1800/000110465920023904/abt-20191231x10k59d41b.htm,2009,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,31904.0,net sales FY2019 (Form 10-K),2026-10-14
Mylan,S2,5.6,"R&D 639.9 / total revenues 11,500.5 (FY2019 Form 10-K)",https://www.sec.gov/Archives/edgar/data/0001623613/000162361320000011/myl10k20191231doc.htm,1210,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,11500.5,total revenues FY2019 (Form 10-K),2026-10-14
Indivior,S2,6.8,R&D 53 / net revenue 785 (FY2019 results),https://www.investegate.co.uk/announcement/rns/indivior--indv/final-results/6248224,1183,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,785.0,net revenue FY2019 (FY2019 results announcement),2026-10-14
Mallinckrodt,Mixed,11.0,"R&D 349.4 / net sales 3,162.5 = 11.0% (FY2019 Form 10-K)",https://www.sec.gov/Archives/edgar/data/0001567892/000156789220000005/mnk10-k122719.htm,1754,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,3162.5,net sales FY2019 (Form 10-K),2026-10-14
J&J,Mixed,13.8,"R&D 11,355 / sales 82,059 (FY2019 Form 10-K; R&D as summarized by Nasdaq, Aug 2020)",https://www.nasdaq.com/articles/top-5-spenders-in-rd-in-biotech-2020-08-12,8449,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,82059.0,sales FY2019 (Form 10-K),2026-10-14
GSK,Mixed,13.5,"R&D GBP 4,568M / turnover GBP 33,754M (FY2019 results announcement)",https://www.gsk.com/media/5820/fy-2019-results-announcement.pdf,7921,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,43920.0,turnover FY2019 in USD,2026-10-14
Pfizer,Mixed,16.7,"R&D 8,650 / revenues 51,750 (FY2019 Form 10-K, as reported; Nasdaq summary Aug 2020)",https://www.nasdaq.com/articles/top-5-spenders-in-rd-in-biotech-2020-08-12,4773,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,51750.0,revenues FY2019 as reported in the FY2019 Form 10-K,2026-10-14
Amgen,Mixed,17.6,"R&D 4,116 / total revenues 23,362 (FY2019 Form 10-K)",https://www.sec.gov/Archives/edgar/data/0000318154/000031815420000017/amgn-12312019x10kq42019.htm,939,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,23362.0,total revenues FY2019 (Form 10-K),2026-10-14
Allergan,Mixed,11.3,"R&D 1,812.0 / total net revenues 16,088.9 (FY2019 Form 10-K)",https://www.sec.gov/Archives/edgar/data/1578845/000156459020005038/agn-10k_20191231.htm,816,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,16088.9,total net revenues FY2019 (Form 10-K),2026-10-14
Novartis,Mixed,19.8,"R&D 9,402 / net sales 47,445 = 19.8% (FY2019 annual report; Nasdaq summary Aug 2020)",https://www.nasdaq.com/articles/top-5-spenders-in-rd-in-biotech-2020-08-12,2948,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,47445.0,"net sales FY2019, continuing operations",2026-10-14
Merck,S1,21.1,"R&D 9,872 / sales 46,840 = 21.07% (FY2019 Form 10-K; Nasdaq summary Aug 2020)",https://www.nasdaq.com/articles/top-5-spenders-in-rd-in-biotech-2020-08-12,1840,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,46840.0,sales FY2019 (Form 10-K),2026-10-14
Eli Lilly,S1,25.1,"R&D 5,595.0 / revenue 22,319.5 = 25.1% (2019 Financial Report)",https://investor.lilly.com/static-files/34d71960-241f-4160-bd20-86fb85df4def,1742,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,22319.5,revenue FY2019,2026-10-14
AstraZeneca,S1,24.8,"R&D 6,059 / total revenue 24,384 (FY2019 Form 20-F)",https://www.sec.gov/Archives/edgar/data/901832/000110465920028557/a20-5700_1ex15d1.htm,1035,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,24384.0,total revenue FY2019 (Form 20-F),2026-10-14
BMS,S1,23.5,"R&D 6,148 / total revenues 26,145 (FY2019 results, Form 8-K exhibit 99.2)",https://www.sec.gov/Archives/edgar/data/14272/000001427220000075/q42019ex992.htm,1823,Public Citizen (2024) Table 4: federal+state total 1991-2021,https://www.citizen.org/wp-content/uploads/2681.pdf,26145.0,total revenues FY2019,2026-10-14
"""


def _read(name, fallback):
    for c in (_HERE / "data" / name, _HERE / name):
        if c.exists():
            return pd.read_csv(c)
    return pd.read_csv(io.StringIO(fallback))


def load_indices():
    return _read("pharma_indices.csv", _EMBEDDED_INDICES)


def load_firms():
    return _read("firm_classification.csv", _EMBEDDED_FIRMS)


# calibration
# Order of the fitted constants everywhere below.
PARAM_NAMES = ["C_base", "S_pol", "alpha", "B_r", "C_r", "gamma", "B_p", "C_p"]
# Bounds: costs and penalties in (0, 1]; benefits at least the matching costs.
BOUNDS = [(0.1, 0.5), (0.1, 1.0), (0.1, 1.0), (0.5, 1.0),
          (0.1, 0.5), (0.1, 1.0), (0.4, 1.0), (0.1, 0.5)]
INITIAL_GUESS = [0.2, 0.4, 0.2, 0.8, 0.25, 0.6, 0.6, 0.15]
X_WEIGHT = 8.0          # extra weight on the compliance series
TIME_WEIGHTS = (1.0, 3.0)  # later years weighted more


def _sys_training(state, t, params, t_emp, k_emp):
    x, y, z = (min(1 - 1e-5, max(1e-5, v)) for v in state)
    C_b, S_p, alp, B_r, C_r, gam, B_p, C_p = params
    k_t = np.interp(t, t_emp, k_emp)  # friction follows the data between observations
    F = -(C_b + k_t**3 * y) + y * (S_p + alp) + z * gam
    return [x * (1 - x) * F,
            y * (1 - y) * ((1 - x) * B_r - C_r),
            z * (1 - z) * ((1 - x) * B_p - C_p)]


def _wmse(a, b, w=None):
    w = np.ones_like(a) if w is None else w
    return float(np.sum(w * (a - b) ** 2) / np.sum(w))


def _columns(df):
    return (df[c].values for c in
            ["Evolutionary_t", "Ethical_Firms_x", "Regulators_y", "Public_z", "Friction_k"])


def calibrate(df=None):
    """L-BFGS-B fit of the eight constants to the biennial indices."""
    df = load_indices() if df is None else df
    t, x, y, z, k = _columns(df)
    w = np.linspace(*TIME_WEIGHTS, len(t))

    def loss(p):
        try:
            sol = odeint(_sys_training, [x[0], y[0], z[0]], t, args=(p, t, k))
            return X_WEIGHT * _wmse(x, sol[:, 0], w) + _wmse(y, sol[:, 1]) + _wmse(z, sol[:, 2])
        except Exception:
            return 1e6

    res = minimize(loss, INITIAL_GUESS, method="L-BFGS-B", bounds=BOUNDS)
    return dict(zip(PARAM_NAMES, res.x)), res


@lru_cache(maxsize=1)
def fitted_params():
    p, _ = calibrate()
    return tuple(p[n] for n in PARAM_NAMES)


def P(**changes):
    """Fitted constants as a dict; pass e.g. S_pol=0.5 to change one."""
    d = dict(zip(PARAM_NAMES, fitted_params()))
    d.update(changes)
    return d


def calibrated_path(t_eval, df=None):
    df = load_indices() if df is None else df
    t, x, y, z, k = _columns(df)
    return odeint(_sys_training, [x[0], y[0], z[0]], t_eval, args=(fitted_params(), t, k))


# constant-k model
K_BASELINE = 0.60  # friction index in 2010


def k_mean_historical(df=None):
    df = load_indices() if df is None else df
    return float(df["Friction_k"].mean())  # 0.7975 (between k* and k**)


def coefficients(p, k):
    return dict(a0=-p["C_base"], ay=p["S_pol"] + p["alpha"] - k**3, az=p["gamma"],
                b0=p["B_r"] - p["C_r"], bx=-p["B_r"],
                c0=p["B_p"] - p["C_p"], cx=-p["B_p"])


def rhs(state, t, p, k):
    x, y, z = state
    c = coefficients(p, k)
    return [x * (1 - x) * (c["a0"] + c["ay"] * y + c["az"] * z),
            y * (1 - y) * (c["b0"] + c["bx"] * x),
            z * (1 - z) * (c["c0"] + c["cx"] * x)]


def jacobian(state, p, k):
    x, y, z = state
    c = coefficients(p, k)
    F = c["a0"] + c["ay"] * y + c["az"] * z
    G = c["b0"] + c["bx"] * x
    H = c["c0"] + c["cx"] * x
    return np.array([[(1 - 2 * x) * F, x * (1 - x) * c["ay"], x * (1 - x) * c["az"]],
                     [y * (1 - y) * c["bx"], (1 - 2 * y) * G, 0.0],
                     [z * (1 - z) * c["cx"], 0.0, (1 - 2 * z) * H]])


def _expit(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))  # logistic without overflow


def simulate(x0, t, k=K_BASELINE, p=None, **changes):
    """Integrate at constant k.

    We integrate u = ln(x/(1-x)) (same for y, z). The right-hand side is bounded in those
    variables, so the solver stays accurate near 0 and 1. A coordinate that starts exactly at
    0 or 1 stays there.
    """
    p = P(**changes) if p is None else {**p, **changes}
    c = coefficients(p, k)
    x0 = np.asarray(x0, float)
    pinned = (x0 <= 0.0) | (x0 >= 1.0)
    pinned_val = np.where(x0 >= 1.0, 1.0, 0.0)
    xc = np.clip(x0, 1e-300, 1 - 1e-16)
    u0 = np.where(pinned, 0.0, np.log(xc) - np.log1p(-xc))

    def f(u, _t):
        s = np.where(pinned, pinned_val, _expit(u))
        du = np.array([c["a0"] + c["ay"] * s[1] + c["az"] * s[2],
                       c["b0"] + c["bx"] * s[0],
                       c["c0"] + c["cx"] * s[0]])
        return np.where(pinned, 0.0, du)

    U = odeint(f, u0, t, rtol=1e-10, atol=1e-10, mxstep=100000)
    return np.where(pinned, pinned_val, _expit(U))


# Averaging window: starts after most of the z transient and covers 2+ cycles (periods 80-160).
LATE_WINDOW = (200.0, 500.0)


def long_run_mean_x(k=None, x0=(0.5, 0.5, 0.5), t_window=LATE_WINDOW, n=600, **changes):
    """Time-average of x over the late window."""
    k = k_mean_historical() if k is None else k
    t = np.concatenate([[0.0], np.linspace(*t_window, n)])
    X = simulate(x0, t, k=k, **changes)
    return float(np.clip(X[1:, 0], 0, 1).mean())


def trough_x(k=None, x0=(0.5, 0.5, 0.5), t_window=(200.0, 800.0), n=1200, **changes):
    """Lowest x reached over a late window (how deep the cycle dips)."""
    k = k_mean_historical() if k is None else k
    t = np.concatenate([[0.0], np.linspace(*t_window, n)])
    X = simulate(x0, t, k=k, **changes)
    return float(np.clip(X[1:, 0], 0, 1).min())


# analytic results
def nullcline_x(p):
    # x where ydot's bracket vanishes (faces z=0,1) and where zdot's does (faces y=0,1)
    return dict(x_z=1 - p["C_r"] / p["B_r"], x_y=1 - p["C_p"] / p["B_p"])


def thresholds(p=None):
    """k*: center on z=0 hits y=1.  k**: center on y=1 hits z=1 (collapse)."""
    p = P() if p is None else p
    return dict(k_star=(p["S_pol"] + p["alpha"] - p["C_base"]) ** (1 / 3),
                k_2star=(p["S_pol"] + p["alpha"] + p["gamma"] - p["C_base"]) ** (1 / 3))


def face_fixed_points(p, k):
    c = coefficients(p, k)
    xz, xy = -c["b0"] / c["bx"], -c["c0"] / c["cx"]
    ay = c["ay"] if c["ay"] != 0 else np.nan
    return {"y=0": (xy, 0.0, -c["a0"] / c["az"]),
            "y=1": (xy, 1.0, -(c["a0"] + c["ay"]) / c["az"]),
            "z=0": (xz, -c["a0"] / ay, 0.0),
            "z=1": (xz, -(c["a0"] + c["az"]) / ay, 1.0)}


def classify(eigs, tol=1e-10):
    re, im = np.real(eigs), np.imag(eigs)
    if np.any(np.abs(re) < tol) and np.any(np.abs(im) > tol):
        return "center in face (non-hyperbolic)"
    if np.any(np.abs(eigs) < tol):
        return "degenerate (zero eigenvalue)"
    if np.all(re < 0):
        return "sink"
    if np.all(re > 0):
        return "source"
    return "saddle"


def fixed_point_table(p=None, k=K_BASELINE):
    p = P() if p is None else p
    rows = []
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                ev = np.linalg.eigvals(jacobian((x, y, z), p, k))
                rows.append(dict(kind="corner", point=(x, y, z), in_cube=True,
                                 eigenvalues=np.round(ev, 4), type=classify(ev)))
    for face, pt in face_fixed_points(p, k).items():
        ok = np.all(np.isfinite(pt))
        inside = ok and all(-1e-12 <= v <= 1 + 1e-12 for v in pt)
        ev = np.linalg.eigvals(jacobian(pt, p, k)) if ok else np.full(3, np.nan)
        rows.append(dict(kind=f"face {face}", point=tuple(np.round(pt, 4)), in_cube=inside,
                         eigenvalues=np.round(ev, 4), type=classify(ev) if inside else "outside cube"))
    return pd.DataFrame(rows)


def conserved_H_z0(x, y, p, k):
    """ln H on the face z=0 (H from Appendix C); constant along every orbit there."""
    c = coefficients(p, k)
    return (c["b0"] * np.log(x) - (c["b0"] + c["bx"]) * np.log(1 - x)
            - c["a0"] * np.log(y) + (c["a0"] + c["ay"]) * np.log(1 - y))


def small_orbit_period(p, k):
    c = coefficients(p, k)
    lam2 = (c["a0"] + c["ay"]) * (c["b0"] + c["bx"]) * c["a0"] * c["b0"] / (c["ay"] * c["bx"])
    return 2 * np.pi / np.sqrt(-lam2) if lam2 < 0 else np.nan


def drift_V(p):
    # d/dt [ln(y/(1-y))/B_r - ln(z/(1-z))/B_p] = C_p/B_p - C_r/B_r, for every state
    return p["C_p"] / p["B_p"] - p["C_r"] / p["B_r"]


# Lyapunov exponents
def lyapunov_spectrum(x0=(0.5, 0.5, 0.5), k=K_BASELINE, T=4000.0, dt_orth=1.0,
                      p=None, return_history=False):
    """Benettin's method: evolve 3 tangent vectors, re-orthonormalize (QR) every
    dt_orth, average the log stretch factors.

    The trajectory is stepped in log-odds (see simulate); stepping in x, y, z lets y
    overshoot 1 near the y=1 face, which gives spurious positive exponents.
    With T=10 and dt_orth=0.01 this matches the values from chaos/chaos.py.
    """
    from scipy.integrate import solve_ivp
    p = P() if p is None else p
    c = coefficients(p, k)

    def aug(t, w):
        s = _expit(w[:3])
        Q = w[3:].reshape(3, 3)
        du = [c["a0"] + c["ay"] * s[1] + c["az"] * s[2],
              c["b0"] + c["bx"] * s[0],
              c["c0"] + c["cx"] * s[0]]
        return np.concatenate([du, (jacobian(s, p, k) @ Q).ravel()])

    x0 = np.clip(np.asarray(x0, float), 1e-12, 1 - 1e-12)
    u, Q, acc = np.log(x0) - np.log1p(-x0), np.eye(3), np.zeros(3)
    n = int(round(T / dt_orth))
    hist = []
    for i in range(n):
        sol = solve_ivp(aug, (0, dt_orth), np.concatenate([u, Q.ravel()]),
                        method="DOP853", rtol=1e-10, atol=1e-12)
        w = sol.y[:, -1]
        u = w[:3]
        Q, R = np.linalg.qr(w[3:].reshape(3, 3))
        sgn = np.sign(np.diag(R))
        Q, R = Q * sgn, (R.T * sgn).T
        acc += np.log(np.abs(np.diag(R)))
        if return_history:
            hist.append(np.sort(acc / ((i + 1) * dt_orth))[::-1])
    le = np.sort(acc / (n * dt_orth))[::-1]
    return (le, np.array(hist)) if return_history else le
