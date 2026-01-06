#!/usr/bin/env python3
"""
estimators/iterative.py

Iterative trace-only estimator for step-level execution coefficients (betas)
for a vLLM-style inference engine.

Implements the paper section:
  "Step-Density Reweighted Estimation" + "MM-Style Iterative NNLS"

Design goals
------------
- *Faithful to the paper* (iteration semantics, lagged effective prefill pressure,
  and step-density–reweighted features).
- *Maximal reuse* of the baseline implementation in estimators/baseline.py.
- Trace-only: uses only phase start/end times and phase token totals.
  Never uses hidden step boundaries or per-step timings.

High-level algorithm (paper-consistent)
--------------------------------------
Given a trace of phase instances (one row per phase instance i, either prefill or decode):

  0) Initialize beta^(0) using the baseline NNLS estimator (Eq. (baseline_nnls)).
     Initialization correction mode is configurable (end|uniform|none).

  1) Construct the trace-induced time grid from phase boundaries.
     Reconstruct instantaneous pressures on the grid:
       - p_pf_full(t) = C * (# active prefill phases)     [tokens/step]
       - p_dec(t)     =     (# active decode phases)      [tokens/step]

  2) Iterate m = 1,2,... (outer MM iterations):
       Given (beta^(m-1), tilde_p_pf^(m-1)(t)):

       (a) Local step-time model:
             Delta^(m)(t) = beta0^(m-1) + beta1^(m-1)*tilde_p_pf^(m-1)(t) + beta2^(m-1)*p_dec(t)
           Step density:
             lambda^(m)(t) = 1 / Delta^(m)(t)

       (b) Prefill partial-chunk correction (mode-dependent):
           - end/uniform/none: trace-only correction signal c(t) (tokens/sec)
           - beta_informed: compute c^(m)(t) using lambda^(m) and the request-level
             last-step window W_r^(m) (Eq. (qr_def_m)–(last_step_window_m))

           Effective prefill pressure:
             tilde_p_pf^(m)(t) = p_pf_full(t) - c^(m)(t)/lambda^(m)(t)   [tokens/step]

       (c) Phase attribution q_i^(m)(t) proportional to lambda^(m) over [t_{i,s}, t_{i,e}):
             q_i^(m)(t) = lambda^(m)(t) / ∫ lambda^(m) over the phase

       (d) Step-averaged pressures (Eq. (step_averaged_pressures_m)):
             bar_p_pf_i^(m)  = ∫ tilde_p_pf^(m)(t) q_i^(m)(t) dt
             bar_p_dec_i^(m) = ∫ p_dec(t)          q_i^(m)(t) dt

           Step-weighted exposures used in NNLS:
             X_i = [ N_i,  N_i*bar_p_pf_i^(m),  N_i*bar_p_dec_i^(m) ]

       (e) Frozen-feature NNLS subproblem (Eq. (sr_nnls_m)):
             beta_fit = argmin_{beta>=0} Σ_i ( beta0*N_i + beta1*N_i*bar_p_pf_i^(m) +
                                              beta2*N_i*bar_p_dec_i^(m) - T_i )^2

           Optional damping:
             beta^(m) = (1-eta)*beta^(m-1) + eta*beta_fit

       Stop when ||beta^(m) - beta^(m-1)||_2 < tol, or max_outer_iters reached.

IMPORTANT iteration semantics (must match paper)
-----------------------------------------------
At the start of iteration m, beta^(m-1) and tilde_p_pf^(m-1)(t) are available.
All superscript-(m) quantities are computed during iteration m from beta^(m-1)
(and trace-derived signals), then held fixed while solving for beta^(m).

Dependencies
------------
pip install numpy pandas scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Any

import numpy as np
import pandas as pd
import glob
import os
import sys
import json
import yaml
from sklearn.linear_model import LinearRegression

# Reuse as much as possible from baseline.
# These helpers implement schema validation, grid construction, overlap pressures,
# and trace-only end/uniform/none correction signals in baseline's integrated form.
from estimators.baseline import (  # noqa: F401
    BaselineBetaResult,
    PartialChunkCorrectionMode,
    _build_global_time_grid_from_df,
    _build_partial_chunk_correction_rate,
    _build_time_grid,
    _check_baseline_prefill_uniqueness,
    _compute_durations_and_lambda_hat,
    _infer_or_read_step_counts,
    _integral_on_interval,
    _prefix_integral,
    _reconstruct_pressures,
    _require_columns,
    _validate_and_normalize_schema,
    estimate_betas_baseline,
)

from blis_alpha_model import train_alpha_model


# =============================================================================
# Result structure
# =============================================================================
PrefillCorrectionIterativeMode = Literal["uniform", "end", "none", "beta_informed"]


@dataclass(frozen=True)
class IterativeBetaResult:
    """
    Output of iterative step-density reweighted beta estimation.

    Attributes
    ----------
    beta0 : float
        Per-step fixed overhead β0 in seconds/step.
    beta1 : float
        Prefill token cost β1 in seconds/token.
    beta2 : float
        Decode token cost β2 in seconds/token.

    converged : bool
        Whether the outer MM loop met the convergence criterion.

    n_outer_iters : int
        Number of outer iterations actually executed (>= 0).

    diagnostics : Dict[str, float]
        Convenience metrics (final fit R^2 / RMSE, feature scales, etc.)

    history : Dict[str, object]
        Optional iteration history for debugging:
          - "beta": list of (beta0,beta1,beta2) per outer iteration, including beta^(0)
          - "delta_beta_l2": list of ||beta^m - beta^{m-1}||_2 for m>=1
          - "rmse_seconds": list of RMSE per iteration (frozen-feature fit)
          - "notes": list[str] with lightweight iteration notes

    Notes on units
    --------------
    The frozen-feature regression model at iteration m is:
        T_i ≈ β0*N_i + β1*(N_i*bar_p_pf_i^(m)) + β2*(N_i*bar_p_dec_i^(m))
    where:
        T_i                    : seconds
        N_i                    : steps
        bar_p_*_i^(m)          : tokens/step
        N_i*bar_p_*_i^(m)      : tokens
    so each term has units of seconds.
    """

    beta0: float
    beta1: float
    beta2: float
    converged: bool
    n_outer_iters: int
    diagnostics: Dict[str, float]
    history: Dict[str, Any]


# =============================================================================
# Small helpers (iterative-specific but kept tiny and well-commented)
# =============================================================================
def _fit_nnls_from_features(
    *,
    N_i: np.ndarray,
    exp_pf_tokens: np.ndarray,
    exp_dec_tokens: np.ndarray,
    T_i: np.ndarray,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Solve the frozen-feature NNLS subproblem for beta.

    Paper mapping
    -------------
    This matches Eq. (sr_nnls_m) with the design matrix:
        X[:,0] = N_i
        X[:,1] = N_i * bar_p_pf_i^(m)   (tokens)
        X[:,2] = N_i * bar_p_dec_i^(m)  (tokens)
    and response y = T_i (seconds).

    Returns beta_fit and basic fit diagnostics (R^2, RMSE).
    """
    X = np.column_stack([N_i, exp_pf_tokens, exp_dec_tokens]).astype(np.float64, copy=False)
    y = T_i.astype(np.float64, copy=False)

    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)
    beta0, beta1, beta2 = map(float, model.coef_)

    # Diagnostics (lightweight)
    y_hat = model.predict(X)
    resid = y_hat - y
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y))) if len(y) else float("nan")

    diag = {"r2": r2, "rmse_seconds": rmse}
    return beta0, beta1, beta2, diag


def _prefill_request_table(
    df: pd.DataFrame,
    *,
    request_id_col: str,
    phase_type_col: str,
    t_start_col: str,
    t_end_col: str,
    prefill_tokens_col: str,
    chunk_size: int,
    n_steps_col: str,
) -> pd.DataFrame:
    """
    Build a per-request prefill table used by correction schemes.

    Assumptions (consistent with baseline)
    --------------------------------------
    This implementation assumes at most one prefill row per request id.
    That is enforced via _check_baseline_prefill_uniqueness().

    Returns a prefill-only DataFrame with:
      - request_id
      - t_{r,s}, t_{r,e}
      - N_r (steps)
      - P_r (tokens)
      - mu_r (missing token mass of final prefill chunk), in [0, C]

    mu_r matches paper:
      rho_r = P_r - C*(N_r-1)
      mu_r  = C - rho_r
    """
    prefill_df = _check_baseline_prefill_uniqueness(df, request_id_col=request_id_col, phase_type_col=phase_type_col).copy()
    if prefill_df.empty:
        return prefill_df

    _require_columns(prefill_df, [prefill_tokens_col, n_steps_col, t_start_col, t_end_col, request_id_col])

    P = prefill_df[prefill_tokens_col].astype(float).to_numpy()
    N = prefill_df[n_steps_col].astype(float).to_numpy()
    C = float(chunk_size)

    rho = P - C * (N - 1.0)
    rho = np.clip(rho, 0.0, C)
    mu = np.clip(C - rho, 0.0, C)

    prefill_df["P_r"] = P
    prefill_df["N_r"] = N
    prefill_df["mu_r"] = mu
    prefill_df["t_rs"] = prefill_df[t_start_col].astype(float)
    prefill_df["t_re"] = prefill_df[t_end_col].astype(float)
    return prefill_df


def _lambda_segment_from_beta_and_pressures(
    *,
    beta_prev: np.ndarray,            # shape (3,)
    tilde_p_pf_prev: np.ndarray,      # shape (S,), tokens/step
    p_dec: np.ndarray,                # shape (S,), tokens/step
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Delta^(m)(t) and lambda^(m)(t) on grid segments from beta^(m-1).

    Paper:
      Delta^(m)(t) = beta0^(m-1) + beta1^(m-1)*tilde_p_pf^(m-1)(t) + beta2^(m-1)*p_dec(t)
      lambda^(m)(t) = 1 / Delta^(m)(t)

    Returns arrays over segments:
      - Delta_seg : sec/step
      - lambda_seg: steps/sec
    """
    b0, b1, b2 = map(float, beta_prev)
    Delta = b0 + b1 * tilde_p_pf_prev + b2 * p_dec
    Delta = np.maximum(Delta, eps)  # enforce positivity for numerical stability
    lam = 1.0 / Delta
    return Delta, lam


def _beta_informed_correction_rate_on_grid(
    *,
    grid: np.ndarray,              # shape (M,)
    lambda_seg: np.ndarray,        # shape (S=M-1,), steps/sec (piecewise constant)
    prefill_req: pd.DataFrame,     # per-request prefill table
) -> np.ndarray:
    """
    Construct the beta-informed correction signal c^(m)(t) (tokens/sec) on grid segments.

    Faithful construction (paper)
    -----------------------------
    For each request r:

      q_r^(m)(t) = lambda^(m)(t) / ∫_{t_rs}^{t_re} lambda^(m)(u) du
      F_r^(m)(t) = ∫_{t_rs}^{t} q_r^(m)(u) du
      W_r^(m)    = { t : F_r^(m)(t) >= 1 - 1/N_r }   (last-step window)
      c_r^(m)(t) = mu_r * q_r^(m)(t) 1{t in W_r} / ∫_{W_r} q_r^(m)(u) du

    Implementation details (trace-grid)
    -----------------------------------
    We represent c^(m)(t) as piecewise-constant over grid segments by distributing each
    request's missing mass mu_r across segments overlapping W_r, proportional to lambda
    within those overlaps, and then converting segment-mass to a per-segment rate:

      mass_{r,j} = mu_r * (lambda_j * overlap_len_{r,j}) / ∫_{W_r} lambda(u) du
      corr_rate[j] += mass_{r,j} / seg_len_j

    This preserves the required invariants:
      - c^(m)(t) has units tokens/sec
      - ∫ c_r^(m)(t) dt = mu_r exactly (up to floating rounding)

    Returns
    -------
    corr_rate : np.ndarray, shape (S=M-1,), tokens/sec
    """
    S = grid.size - 1
    corr_rate = np.zeros(S, dtype=np.float64)

    if prefill_req.empty:
        return corr_rate

    dt_seg = grid[1:] - grid[:-1]
    if np.any(dt_seg <= 0):
        raise ValueError("Non-positive grid segment durations encountered.")

    # Prefix integral of lambda over time: I_lambda[k] = ∫_{grid[0]}^{grid[k]} lambda(t) dt (units: steps)
    I_lambda = _prefix_integral(grid, lambda_seg)

    # Helper: integral of lambda over [a,b)
    def int_lambda(a: float, b: float) -> float:
        return float(_integral_on_interval(grid, I_lambda, a, b, lambda_seg))

    for _, row in prefill_req.iterrows():
        t_rs = float(row["t_rs"])
        t_re = float(row["t_re"])
        N_r = float(row["N_r"])
        mu_r = float(row["mu_r"])

        if mu_r <= 0.0:
            continue
        if t_re <= t_rs:
            continue
        if N_r <= 0.0:
            continue

        denom = int_lambda(t_rs, t_re)  # steps
        if denom <= 0.0:
            continue

        # Threshold for last-step window:
        # F(t) >= 1 - 1/N  <=>  ∫_{t_rs}^{t} lambda >= denom*(1-1/N) = denom - denom/N
        target = denom - denom / N_r
        target = max(0.0, min(target, denom))

        # Find t_star: smallest t in [t_rs,t_re] such that ∫_{t_rs}^{t} lambda >= target.
        # We do this on the grid with a conservative segment-level search.
        # (t_star need not be a grid boundary; we handle partial overlap in the start segment.)
        # Locate segment indices.
        j_s = int(np.searchsorted(grid, t_rs, side="right") - 1)
        j_e = int(np.searchsorted(grid, t_re, side="right") - 1)
        j_s = int(np.clip(j_s, 0, S - 1))
        j_e = int(np.clip(j_e, 0, S - 1))

        # Accumulate mass from t_rs forward until reaching target.
        acc = 0.0
        t_star = t_rs

        # Start within segment j_s at offset.
        j = j_s
        while True:
            seg_a = grid[j]
            seg_b = grid[j + 1]
            lam_j = float(lambda_seg[j])

            a = max(t_rs, seg_a)
            b = min(t_re, seg_b)
            if b <= a:
                if j >= j_e:
                    t_star = t_re
                    break
                j += 1
                continue

            seg_mass = lam_j * (b - a)  # steps
            if acc + seg_mass >= target - 1e-15:
                # Target reached inside this segment.
                remaining = max(0.0, target - acc)
                if lam_j > 0:
                    t_star = a + remaining / lam_j
                else:
                    # Degenerate: no lambda in this segment; push to end of overlap.
                    t_star = b
                t_star = float(np.clip(t_star, t_rs, t_re))
                break

            acc += seg_mass
            if j >= j_e:
                t_star = t_re
                break
            j += 1

        # Window is [t_star, t_re)
        if t_star >= t_re:
            continue

        mass_window = int_lambda(t_star, t_re)  # steps
        if mass_window <= 0.0:
            continue

        # Distribute mu_r across overlapping segments proportionally to lambda*overlap_len,
        # then convert to per-segment rate by dividing by seg length.
        j_w_s = int(np.searchsorted(grid, t_star, side="right") - 1)
        j_w_e = int(np.searchsorted(grid, t_re, side="right") - 1)
        j_w_s = int(np.clip(j_w_s, 0, S - 1))
        j_w_e = int(np.clip(j_w_e, 0, S - 1))

        for jj in range(j_w_s, j_w_e + 1):
            seg_a = grid[jj]
            seg_b = grid[jj + 1]
            overlap_a = max(t_star, seg_a)
            overlap_b = min(t_re, seg_b)
            if overlap_b <= overlap_a:
                continue
            overlap_len = overlap_b - overlap_a
            seg_len = float(dt_seg[jj])
            lam_j = float(lambda_seg[jj])

            # token mass assigned to overlap portion
            mass_j = mu_r * (lam_j * overlap_len) / mass_window  # tokens
            corr_rate[jj] += mass_j / seg_len  # tokens/sec over full segment

    return corr_rate


def _trace_only_correction_rate_on_grid(
    *,
    mode: PrefillCorrectionIterativeMode,
    df_full: pd.DataFrame,
    grid: np.ndarray,
    chunk_size: int,
    prefill_df: pd.DataFrame,
    t_start_col: str,
    t_end_col: str,
    prefill_tokens_col: str,
    n_steps_col: str,
    lambda_seg: Optional[np.ndarray],
    prefill_req: pd.DataFrame,
) -> np.ndarray:
    """
    Build the iteration-m correction rate c^(m)(t) (tokens/sec) on the trace grid.

    Modes
    -----
    - "none": zero correction
    - "end"/"uniform": trace-only correction signals independent of beta/lambda
                      (reused from baseline implementation)
    - "beta_informed": paper Eq. (cr_beta_informed_m), depends on lambda^(m)

    Returns corr_rate (shape S), tokens/sec.
    """
    S = grid.size - 1
    if mode == "none":
        return np.zeros(S, dtype=np.float64)

    if mode in ("end", "uniform"):
        corr_rate, _I_corr = _build_partial_chunk_correction_rate(
            df_full,
            grid=grid,
            chunk_size=chunk_size,
            prefill_df=prefill_df,
            t_start_col=t_start_col,
            t_end_col=t_end_col,
            prefill_tokens_col=prefill_tokens_col,
            n_steps_col=n_steps_col,
            mode=mode,  # type: ignore[arg-type]
        )
        return corr_rate

    # beta_informed
    if lambda_seg is None:
        raise ValueError("lambda_seg must be provided for beta_informed correction.")
    return _beta_informed_correction_rate_on_grid(grid=grid, lambda_seg=lambda_seg, prefill_req=prefill_req)


def _step_averaged_pressures_per_phase(
    *,
    df: pd.DataFrame,
    grid: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    lambda_seg: np.ndarray,           # steps/sec on segments
    tilde_p_pf_seg: np.ndarray,       # tokens/step on segments
    p_dec: np.ndarray,                # tokens/step on segments
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bar_p_pf_i^(m) and bar_p_dec_i^(m) (tokens/step) for each phase.

    Faithful to paper Eq. (step_averaged_pressures_m)
    -------------------------------------------------
    q_i^(m)(t) = lambda^(m)(t) / Λ_i^(m),  where Λ_i^(m) = ∫_{t_is}^{t_ie} lambda^(m)(t) dt

    bar_p_pf_i^(m)  = ∫ tilde_p_pf^(m)(t) q_i^(m)(t) dt
                    = (1/Λ_i) ∫ tilde_p_pf^(m)(t) lambda^(m)(t) dt

    bar_p_dec_i^(m) = (1/Λ_i) ∫ p_dec(t) lambda^(m)(t) dt
    """
    # Build segment signals to integrate.
    # y_pf = tilde_p_pf * lambda  has units tokens/sec
    y_pf = tilde_p_pf_seg * lambda_seg
    y_dec = p_dec * lambda_seg

    I_lambda = _prefix_integral(grid, lambda_seg)  # steps
    I_y_pf = _prefix_integral(grid, y_pf)          # tokens
    I_y_dec = _prefix_integral(grid, y_dec)        # tokens

    n = len(df)
    bar_pf = np.zeros(n, dtype=np.float64)
    bar_dec = np.zeros(n, dtype=np.float64)

    for i in range(n):
        a = float(starts[i])
        b = float(ends[i])

        Lambda_i = float(_integral_on_interval(grid, I_lambda, a, b, lambda_seg))
        if Lambda_i <= 0.0:
            # Degenerate; if lambda is ~0, attribution is undefined.
            # We fall back to 0 pressures (safe, but indicates pathological data/betas).
            bar_pf[i] = 0.0
            bar_dec[i] = 0.0
            continue

        num_pf = float(_integral_on_interval(grid, I_y_pf, a, b, y_pf))
        num_dec = float(_integral_on_interval(grid, I_y_dec, a, b, y_dec))

        bar_pf[i] = num_pf / Lambda_i
        bar_dec[i] = num_dec / Lambda_i

    return bar_pf, bar_dec

def _compute_r2_rmse_from_xy(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    resid = y_hat - y
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y))) if len(y) else float("nan")
    return r2, rmse


# =============================================================================
# Main estimator
# =============================================================================
def estimate_betas_iterative(
    phases: pd.DataFrame,
    *,
    chunk_size: int,
    # Column names (match baseline defaults)
    request_id_col: str = "request_id",
    phase_type_col: str = "phase_type",         # "prefill" or "decode"
    t_start_col: str = "t_start",
    t_end_col: str = "t_end",
    prefill_tokens_col: str = "prefill_tokens",
    decode_tokens_col: str = "decode_tokens",
    n_steps_col: str = "N_steps",
    min_duration_sec: float = 0.0,

    # Iterative controls
    prefill_correction_mode: PrefillCorrectionIterativeMode = "beta_informed",
    max_outer_iters: int = 25,
    tol: float = 1e-6,
    damping_eta: float = 1.0,
    init_via_baseline_correction: PartialChunkCorrectionMode = "end",
) -> IterativeBetaResult:
    """
    Iterative estimator (step-density reweighted, MM-style NNLS).

    This is the public entrypoint used by examples/run.py.

    Inputs
    ------
    phases : pd.DataFrame
        One row per phase instance i (prefill or decode).
        Same schema expectations as the baseline estimator.

    chunk_size : int
        Prefill chunk size C > 0 (tokens per prefill step).

    prefill_correction_mode : {"end","uniform","none","beta_informed"}
        - end/uniform/none:
            Use the corresponding trace-only correction signal c(t) (tokens/sec)
            for all outer iterations, independent of beta/lambda.
        - beta_informed:
            Use the paper's beta-informed localization of missing mass via
            lambda^(m) and last-step windows (Eq. (qr_def_m)–(cr_beta_informed_m)).

    init_via_baseline_correction : {"end","uniform","none"}
        Correction mode used for baseline initialization beta^(0) (unless the caller
        wants to provide beta^(0) externally; not supported in this minimal API).

    Outputs
    -------
    IterativeBetaResult
        Includes final betas, convergence info, diagnostics, and beta history.

    Paper-faithfulness notes
    ------------------------
    - This implementation follows the paper's outer-loop semantics exactly:
        (beta^(m-1), tilde_p_pf^(m-1)) -> lambda^(m) -> features^(m) -> beta^(m)
      with features frozen during the NNLS solve.
    - All attribution is trace-only: uses only phase boundaries and grid integrals.
    - No step boundaries or per-step timings are used or inferred.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if max_outer_iters <= 0:
        raise ValueError("max_outer_iters must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if not (0.0 < damping_eta <= 1.0):
        raise ValueError("damping_eta must be in (0,1].")

    # -------------------------------------------------------------------------
    # 1) Validate schema and normalize (reuse baseline helper)
    # -------------------------------------------------------------------------
    df = _validate_and_normalize_schema(
        phases,
        request_id_col=request_id_col,
        phase_type_col=phase_type_col,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
    )
    if df.empty:
        raise ValueError("No phase rows provided after validation.")

    # -------------------------------------------------------------------------
    # 2) Ensure step counts N_i exist (reuse baseline helper)
    # -------------------------------------------------------------------------
    df = _infer_or_read_step_counts(
        df,
        chunk_size=chunk_size,
        phase_type_col=phase_type_col,
        prefill_tokens_col=prefill_tokens_col,
        decode_tokens_col=decode_tokens_col,
        n_steps_col=n_steps_col,
    )

    # -------------------------------------------------------------------------
    # 3) Durations and (baseline) lambda_hat (reuse baseline helper)
    #     We still compute T_i here because it's the regression target.
    # -------------------------------------------------------------------------
    df = _compute_durations_and_lambda_hat(
        df,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
        min_duration_sec=min_duration_sec,
    )

    # Core numeric arrays used throughout.
    starts, ends, grid = _build_global_time_grid_from_df(df, t_start_col=t_start_col, t_end_col=t_end_col)
    n = len(df)
    N_i = df["N_i"].to_numpy(dtype=np.float64)
    T_i = df["T_i"].to_numpy(dtype=np.float64)

    # -------------------------------------------------------------------------
    # 4) Reconstruct instantaneous pressures on the trace grid (reuse baseline helper)
    # -------------------------------------------------------------------------
    p_pf_full, p_dec, _I_pf_full, _I_dec = _reconstruct_pressures(
        df,
        grid=grid,
        starts=starts,
        ends=ends,
        chunk_size=chunk_size,
        phase_type_col=phase_type_col,
    )
    S = grid.size - 1
    if p_pf_full.shape != (S,) or p_dec.shape != (S,):
        raise ValueError("Pressure reconstruction returned unexpected shapes.")

    # -------------------------------------------------------------------------
    # 5) Build prefill request table for correction logic
    # -------------------------------------------------------------------------
    prefill_df = _check_baseline_prefill_uniqueness(
        df,
        request_id_col=request_id_col,
        phase_type_col=phase_type_col,
    )
    prefill_req = _prefill_request_table(
        df,
        request_id_col=request_id_col,
        phase_type_col=phase_type_col,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
        prefill_tokens_col=prefill_tokens_col,
        chunk_size=chunk_size,
        n_steps_col=n_steps_col,
    )

    # -------------------------------------------------------------------------
    # 6) Initialize beta^(0) via baseline (paper recommendation / common practice)
    # -------------------------------------------------------------------------
    init_res: BaselineBetaResult = estimate_betas_baseline(
        df,
        chunk_size=chunk_size,
        request_id_col=request_id_col,
        phase_type_col=phase_type_col,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
        prefill_tokens_col=prefill_tokens_col,
        decode_tokens_col=decode_tokens_col,
        n_steps_col=n_steps_col,
        min_duration_sec=min_duration_sec,
        correction_mode=init_via_baseline_correction,
    )
    beta_prev = np.array([init_res.beta0, init_res.beta1, init_res.beta2], dtype=np.float64)

    # -------------------------------------------------------------------------
    # 7) Initialize tilde_p_pf^(0)(t) (lagged effective prefill pressure)
    #
    # Paper view:
    #   Iteration m uses tilde_p_pf^(m-1) inside Delta^(m).
    #
    # Practical choice (trace-only, stable):
    #   Start with tilde_p_pf^(0) = p_pf_full (no step-density correction yet).
    #
    # This matches the paper’s “lagged effective pressure” semantics:
    #   you need some tilde_p_pf^(0) before you can compute lambda^(1).
    # -------------------------------------------------------------------------
    tilde_p_pf_prev = p_pf_full.astype(np.float64, copy=True)

    # -------------------------------------------------------------------------
    # History containers
    # -------------------------------------------------------------------------
    beta_hist = [tuple(map(float, beta_prev))]
    delta_hist: list[float] = []
    r2_hist: list[float] = []
    rmse_hist: list[float] = []
    notes: list[str] = []

    converged = False
    n_outer_executed = 0

    # -------------------------------------------------------------------------
    # 8) Outer MM loop
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 8) Outer MM loop (paper-faithful: NO inner passes)
    # -------------------------------------------------------------------------
    for m in range(1, max_outer_iters + 1):
        # (a) Compute lambda^(m) from beta^(m-1) and tilde_p_pf^(m-1)
        #     Paper Eq. (local_step_time_m) + Eq. (step_density_m):
        #       Delta^(m)(t) = beta^(m-1)_0 + beta^(m-1)_1 * tilde_p_pf^(m-1)(t) + beta^(m-1)_2 * p_dec(t)
        #       lambda^(m)(t) = 1 / Delta^(m)(t)
        _Delta_seg, lambda_seg = _lambda_segment_from_beta_and_pressures(
            beta_prev=beta_prev,
            tilde_p_pf_prev=tilde_p_pf_prev,   # <-- lagged effective prefill pressure (m-1)
            p_dec=p_dec,
        )

        # (b) Compute correction c^(m)(t) using frozen lambda^(m), then tilde_p_pf^(m)(t)
        #     Paper Eq. (cr_beta_informed_m) + Eq. (prefill_pressure_beta_informed_m):
        #       tilde_p_pf^(m)(t) = p_pf_full(t) - c^(m)(t) / lambda^(m)(t)
        #
        # IMPORTANT (paper semantics): c^(m) depends on lambda^(m), but lambda^(m)
        # depends ONLY on tilde_p_pf^(m-1), not on tilde_p_pf^(m).
        corr_rate_m = _trace_only_correction_rate_on_grid(
            mode=prefill_correction_mode,
            df_full=df,
            grid=grid,
            chunk_size=chunk_size,
            prefill_df=prefill_df,
            t_start_col=t_start_col,
            t_end_col=t_end_col,
            prefill_tokens_col=prefill_tokens_col,
            n_steps_col=n_steps_col,
            lambda_seg=lambda_seg,            # <-- frozen lambda^(m)
            prefill_req=prefill_req,
        )

        denom = np.maximum(lambda_seg, 1e-12)  # guard division
        tilde_p_pf_m = p_pf_full - (corr_rate_m / denom)
        tilde_p_pf_m = np.maximum(tilde_p_pf_m, 0.0)  # numeric safety

        # (c) Compute phase step-averaged pressures using frozen lambda^(m)
        #     Paper Eq. (step_averaged_pressures_m):
        #       bar_p_i^(m) = ∫ p(t) q_i^(m)(t) dt, where q_i^(m) ∝ lambda^(m)
        bar_pf, bar_dec = _step_averaged_pressures_per_phase(
            df=df,
            grid=grid,
            starts=starts,
            ends=ends,
            lambda_seg=lambda_seg,            # <-- frozen lambda^(m)
            tilde_p_pf_seg=tilde_p_pf_m,      # <-- tilde_p_pf^(m)
            p_dec=p_dec,
        )

        # Exposures in tokens for NNLS (Eq. (sr_nnls_m)):
        exp_pf = N_i * bar_pf
        exp_dec = N_i * bar_dec

        # (d) Solve frozen-feature NNLS for beta_fit, then apply damping to obtain beta^(m).
        b0_fit, b1_fit, b2_fit, fit_diag = _fit_nnls_from_features(
            N_i=N_i,
            exp_pf_tokens=exp_pf,
            exp_dec_tokens=exp_dec,
            T_i=T_i,
        )
        beta_fit = np.array([b0_fit, b1_fit, b2_fit], dtype=np.float64)
        beta_new = (1.0 - damping_eta) * beta_prev + damping_eta * beta_fit

        # (e) Convergence check
        d = float(np.linalg.norm(beta_new - beta_prev, ord=2))
        beta_prev = beta_new
        tilde_p_pf_prev = tilde_p_pf_m  # advance lagged effective pressure

        beta_hist.append(tuple(map(float, beta_prev)))
        delta_hist.append(d)
        r2_hist.append(float(fit_diag.get("r2", float("nan"))))
        rmse_hist.append(float(fit_diag.get("rmse_seconds", float("nan"))))
        notes.append(f"outer_iter={m}")

        n_outer_executed = m
        if d < tol:
            converged = True
            break

    # -------------------------------------------------------------------------
    # Final diagnostics (simple, final-iteration scale checks)
    # -------------------------------------------------------------------------
    diagnostics: Dict[str, Any] = {
        "converged": converged,
        "n_outer_iters": float(n_outer_executed),
        "tol": float(tol),
        "damping_eta": float(damping_eta),
        "init_beta0": float(init_res.beta0),
        "init_beta1": float(init_res.beta1),
        "init_beta2": float(init_res.beta2),
        "final_beta0": float(beta_prev[0]),
        "final_beta1": float(beta_prev[1]),
        "final_beta2": float(beta_prev[2]),
    }


    diagnostics["final_r2"] = float(r2_hist[-1]) if r2_hist else float("nan")
    diagnostics["final_rmse_seconds"] = float(rmse_hist[-1]) if rmse_hist else float("nan")
    if delta_hist:
        diagnostics["final_delta_beta_l2"] = float(delta_hist[-1])

    # Feature scale diagnostics (helpful for debugging unit mistakes)
    diagnostics["mean_steps"] = float(np.mean(N_i)) if n else float("nan")
    diagnostics["mean_duration_seconds"] = float(np.mean(T_i)) if n else float("nan")
    diagnostics["n_phases"] = float(n)

    history: Dict[str, object] = {
        "beta": beta_hist,
        "delta_beta_l2": delta_hist,
        "r2": r2_hist,
        "rmse_seconds": rmse_hist,
        "notes": notes,
    }

    return IterativeBetaResult(
        beta0=float(beta_prev[0]),
        beta1=float(beta_prev[1]),
        beta2=float(beta_prev[2]),
        converged=converged,
        n_outer_iters=n_outer_executed,
        diagnostics=diagnostics,
        history=history,
    )


# =============================================================================
# Minimal example (mirrors baseline's style)
# =============================================================================
def _example() -> None:
    """
    Tiny end-to-end example.

    Creates two requests (A,B), each with:
      - one prefill phase
      - one decode phase

    This is a smoke test demonstrating expected input schema and the iterative
    estimator invocation.

    Notes
    -----
    This does *not* validate correctness of recovered betas (needs a controlled
    synthetic generator for that). It only validates:
      - code runs,
      - shapes are consistent,
      - outer loop converges on simple inputs.
    """
    # phases = pd.DataFrame(
    #     [
    #         {"request_id": "A", "phase_type": "prefill", "t_start": 0.0, "t_end": 2.0, "prefill_tokens": 96, "decode_tokens": 0},
    #         {"request_id": "A", "phase_type": "decode",  "t_start": 2.0, "t_end": 6.0, "prefill_tokens": 0,  "decode_tokens": 4},
    #         {"request_id": "B", "phase_type": "prefill", "t_start": 1.0, "t_end": 3.0, "prefill_tokens": 64, "decode_tokens": 0},
    #         {"request_id": "B", "phase_type": "decode",  "t_start": 3.0, "t_end": 5.0, "prefill_tokens": 0,  "decode_tokens": 2},
    #     ]
    # )
    train_path = os.path.join("train", "*/")
    all_train = glob.glob(train_path)
    for train_path in all_train:
        phases_path = os.path.join(train_path, "vllm_phases.csv")
        vllm_config_path = os.path.join(train_path, "exp-config.yaml")
        # get TP value from vllm_config
        try:
            with open(vllm_config_path, 'r') as f:
                vllm_config = yaml.safe_load(f)
                model = vllm_config["model"].split("/")[1].lower()
                tp = vllm_config["tensor_parallelism"]
                chunk_size = vllm_config["max_num_batched_tokens"]
        except:
            print("Could not load vllm config data.")
            sys.exit()
        phases = pd.read_csv(phases_path)
        models_dir = os.path.join("models", f"model_{model}_tp_{tp}")
        os.makedirs(models_dir, exist_ok=True)
        train_alpha_model(train_path, models_dir)
        res = estimate_betas_iterative(
            phases,
            chunk_size=chunk_size,
            prefill_correction_mode="beta_informed",
            max_outer_iters=10,
            tol=1e-6,
            damping_eta=1.0,
            init_via_baseline_correction="end",
        )
        models_path = os.path.join(models_dir, "BLIS_beta_metrics.json")
        betas = {
                "best_params": {
                "beta0": res.beta0,
                "beta1": res.beta1,
                "beta2": res.beta2
            }
        }
        with open(models_path, "w+") as f:
            json.dump(betas, f)

        print("Estimated betas (iterative):")
        print(f"  beta0 (sec/step):  {res.beta0:.6f}")
        print(f"  beta1 (sec/token): {res.beta1:.6f}")
        print(f"  beta2 (sec/token): {res.beta2:.6f}")
        print()
        print(f"Converged: {res.converged}  (outer iters: {res.n_outer_iters})")
        print("\nDiagnostics:")
        for k, v in res.diagnostics.items():
            print(f"  {k}: {v}")

        beta_hist = res.history.get("beta", None)
        if beta_hist is not None and len(beta_hist) > 0:
            tail = beta_hist[-min(5, len(beta_hist)) :]
            print("\nBeta history (last few iters):")
            start_idx = len(beta_hist) - len(tail)
            for j, b in enumerate(tail, start=start_idx):
                print(f"  iter {j:>3d}: beta0={b[0]:.6f}, beta1={b[1]:.6f}, beta2={b[2]:.6f}")


if __name__ == "__main__":  # pragma: no cover
    _example()
