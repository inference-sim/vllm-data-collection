#!/usr/bin/env python3
"""
baseline_step_betas.py

Baseline trace-only estimator for step-level execution coefficients (betas)
for a vLLM-style inference engine.

Implements the *baseline* estimator in the paper section
"Baseline: Time-Integrated NNLS Estimation" (Eq. (baseline_nnls)).

High-level pipeline (paper-consistent)
--------------------------------------
Given a trace of *phase instances* (one row per phase instance i, where i is
either the prefill phase or decode phase of a request r_i):

  1) Validate/normalize schema and timestamps.
  2) Infer trace-level step counts N_i (if not already provided).
       - Decode: N_i = D_{r_i}  (1 token/step)
       - Prefill: N_i = ceil(P_{r_i} / C)
  3) Compute phase-local step-density proxy λ̂_i = N_i / ℓ_i
     (Eq. (baseline_lambda_hat)), where ℓ_i is the observed phase duration.
     NOTE: in code, ℓ_i is stored as df["T_i"].
  4) Reconstruct time-varying token pressures from overlap on the trace-induced grid:
       - p_pf_full(t) = C * (# active prefill phases at t)   [tokens/step]
       - p_dec(t)     =      (# active decode phases at t)   [tokens/step]
     Under the decode semantics (1 token/request/step), p_dec(t) equals the
     number of concurrently active decode phases.
  5) Apply partial-chunk correction for prefill chunking (mode: end/uniform/none):
       - subtract missing prefill mass μ_r in *integrated form* (tokens),
         implemented via a correction signal c(t) (tokens/sec).
         NOTE: in code, c(t) is represented by corr_rate (tokens/sec).
  6) Form baseline integrated exposures (paper Eq. (baseline_integrated_exposures)):
       - A_pf_i  [tokens]
       - A_dec_i [tokens]
  7) Fit NNLS (sklearn LinearRegression(positive=True), no intercept) (Eq. (baseline_nnls)):
       ℓ_i ≈ β0*N_i + β1*A_pf_i + β2*A_dec_i
     NOTE: in code, ℓ_i is stored as df["T_i"].

IMPORTANT invariants (trace-only)
---------------------------------
- This is a trace-only estimator: it never uses per-step timing or step boundaries.
- The simulator later consumes only β = (β0, β1, β2); it generates its own step-level
  token counts and KV dynamics. No trace timing is embedded into the simulator.

Notation bridge to the paper
----------------------------
- Phase instance i (prefill or decode of request r_i):
    t_{i,s}, t_{i,e}  -> df[t_start_col], df[t_end_col]
    ℓ_i               -> df["T_i"]              (observed duration in seconds)
    N_i               -> df["N_i"]              (trace-inferred step count)
    λ̂_i              -> df["lambda_hat"]       (steps/sec)
    A_pf_i, A_dec_i   -> df["A_pf_i"], df["A_dec_i"] (tokens)
- Chunk size:
    C                 -> chunk_size
- Pressures:
    p_pf_full(t)      -> p_pf_full (piecewise-constant over time-grid segments)
    p_dec(t)          -> p_dec
- Partial-chunk correction:
    c(t)              -> corr_rate (tokens/sec over segments)
    ∫ c(t) dt         -> corr_tokens (tokens)

Shapes convention
-----------------
Let:
  n = number of phase instances (rows in df)
  M = number of unique time boundaries in the global time grid
  S = M-1 = number of time segments (intervals [grid[j], grid[j+1]))

Then:
  starts, ends          : (n,) float64
  grid                  : (M,) float64
  n_pf, n_dec            : (S,) int64
  p_pf_full, p_dec       : (S,) float64   (tokens/step)
  corr_rate              : (S,) float64   (tokens/second)  [paper: c(t)]
  I_pf_full, I_dec       : (M,) float64
  I_corr                 : (M,) float64
  A_pf, A_dec            : (n,) float64   (tokens)

Dependencies
------------
pip install numpy pandas scikit-learn
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union, Literal

import json
import numpy as np
import os
import pandas as pd
import glob
import yaml
import sys
from sklearn.linear_model import LinearRegression
from blis_alpha_model import train_alpha_model


# =============================================================================
# Result structure
# =============================================================================
@dataclass(frozen=True)
class BaselineBetaResult:
    """
    Output of baseline beta estimation.

    Attributes
    ----------
    beta0 : float
        Per-step fixed overhead β0 in seconds/step.
        Interpreted as the additive cost of executing one busy-loop step
        independent of token processing. See Eq. (step_model).

    beta1 : float
        Prefill token cost β1 in seconds/token.
        Multiplies the prefill token exposure A_pf_i (tokens) in Eq. (baseline_predictor).

    beta2 : float
        Decode token cost β2 in seconds/token.
        Multiplies the decode token exposure A_dec_i (tokens) in Eq. (baseline_predictor).

    diagnostics : Dict[str, float]
        Convenience metrics for sanity checks only (R^2, RMSE, feature means, etc.).
        These are not used by the estimator itself.

    Notes on units
    --------------
    The regression model is:
        T_i ≈ β0*N_i + β1*A_pf_i + β2*A_dec_i
    where:
        T_i    : seconds
        N_i    : steps
        A_*_i  : tokens
    so each term has units of seconds.
    """
    beta0: float
    beta1: float
    beta2: float
    diagnostics: Dict[str, float]


# =============================================================================
# Small helpers (kept tiny and well-commented)
# =============================================================================
def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Validate that all required columns exist.

    Inputs
    ------
    df : pd.DataFrame
        Any DataFrame.
    cols : list[str]
        Column names that must exist.

    Outputs
    -------
    None
        Raises ValueError if any required columns are missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


ArrayLike1D = Union[np.ndarray, pd.Series]


def _assert_integerish(x: ArrayLike1D, *, name: str, tol: float = 1e-6) -> None:
    """
    Assert that an array is non-negative and integer-valued up to tolerance.

    This is used to enforce the paper assumption that N_i is an integer
    step count (N_i ∈ ℕ).

    Inputs
    ------
    x : np.ndarray or pd.Series, shape (n,)
        Candidate step counts.
    name : str
        Name used in error messages.
    tol : float
        Allowed deviation from nearest integer.

    Outputs
    -------
    None
        Raises ValueError if negatives exist or values are non-integer within tol.

    Notes
    -----
    Accepts float arrays because upstream inference uses ceil() and casts to float.
    """
    arr = x.to_numpy(dtype=np.float64) if isinstance(x, pd.Series) else x.astype(np.float64, copy=False)

    if np.any(arr < -tol):
        raise ValueError(f"{name} contains negative values.")
    if np.any(np.abs(arr - np.round(arr)) > tol):
        raise ValueError(f"{name} must be integer-valued within tol={tol}.")


def _build_time_grid(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Construct the global time grid used for piecewise-constant overlap signals.

    The estimator assumes that activity counts (how many phases are active)
    only change at phase boundaries. Therefore, pressures are piecewise-constant
    on intervals between unique boundaries.

    Inputs
    ------
    starts : np.ndarray, shape (n,)
        Phase start times (seconds).
    ends : np.ndarray, shape (n,)
        Phase end times (seconds).

    Outputs
    -------
    grid : np.ndarray, shape (M,)
        Sorted unique timestamps from starts and ends.
        Defines segments [grid[j], grid[j+1]) for j=0..M-2.

    Raises
    ------
    ValueError
        If there are fewer than 2 distinct timestamps.
    """
    grid = np.unique(np.concatenate([starts, ends]))
    grid.sort()
    if grid.size < 2:
        raise ValueError("Need at least two distinct timestamps for a time grid.")
    return grid


def _sweepline_counts(grid: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Sweep-line active phase counts per segment of the time grid.

    For a set of intervals [starts[m], ends[m]), this computes how many intervals
    are active on each segment [grid[j], grid[j+1]).

    Inputs
    ------
    grid : np.ndarray, shape (M,)
        Global time grid (sorted unique boundaries).
    starts : np.ndarray, shape (k,)
        Start times for a subset of phases (e.g., prefill-only).
    ends : np.ndarray, shape (k,)
        End times for that same subset.

    Outputs
    -------
    counts : np.ndarray, shape (S=M-1,)
        counts[j] = number of intervals active on [grid[j], grid[j+1]).

    Notes
    -----
    Implementation uses a standard difference-array / prefix-sum trick:
      delta[start_idx] += 1
      delta[end_idx]   -= 1
      counts = cumsum(delta)[:-1]
    """
    M = grid.size
    delta = np.zeros(M, dtype=np.int64)

    s_idx = np.searchsorted(grid, starts, side="left")
    e_idx = np.searchsorted(grid, ends, side="left")

    np.add.at(delta, s_idx, 1)
    np.add.at(delta, e_idx, -1)

    return np.cumsum(delta)[:-1]


def _prefix_integral(grid: np.ndarray, y_segment: np.ndarray) -> np.ndarray:
    """
    Prefix integral for a piecewise-constant signal defined on grid segments.

    If y_segment[j] is the constant value of y(t) on [grid[j], grid[j+1]),
    then this returns an array I such that:
        I[k] = ∫_{grid[0]}^{grid[k]} y(t) dt

    Inputs
    ------
    grid : np.ndarray, shape (M,)
        Time grid.
    y_segment : np.ndarray, shape (S=M-1,)
        Segment values (constant per segment).

    Outputs
    -------
    I : np.ndarray, shape (M,)
        Prefix integral evaluated at grid points.

    Units
    -----
    If y_segment is:
      - tokens/step, then I is (tokens/step)*seconds
      - tokens/second, then I is tokens
    """
    dt = grid[1:] - grid[:-1]        # shape (S,)
    area = y_segment * dt            # shape (S,)
    I = np.zeros(grid.size, dtype=np.float64)
    I[1:] = np.cumsum(area)
    return I


def _integral_on_interval(
    grid: np.ndarray,
    I: np.ndarray,
    a: float,
    b: float,
    y_segment: np.ndarray,
) -> float:
    """
    Compute ∫_{a}^{b} y(t) dt where y is piecewise-constant on the time grid.

    Inputs
    ------
    grid : np.ndarray, shape (M,)
        Time grid boundaries.
    I : np.ndarray, shape (M,)
        Prefix integral from _prefix_integral(grid, y_segment).
    a : float
        Interval start time (seconds).
    b : float
        Interval end time (seconds).
    y_segment : np.ndarray, shape (S=M-1,)
        Piecewise-constant segment values of y(t).

    Output
    ------
    integral : float
        Value of the definite integral over [a, b).

    Notes
    -----
    Uses:
      - prefix integrals for fully-covered interior segments
      - explicit left/right partial segment corrections

    This function is called in a per-phase loop, so it must be correct and stable.
    """
    if b <= a:
        return 0.0

    ia = np.searchsorted(grid, a, side="right") - 1
    ib = np.searchsorted(grid, b, side="right") - 1

    ia = int(np.clip(ia, 0, grid.size - 2))
    ib = int(np.clip(ib, 0, grid.size - 2))

    if ia == ib:
        return float(y_segment[ia] * (b - a))

    full = float(I[ib] - I[ia + 1])  # ∫ grid[ia+1]..grid[ib]
    left = float(y_segment[ia] * (grid[ia + 1] - a))
    right = float(y_segment[ib] * (b - grid[ib]))
    return left + full + right


# =============================================================================
# Refactoring: small, reusable pipeline functions
# =============================================================================
def _validate_and_normalize_schema(
    phases: pd.DataFrame,
    *,
    request_id_col: str,
    phase_type_col: str,
    t_start_col: str,
    t_end_col: str,
) -> pd.DataFrame:
    """
    Validate the input schema and normalize basic types.

    Conceptually, this corresponds to the paper's assumption that the trace provides
    well-defined phase instances with:
      - request identifier
      - phase type (prefill or decode)
      - start/end times defining a positive duration

    Inputs
    ------
    phases : pd.DataFrame, shape (n, ...)
        One row per phase instance i.
        Required columns (names configurable via *_col parameters):
          - request_id_col : request identifier (any hashable)
          - phase_type_col : string "prefill" or "decode" (case-insensitive)
          - t_start_col    : phase start time t_{i,s} (seconds)
          - t_end_col      : phase end time t_{i,e} (seconds)

    Outputs
    -------
    df : pd.DataFrame, shape (n, ...)
        Copy of the input with:
          - t_start_col, t_end_col cast to float
          - phase_type_col normalized to lowercase
        All original columns are preserved (no data is dropped).

    Raises
    ------
    ValueError
        If required columns are missing, phase_type has invalid values,
        or any row has non-positive duration (t_end <= t_start).
    """
    df = phases.copy()

    _require_columns(df, [request_id_col, phase_type_col, t_start_col, t_end_col])

    df[t_start_col] = df[t_start_col].astype(float)
    df[t_end_col] = df[t_end_col].astype(float)

    if (df[t_end_col] <= df[t_start_col]).any():
        bad = df[df[t_end_col] <= df[t_start_col]][[request_id_col, phase_type_col, t_start_col, t_end_col]]
        raise ValueError(f"Found non-positive durations (t_end <= t_start):\n{bad}")

    df[phase_type_col] = df[phase_type_col].astype(str).str.lower()
    valid_types = {"prefill", "decode"}
    bad_types = sorted(set(df[phase_type_col]) - valid_types)
    if bad_types:
        raise ValueError(f"phase_type must be one of {sorted(valid_types)}. Bad values: {bad_types}")

    return df


def _infer_or_read_step_counts(
    df: pd.DataFrame,
    *,
    chunk_size: int,
    phase_type_col: str,
    prefill_tokens_col: str,
    decode_tokens_col: str,
    n_steps_col: str,
) -> pd.DataFrame:
    """
    Ensure trace-inferred step counts N_i exist and are integer-valued.

    Paper mapping (Problem Setup)
    -----------------------------
    Each row is a *phase instance* i (prefill or decode) of a request r_i.

    - Decode phase instance i:
        N_i = D_{r_i}
      because decode generates 1 token per busy-loop step.

    - Prefill phase instance i (chunk size C):
        N_i = ceil(P_{r_i} / C)

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain phase_type_col.
        If n_steps_col is missing, must also contain:
          - prefill_tokens_col : interpreted as P_{r_i} for prefill rows
          - decode_tokens_col  : interpreted as D_{r_i} for decode rows
    chunk_size : int
        Prefill chunk size C (tokens per prefill step), C > 0.
    phase_type_col : str
        Column name for phase type ("prefill" or "decode").
    prefill_tokens_col : str
        Column name storing P_{r_i} on prefill rows.
    decode_tokens_col : str
        Column name storing D_{r_i} on decode rows.
    n_steps_col : str
        Column name for step count N_i (optional input; inferred if missing).

    Outputs
    -------
    df : pd.DataFrame, shape (n, ...)
        Same DataFrame with:
          - df[n_steps_col] present (float values but integer-ish)
          - df["N_i"] present as float (shape (n,))
        Values satisfy the integer-ish check (within tolerance).

    Notes
    -----
    The estimator treats N_i as known:
      - it is a regressor in the baseline NNLS objective, and
      - it is used to form λ̂_i = N_i / ℓ_i (Eq. (baseline_lambda_hat)).
    """
    if n_steps_col not in df.columns:
        _require_columns(df, [prefill_tokens_col, decode_tokens_col])

        pre = df[phase_type_col].eq("prefill")
        dec = df[phase_type_col].eq("decode")

        df[n_steps_col] = np.nan

        P = df.loc[pre, prefill_tokens_col].astype(float).to_numpy()
        D = df.loc[dec, decode_tokens_col].astype(float).to_numpy()

        # Prefill: ceil(P/C)
        df.loc[pre, n_steps_col] = np.ceil(P / float(chunk_size))

        # Decode: 1 token per step
        df.loc[dec, n_steps_col] = D

    df["N_i"] = df[n_steps_col].astype(float).to_numpy()
    _assert_integerish(df["N_i"], name="N_steps")
    return df


def _compute_durations_and_lambda_hat(
    df: pd.DataFrame,
    *,
    t_start_col: str,
    t_end_col: str,
    min_duration_sec: float,
) -> pd.DataFrame:
    """
    Compute observed phase duration and phase-local step-density proxy λ̂_i.

    Paper mapping
    -------------
    For a phase instance i with trace-visible boundaries:
      - ℓ_i = t_{i,e} - t_{i,s}     (seconds)
      - λ̂_i = N_i / ℓ_i           (steps/sec), Eq. (baseline_lambda_hat)

    Code naming note (important)
    ----------------------------
    The paper denotes duration by ℓ_i. This implementation stores ℓ_i as df["T_i"].

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain:
          - df["N_i"] (steps, shape (n,))
          - t_start_col, t_end_col (seconds)
    min_duration_sec : float
        Optional clamp applied to ℓ_i when forming λ̂_i:
          ℓ_i := max(ℓ_i, min_duration_sec)
        This corresponds to the implementation footnote-style safeguard mentioned
        in the paper text (to avoid extreme λ̂_i when durations are extremely small).
        Set to 0.0 for the paper-faithful definition.

    Outputs
    -------
    df : pd.DataFrame, shape (n, ...)
        Adds:
          - df["T_i"]        : ℓ_i, seconds
          - df["lambda_hat"] : λ̂_i, steps/second

    Raises
    ------
    ValueError
        If any resulting durations are non-positive.
    """

    T = (df[t_end_col] - df[t_start_col]).to_numpy(dtype=np.float64)
    if min_duration_sec > 0:
        T = np.maximum(T, float(min_duration_sec))
    df["T_i"] = T

    if (df["T_i"] <= 0).any():
        raise ValueError("All phase durations must be positive.")

    df["lambda_hat"] = (df["N_i"] / df["T_i"]).to_numpy(dtype=np.float64)
    return df


def _check_baseline_prefill_uniqueness(
    df: pd.DataFrame,
    *,
    request_id_col: str,
    phase_type_col: str,
) -> pd.DataFrame:
    """
    Enforce a baseline-only trace schema assumption: one prefill row per request.

    Why this exists
    ---------------
    The partial-chunk correction is implemented per request using the single
    prefill interval [t_start, t_end). If a request's prefill is split into multiple
    rows/segments, this baseline implementation would need the segments to be made unique first.

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain request_id_col and phase_type_col.

    Outputs
    -------
    prefill_df : pd.DataFrame, shape (n_pf, ...)
        Subset of df with only prefill phases (phase_type == "prefill").

    Raises
    ------
    ValueError
        If any request_id appears in more than one prefill row.
    """
    prefill_df = df[df[phase_type_col].eq("prefill")]
    if not prefill_df.empty:
        counts = prefill_df[request_id_col].value_counts()
        if (counts > 1).any():
            bad_ids = counts[counts > 1].index.tolist()[:10]
            raise ValueError(
                "Multiple prefill rows per request found. Baseline assumes at most one. "
                f"Make the segments unique first. Example request_ids: {bad_ids}"
            )
    return prefill_df


def _build_global_time_grid_from_df(
    df: pd.DataFrame,
    *,
    t_start_col: str,
    t_end_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract phase start/end arrays and construct the global time grid.

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain t_start_col and t_end_col.

    Outputs
    -------
    starts : np.ndarray, shape (n,), float64
        Phase start times.
    ends : np.ndarray, shape (n,), float64
        Phase end times.
    grid : np.ndarray, shape (M,), float64
        Sorted unique union of starts and ends.

    Semantics
    ---------
    The grid defines S = M-1 segments.
    All reconstructed time-varying signals are represented as arrays of length S
    (one value per segment).
    """
    starts = df[t_start_col].to_numpy(dtype=np.float64)
    ends = df[t_end_col].to_numpy(dtype=np.float64)
    grid = _build_time_grid(starts, ends)
    return starts, ends, grid


def _reconstruct_pressures(
    df: pd.DataFrame,
    *,
    grid: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    chunk_size: int,
    phase_type_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct piecewise-constant token pressures from phase overlap.

    Paper mapping (Trace-derived token pressures)
    ---------------------------------------------
    Pressures are defined in units of tokens/step on the trace-induced time grid.

    - Prefill (naive full-chunk pressure):
        p_pf_full(t) = C * (# active prefill phase instances at time t)
      where C is the prefill chunk size (tokens per prefill step).

    - Decode:
        Under decode semantics (1 token/request/step),
        p_dec(t) = (# active decode phase instances at time t).

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain phase_type_col with values "prefill"/"decode".
    grid : np.ndarray, shape (M,)
        Global time grid (sorted unique phase boundaries).
    starts : np.ndarray, shape (n,)
        Phase start times aligned with df rows.
    ends : np.ndarray, shape (n,)
        Phase end times aligned with df rows.
    chunk_size : int
        C, tokens per prefill step.
    phase_type_col : str
        Column name for phase type.

    Outputs
    -------
    p_pf_full : np.ndarray, shape (S=M-1,), float64
        p_pf_full(t) on each grid segment, units: tokens/step.
    p_dec : np.ndarray, shape (S=M-1,), float64
        p_dec(t) on each grid segment, units: tokens/step.
    I_pf_full : np.ndarray, shape (M,), float64
        Prefix integral of p_pf_full over time:
          I_pf_full[k] = ∫_{grid[0]}^{grid[k]} p_pf_full(t) dt
        units: (tokens/step)*seconds.
    I_dec : np.ndarray, shape (M,), float64
        Prefix integral of p_dec over time, same units.

    Notes
    -----
    These pressures are *per step* (tokens/step), not per unit time.
    In the baseline, time-integrals of pressures are converted into token exposures
    by multiplying by the phase-local step-density proxy λ̂_i (steps/sec).
    """

    pre_mask = df[phase_type_col].eq("prefill").to_numpy()
    dec_mask = df[phase_type_col].eq("decode").to_numpy()

    n_pf = _sweepline_counts(grid, starts[pre_mask], ends[pre_mask]) if pre_mask.any() else np.zeros(grid.size - 1, dtype=np.int64)
    n_dec = _sweepline_counts(grid, starts[dec_mask], ends[dec_mask]) if dec_mask.any() else np.zeros(grid.size - 1, dtype=np.int64)

    p_pf_full = float(chunk_size) * n_pf.astype(np.float64)
    p_dec = n_dec.astype(np.float64)

    I_pf_full = _prefix_integral(grid, p_pf_full)
    I_dec = _prefix_integral(grid, p_dec)

    return p_pf_full, p_dec, I_pf_full, I_dec

# baseline.py
def _prefill_missing_mass_mu(
    P: np.ndarray,
    N: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """
    Compute missing prefill token mass μ = C - ρ where ρ = P - C*(N-1).

    Inputs
    ------
    P : array, tokens
    N : array, steps (integer-ish)
    chunk_size : int, C

    Output
    ------
    mu : array, tokens, in [0, C]
    """
    C = float(chunk_size)
    rho = P - C * (N - 1.0)
    rho = np.clip(rho, 0.0, C)
    mu = C - rho
    return np.clip(mu, 0.0, C)


def _build_uniform_partial_chunk_correction_rate(
    df: pd.DataFrame,
    *,
    grid: np.ndarray,
    chunk_size: int,
    prefill_df: pd.DataFrame,
    t_start_col: str,
    t_end_col: str,
    prefill_tokens_col: str,
    n_steps_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the baseline *uniform redistribution* prefill partial-chunk correction signal c(t).

    Paper mapping (Prefill chunking and correction)
    -----------------------------------------------
    For each request r with prefill tokens P_r and inferred prefill steps N_r:
      ρ_r = P_r - C*(N_r - 1)      in (0, C]
      μ_r = C - ρ_r                in [0, C)

    Uniform redistribution assumes the missing mass μ_r is spread uniformly over
    the trace-visible prefill interval [t_{r,s}, t_{r,e}):

      c_r(t) = μ_r / (t_{r,e} - t_{r,s})  on [t_{r,s}, t_{r,e})      [tokens/sec]
      c(t)   = Σ_r c_r(t)

    Code naming note (important)
    ----------------------------
    The paper denotes the correction signal by c(t) (tokens/sec).
    This implementation returns it as corr_rate (tokens/sec).

    The estimator uses the correction only through integrated form:
      ∫ c(t) dt  (tokens)

    Inputs
    ------
    df : pd.DataFrame
        Full phase table; used for column presence checks.
    grid : np.ndarray, shape (M,)
        Global time grid.
    chunk_size : int
        C, tokens per prefill step.
    prefill_df : pd.DataFrame
        Prefill-only subset (one row per request, enforced by caller).
        Interprets:
          - prefill_tokens_col as P_r
          - n_steps_col as N_r
          - [t_start_col, t_end_col) as [t_{r,s}, t_{r,e})

    Outputs
    -------
    corr_rate : np.ndarray, shape (S=M-1,), float64
        Aggregate correction signal c(t) on each segment, units: tokens/second.
    I_corr : np.ndarray, shape (M,), float64
        Prefix integral of corr_rate over time, units: tokens.

    Unit reminder (critical)
    ------------------------
    - p_pf_full is tokens/step ⇒ ∫ p_pf_full dt has units (tokens/step)*sec
    - c(t) is tokens/sec     ⇒ ∫ c(t) dt has units tokens

    In the exposure computation A_pf_i, the integrated correction (tokens) is
    subtracted directly and MUST NOT be multiplied by λ̂_i.
    """

    corr_rate = np.zeros(grid.size - 1, dtype=np.float64)

    if prefill_df.empty:
        I_corr = _prefix_integral(grid, corr_rate)
        return corr_rate, I_corr

    _require_columns(df, [prefill_tokens_col])

    rs = prefill_df[t_start_col].to_numpy(dtype=np.float64)
    re = prefill_df[t_end_col].to_numpy(dtype=np.float64)
    P = prefill_df[prefill_tokens_col].astype(float).to_numpy()
    N = prefill_df[n_steps_col].astype(float).to_numpy()

    mu = _prefill_missing_mass_mu(P, N, chunk_size=chunk_size)

    dur = re - rs
    if np.any(dur <= 0):
        raise ValueError("Prefill interval with non-positive duration found; cannot compute uniform correction.")

    rate = mu / dur  # tokens/second

    delta = np.zeros(grid.size, dtype=np.float64)
    s_idx = np.searchsorted(grid, rs, side="left")
    e_idx = np.searchsorted(grid, re, side="left")

    np.add.at(delta, s_idx, rate)
    np.add.at(delta, e_idx, -rate)

    corr_rate = np.cumsum(delta)[:-1]
    I_corr = _prefix_integral(grid, corr_rate)
    return corr_rate, I_corr

def _build_end_partial_chunk_correction_rate(
    df: pd.DataFrame,
    *,
    grid: np.ndarray,
    chunk_size: int,
    prefill_df: pd.DataFrame,
    t_start_col: str,
    t_end_col: str,
    prefill_tokens_col: str,
    n_steps_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the baseline *end-localized* prefill partial-chunk correction signal c(t).

    Paper mapping (End-localized correction)
    ----------------------------------------
    For each request r with prefill tokens P_r and inferred prefill steps N_r:
      ρ_r = P_r - C*(N_r - 1)      in (0, C]
      μ_r = C - ρ_r                in [0, C)

    End-localized correction assumes the final (possibly partial) prefill step
    completes near the trace-visible prefill end time t_{r,e}. We therefore assign
    the entire missing mass μ_r to the single trace-induced grid segment that contains
    t_{r,e}:

      If t_{r,e} ∈ [grid[j], grid[j+1]), then on that segment:
        c_r(t) = μ_r / (grid[j+1] - grid[j])   [tokens/sec]
      and c(t) = Σ_r c_r(t).

    Code naming note (important)
    ----------------------------
    The paper denotes the correction signal by c(t) (tokens/sec).
    This implementation returns it as corr_rate (tokens/sec).

    The estimator uses the correction only through integrated form:
      ∫ c(t) dt  (tokens)

    Outputs
    -------
    corr_rate : np.ndarray, shape (S=M-1,), float64
        Aggregate correction signal c(t) on each segment, units: tokens/second.
    I_corr : np.ndarray, shape (M,), float64
        Prefix integral of corr_rate over time, units: tokens.

    Unit reminder (critical)
    ------------------------
    In A_pf_i, the integrated correction (tokens) is subtracted directly and MUST
    NOT be multiplied by λ̂_i.

    Edge cases
    ----------
    If t_end equals the last grid boundary grid[-1], we clamp to the final segment
    index S-1 to preserve the intended “segment containing t_end” behavior under
    half-open segment conventions.
    """

    corr_rate = np.zeros(grid.size - 1, dtype=np.float64)

    if prefill_df.empty:
        I_corr = _prefix_integral(grid, corr_rate)
        return corr_rate, I_corr

    _require_columns(df, [prefill_tokens_col])

    # Pull per-request quantities
    te = prefill_df[t_end_col].to_numpy(dtype=np.float64)
    P = prefill_df[prefill_tokens_col].astype(float).to_numpy()
    N = prefill_df[n_steps_col].astype(float).to_numpy()

    mu = _prefill_missing_mass_mu(P, N, chunk_size=chunk_size)

    # Identify the grid segment containing t_end.
    # Using half-open segments [grid[j], grid[j+1]), we compute:
    #   j = rightmost grid index <= t_end, then clamp to [0, S-1].
    S = grid.size - 1
    j_end = np.searchsorted(grid, te, side="right") - 1
    j_end = np.clip(j_end, 0, S - 1).astype(np.int64)

    # Segment durations for rate conversion (tokens -> tokens/sec)
    dt = grid[1:] - grid[:-1]  # shape (S,)
    if np.any(dt <= 0):
        raise ValueError("Non-positive grid segment duration found; cannot compute end-localized correction.")

    # Each request contributes μ_r tokens to exactly one segment; convert to rate.
    rate = mu / dt[j_end]  # tokens/second (per request, on its end segment)

    # Accumulate per-segment rates. (No need for a delta sweep: each request hits one segment.)
    np.add.at(corr_rate, j_end, rate)

    I_corr = _prefix_integral(grid, corr_rate)
    return corr_rate, I_corr


PartialChunkCorrectionMode = Literal["uniform", "end", "none"]


def _build_partial_chunk_correction_rate(
    df: pd.DataFrame,
    *,
    grid: np.ndarray,
    chunk_size: int,
    prefill_df: pd.DataFrame,
    t_start_col: str,
    t_end_col: str,
    prefill_tokens_col: str,
    n_steps_col: str,
    mode: PartialChunkCorrectionMode = "end",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch wrapper for partial-chunk correction modes.

    This function exists to:
      (i) support a baseline ablation ("none"),
      (ii) support the paper-described alternatives ("uniform", "end").

    Modes
    -----
    - mode="uniform":
        Uniformly redistributes missing mass μ_r over the entire prefill interval.
        Calls _build_uniform_partial_chunk_correction_rate.

    - mode="end" (default):
        Assigns the entire missing mass μ_r to the grid segment containing the
        prefill end timestamp t_{r,e}. Calls _build_end_partial_chunk_correction_rate.

    - mode="none":
        No correction is applied (baseline ablation). Returns an all-zero correction
        rate and corresponding prefix integral.

    Inputs / Outputs
    ----------------
    Same as _build_uniform_partial_chunk_correction_rate for compatibility.
    Returns (corr_rate, I_corr) with:
      - corr_rate shape (S=M-1,), units tokens/second
      - I_corr shape (M,), units tokens
    """
    if mode == "uniform":
        return _build_uniform_partial_chunk_correction_rate(
            df,
            grid=grid,
            chunk_size=chunk_size,
            prefill_df=prefill_df,
            t_start_col=t_start_col,
            t_end_col=t_end_col,
            prefill_tokens_col=prefill_tokens_col,
            n_steps_col=n_steps_col,
        )

    if mode == "end":
        return _build_end_partial_chunk_correction_rate(
            df,
            grid=grid,
            chunk_size=chunk_size,
            prefill_df=prefill_df,
            t_start_col=t_start_col,
            t_end_col=t_end_col,
            prefill_tokens_col=prefill_tokens_col,
            n_steps_col=n_steps_col,
        )

    if mode == "none":
        corr_rate = np.zeros(grid.size - 1, dtype=np.float64)
        I_corr = _prefix_integral(grid, corr_rate)
        return corr_rate, I_corr

    raise ValueError(f"Unknown correction mode: {mode}. Expected one of: 'uniform', 'end', 'none'.")


def _compute_exposures(
    df: pd.DataFrame,
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    grid: np.ndarray,
    lam: np.ndarray,
    p_pf_full: np.ndarray,
    p_dec: np.ndarray,
    I_pf_full: np.ndarray,
    I_dec: np.ndarray,
    corr_rate: np.ndarray,
    I_corr: np.ndarray,
) -> pd.DataFrame:
    """
    Compute baseline integrated token exposures A_pf_i and A_dec_i for each phase instance i.

    Paper mapping (Baseline integrated exposures)
    ---------------------------------------------
    Pressures are defined as tokens/step functions of wall-clock time t on the
    trace-induced grid. The baseline converts time-integrated pressures into token
    exposures using the phase-local step-density proxy λ̂_i (steps/sec).

    The paper defines:
      A_pf_i  = λ̂_i * ∫_{t_{i,s}}^{t_{i,e}} p_pf_full(t) dt  -  ∫_{t_{i,s}}^{t_{i,e}} c(t) dt
      A_dec_i = λ̂_i * ∫_{t_{i,s}}^{t_{i,e}} p_dec(t) dt

    Code naming notes (important)
    -----------------------------
    - The paper denotes phase duration by ℓ_i. This implementation stores ℓ_i as df["T_i"].
    - The paper denotes the prefill correction signal by c(t) (tokens/sec).
      This implementation stores c(t) as corr_rate (tokens/sec).

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain:
          - df["lambda_hat"] : λ̂_i (steps/second)
        This function adds:
          - df["A_pf_i"], df["A_dec_i"] : token exposures (tokens)
    starts, ends : np.ndarray, shape (n,)
        Phase boundaries aligned with df rows.
    grid : np.ndarray, shape (M,)
        Global time grid.
    lam : np.ndarray, shape (n,)
        λ̂_i per phase, steps/second.
    p_pf_full, p_dec : np.ndarray, shape (S=M-1,)
        Pressures per segment, tokens/step.
    I_pf_full, I_dec : np.ndarray, shape (M,)
        Prefix integrals of pressures, units: (tokens/step)*seconds.
    corr_rate : np.ndarray, shape (S=M-1,)
        Prefill correction signal c(t) per segment, tokens/second.
    I_corr : np.ndarray, shape (M,)
        Prefix integral of corr_rate, units: tokens.

    Unit reminder (do not change)
    -----------------------------
    ∫ corr_rate dt is already in TOKENS, so it is subtracted directly in A_pf_i.
    It MUST NOT be multiplied by λ̂_i.
    """

    A_pf = np.zeros(len(df), dtype=np.float64)
    A_dec = np.zeros(len(df), dtype=np.float64)

    for i in range(len(df)):
        a = float(starts[i])
        b = float(ends[i])

        int_pf_full = _integral_on_interval(grid, I_pf_full, a, b, p_pf_full)  # (tokens/step)*sec
        int_dec = _integral_on_interval(grid, I_dec, a, b, p_dec)              # (tokens/step)*sec
        corr_tokens = _integral_on_interval(grid, I_corr, a, b, corr_rate)     # tokens

        A_pf[i] = lam[i] * int_pf_full - corr_tokens
        A_dec[i] = lam[i] * int_dec

    df["A_pf_i"] = A_pf
    df["A_dec_i"] = A_dec
    return df


def _fit_nnls_betas(
    df: pd.DataFrame,
) -> tuple[LinearRegression, float, float, float, np.ndarray, np.ndarray]:
    """
    Fit non-negative least squares regression for β.

    Paper mapping:
      min_{β >= 0} Σ_i ( (β0*N_i + β1*A_pf_i + β2*A_dec_i) - T_i )^2

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Must contain numeric columns:
          - "N_i"     : steps
          - "A_pf_i"  : tokens
          - "A_dec_i" : tokens
          - "T_i"     : seconds

    Outputs
    -------
    model : sklearn.linear_model.LinearRegression
        Fitted model with fit_intercept=False and positive=True.
        Coefficients correspond to (β0, β1, β2).
    beta0, beta1, beta2 : float
        Estimated coefficients.
    X : np.ndarray, shape (n, 3)
        Design matrix:
          X[:,0] = N_i
          X[:,1] = A_pf_i
          X[:,2] = A_dec_i
    y : np.ndarray, shape (n,)
        Response vector y = T_i.

    Notes
    -----
    sklearn's LinearRegression(positive=True) performs NNLS-like constrained fitting
    for this simple setting.
    """
    X = np.column_stack(
        [
            df["N_i"].to_numpy(dtype=np.float64),
            df["A_pf_i"].to_numpy(dtype=np.float64),
            df["A_dec_i"].to_numpy(dtype=np.float64),
        ]
    )
    y = df["T_i"].to_numpy(dtype=np.float64)

    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)
    beta0, beta1, beta2 = map(float, model.coef_)
    return model, beta0, beta1, beta2, X, y


def _compute_diagnostics(
    df: pd.DataFrame,
    *,
    model: LinearRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Compute optional sanity-check diagnostics.

    Inputs
    ------
    df : pd.DataFrame, shape (n, ...)
        Used to report feature means and row count.
    model : LinearRegression
        Fitted model used to compute predictions.
    X : np.ndarray, shape (n, 3)
        Design matrix (see _fit_nnls_betas).
    y : np.ndarray, shape (n,)
        Observed durations T_i.

    Outputs
    -------
    diagnostics : Dict[str, float]
        Keys:
          - n_phases
          - r2
          - rmse_seconds
          - mean_duration_seconds
          - mean_steps
          - mean_A_pf_tokens
          - mean_A_dec_tokens

    Notes
    -----
    These are not used in estimation; they are guardrails for engineers:
    - r2/rmse: is the linear model plausibly fitting?
    - means: helps detect unit mistakes (e.g., exposures wildly off scale).
    """
    y_hat = model.predict(X)
    resid = y_hat - y
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(df))) if len(df) else float("nan")

    return {
        "n_phases": float(len(df)),
        "r2": r2,
        "rmse_seconds": rmse,
        "mean_duration_seconds": float(np.mean(y)) if len(y) else float("nan"),
        "mean_steps": float(np.mean(df["N_i"])) if len(df) else float("nan"),
        "mean_A_pf_tokens": float(np.mean(df["A_pf_i"])) if len(df) else float("nan"),
        "mean_A_dec_tokens": float(np.mean(df["A_dec_i"])) if len(df) else float("nan"),
    }


# =============================================================================
# Main estimator
# =============================================================================
def estimate_betas_baseline(
    phases: pd.DataFrame,
    *,
    chunk_size: int,
    # Column names (override if your schema differs)
    request_id_col: str = "request_id",
    phase_type_col: str = "phase_type",       # "prefill" or "decode"
    t_start_col: str = "t_start",
    t_end_col: str = "t_end",
    prefill_tokens_col: str = "prefill_tokens", # used for prefill step count + correction
    decode_tokens_col: str = "decode_tokens", # used for decode step count
    n_steps_col: str = "N_steps",             # optional; if missing, inferred
    min_duration_sec: float = 0.0,            # set >0 if you want to clamp ultra-small durations
    correction_mode: PartialChunkCorrectionMode = "end"
) -> BaselineBetaResult:
    """
    Baseline estimator (time-integrated NNLS) with configurable partial-chunk correction.

    This function is the public entrypoint. It orchestrates the full pipeline described
    at the top of this file, without exposing internal representations.

    Inputs
    ------
    phases : pd.DataFrame, shape (n, ...)
        One row per phase instance i.

        Required core columns (names are configurable):
          - request_id_col : request identifier
          - phase_type_col : "prefill" or "decode"
          - t_start_col    : phase start time t_{i,s} (seconds)
          - t_end_col      : phase end time t_{i,e} (seconds)

        If n_steps_col is NOT provided, also required:
          - prefill_tokens_col : number of prefill tokens P_i (prefill only)
          - decode_tokens_col : decode tokens (decode only)

        Optional:
          - n_steps_col : if present, treated as authoritative trace-inferred N_i.

    chunk_size : int
        Prefill chunk size C > 0 (tokens per prefill step).
        Used for:
          - inferring prefill steps N_i = ceil(P_i/C) when needed
          - forming full-chunk prefill pressure p_pf_full(t) = C * (#active prefill)
          - computing partial-chunk missing mass μ_r for the correction

    correction_mode : {"end", "uniform", "none"}, default="end"
        Partial-chunk correction mode for prefill chunking:
          - "end": end-localized correction (paper default). Assigns missing
            prefill mass μ_r to the grid segment containing the prefill end time.
          - "uniform": uniformly redistributes μ_r over the entire prefill interval.
          - "none": disables partial-chunk correction (baseline ablation).
          
    Column-name parameters
    ----------------------
    All *_col parameters let you map your trace schema into the expected semantics.

    min_duration_sec : float
        Optional safety clamp for T_i when forming λ̂_i. See _compute_durations_and_lambda_hat.

    Outputs
    -------
    BaselineBetaResult
        (beta0, beta1, beta2, diagnostics), where:
          - beta0 : seconds/step
          - beta1 : seconds/token (prefill)
          - beta2 : seconds/token (decode)

    Estimation meaning (paper)
    --------------------------
    This corresponds to solving Eq. (baseline_nnls) using baseline features
    from Eq. (baseline_integrated_exposures), with the specified prefill
    partial-chunk correction mode.

    Notation bridges:
      - Paper duration ℓ_i is stored as df["T_i"] in this implementation.
      - Paper correction signal c(t) (tokens/sec) is represented as corr_rate.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    # 1) Validate schema and normalize
    df = _validate_and_normalize_schema(
        phases,
        request_id_col=request_id_col,
        phase_type_col=phase_type_col,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
    )

    if df.empty:
        raise ValueError("No phase rows provided after validation.")

    # 2) Get / infer step counts N_i
    df = _infer_or_read_step_counts(
        df,
        chunk_size=chunk_size,
        phase_type_col=phase_type_col,
        prefill_tokens_col=prefill_tokens_col,
        decode_tokens_col=decode_tokens_col,
        n_steps_col=n_steps_col,
    )

    # 2b) Phase durations and phase-local step density proxy λ̂_i
    df = _compute_durations_and_lambda_hat(
        df,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
        min_duration_sec=min_duration_sec,
    )

    for col in ["N_i", "T_i", "lambda_hat"]:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite values in {col}.")
    if (df["N_i"] < 0).any():
        raise ValueError("Negative N_i found.")
    if (df["T_i"] <= 0).any():
        raise ValueError("Non-positive T_i found.")

    # Baseline schema assumption: at most one prefill row per request
    prefill_df = _check_baseline_prefill_uniqueness(
        df,
        request_id_col=request_id_col,
        phase_type_col=phase_type_col,
    )

    # 3) Build global time grid
    starts, ends, grid = _build_global_time_grid_from_df(df, t_start_col=t_start_col, t_end_col=t_end_col)

    # 4) Reconstruct pressures via sweep-line overlap
    p_pf_full, p_dec, I_pf_full, I_dec = _reconstruct_pressures(
        df,
        grid=grid,
        starts=starts,
        ends=ends,
        chunk_size=chunk_size,
        phase_type_col=phase_type_col,
    )

    # 5) partial-chunk correction
    corr_rate, I_corr = _build_partial_chunk_correction_rate(
        df,
        grid=grid,
        chunk_size=chunk_size,
        prefill_df=prefill_df,
        t_start_col=t_start_col,
        t_end_col=t_end_col,
        prefill_tokens_col=prefill_tokens_col,
        n_steps_col=n_steps_col,
        mode=correction_mode,
    )

    # 6) Compute exposures A_pf_i and A_dec_i
    lam = df["lambda_hat"].to_numpy(dtype=np.float64)
    df = _compute_exposures(
        df,
        starts=starts,
        ends=ends,
        grid=grid,
        lam=lam,
        p_pf_full=p_pf_full,
        p_dec=p_dec,
        I_pf_full=I_pf_full,
        I_dec=I_dec,
        corr_rate=corr_rate,
        I_corr=I_corr,
    )

    mask_bad = (~np.isfinite(df["A_pf_i"])) | (~np.isfinite(df["A_dec_i"]))
    if mask_bad.any():
        bad = df.loc[mask_bad]
        raise ValueError(
            "Non-finite exposures found (A_pf_i/A_dec_i). "
            f"Bad rows (first 5):\n{bad.head(5)}"
        )
    
    # 7) NNLS regression
    model, beta0, beta1, beta2, X, y = _fit_nnls_betas(df)

    # 8) Diagnostics
    diagnostics = _compute_diagnostics(df, model=model, X=X, y=y)

    return BaselineBetaResult(beta0=beta0, beta1=beta1, beta2=beta2, diagnostics=diagnostics)


# =============================================================================
# Minimal example
# =============================================================================
def _example() -> None:
    """
    Tiny end-to-end example.

    Creates two requests (A,B), each with:
      - one prefill phase
      - one decode phase

    This is only a smoke test demonstrating expected input schema.
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
        print(phases_path)
        models_dir = os.path.join("models", f"model_{model}_tp_{tp}")
        os.makedirs(models_dir, exist_ok=True)
        train_alpha_model(train_path, models_dir)
        res = estimate_betas_baseline(phases, chunk_size=chunk_size)
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

        print("Estimated betas (baseline):")
        print(f"  beta0 (sec/step):  {res.beta0:.6f}")
        print(f"  beta1 (sec/token): {res.beta1:.6f}")
        print(f"  beta2 (sec/token): {res.beta2:.6f}")
        print("\nDiagnostics:")
        for k, v in res.diagnostics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__": # pragma: no cover
    _example()
