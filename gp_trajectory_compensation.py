"""
Gaussian Process–based feedforward compensation for robot trajectory tracking
===========================================================================

This module learns to predict the end-effector tracking error of a robot and
pre-distorts (compensates) the commanded trajectory so that the observed motion
follows the desired path more closely — **without** touching low-level control.

Core idea
---------
Given logs from one (or more) runs:
    desired(t)   : R^3  — desired end-effector pose sequence
    commanded(t) : R^3  — pose you sent to the robot
    observed(t)  : R^3  — measured end-effector pose

we learn the axis-wise error   e(t) = observed(t) - desired(t)
with a Gaussian Process as a function of features built from
(commanded, desired, time, finite-difference velocities, …).

At use time, we compute a corrected command via fixed-point iteration:
    cmd^(k+1)(t) = desired(t) - E[e(t) | features(cmd^k(t), desired(t), …)]

API
---
    from gp_trajectory_compensation import TrajectoryCompensatorGP

    comp = TrajectoryCompensatorGP(include_vel=True, dt_hint=0.01,
                                   kernel='auto', damping=1.0)
    comp.fit(desired_xyz, commanded_xyz, observed_xyz, t_sec)
    cmd_corrected = comp.compensate(desired_xyz, t_sec, n_iter=3)

Notes
-----
* Uses scikit-learn GPs if available; otherwise falls back to a small NumPy GP.
* Features are standardized internally; targets too.
* You can add more features (e.g., joint angles, temperatures) by editing
  `_build_features`.

License: MIT
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import numpy as np

# Try scikit-learn; otherwise keep a small GP fallback
try:  # pragma: no cover
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        WhiteKernel,
        ConstantKernel as C,
        RationalQuadratic,
    )
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


# ---------------------------- Utilities ---------------------------- #

def _finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Central finite difference with forward/backward at edges.
    Accepts (T,) or (T,D); returns same shape.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
        one_dim = True
    else:
        one_dim = False
    v = np.zeros_like(arr)
    if len(arr) > 1:
        v[1:-1] = (arr[2:] - arr[:-2]) / (2.0 * dt)
        v[0] = (arr[1] - arr[0]) / dt
        v[-1] = (arr[-1] - arr[-2]) / dt
    if one_dim:
        v = v.ravel()
    return v


def _standardize_train(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-12
    return (x - mu) / std, {"mu": mu, "std": std}


def _standardize_apply(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - stats["mu"]) / stats["std"]


# ----------------------- Minimal NumPy GP fallback ----------------------- #

class _TinyGP:
    """
    A compact GP regressor with an RBF + white kernel.
    - No hyperparameter search; uses a median distance heuristic.
    - Good for small/medium datasets or quick iteration.
    """

    def __init__(self, noise: float = 1e-3, lengthscale: Optional[float] = None, signal: Optional[float] = None):
        self.noise = noise
        self.lengthscale = lengthscale
        self.signal = signal
        self.X: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self._K: Optional[np.ndarray] = None

    def _rbf(self, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
        d2 = (
            np.sum(XA ** 2, axis=1, keepdims=True)
            + np.sum(XB ** 2, axis=1)
            - 2.0 * XA @ XB.T
        )
        return (self.signal ** 2) * np.exp(-0.5 * d2 / (self.lengthscale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        n = X.shape[0]
        if self.lengthscale is None:
            # median pairwise distance heuristic on a subsample
            idx = np.random.choice(n, size=min(n, 256), replace=False)
            XA = X[idx]
            dists = np.sqrt(
                np.maximum(0.0, np.sum((XA[:, None, :] - XA[None, :, :]) ** 2, axis=2))
            )
            med = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
            self.lengthscale = float(med + 1e-6)
        if self.signal is None:
            self.signal = float(max(1e-6, np.std(y)))
        K = self._rbf(X, X) + (self.noise + 1e-6) * np.eye(n)
        self.alpha = np.linalg.solve(K, y)
        self.X = X
        self._K = K

    def predict(self, Xs: np.ndarray, return_std: bool = False):
        Xs = np.asarray(Xs)
        Ks = self._rbf(Xs, self.X)
        mean = Ks @ self.alpha
        if not return_std:
            return mean.ravel()
        # crude variance estimate (sufficient for diagnostics)
        v = np.maximum(
            1e-12,
            (self.signal ** 2)
            - np.sum(Ks * np.linalg.solve(self._K, Ks.T).T, axis=1),
        )
        return mean.ravel(), np.sqrt(v)


# ---------------------------- GP Wrapper ---------------------------- #

class _AxisModel:
    """Container for one-axis GP and its (X,y) standardization stats."""

    def __init__(self, gp: Any, x_stats: Dict[str, np.ndarray], y_stats: Dict[str, np.ndarray]):
        self.gp = gp
        self.x_stats = x_stats
        self.y_stats = y_stats


class TrajectoryCompensatorGP:
    """
    Learn axis-wise error e = observed − desired from features built from
    commanded, desired, time, (optional) velocities. Then compensate the
    commanded trajectory with a fixed-point iteration.

    Parameters
    ----------
    include_vel : bool
        Include finite-difference velocities for commanded & desired.
    dt_hint : float
        Time-step used if t_sec is None or length < 2.
    kernel : {'auto','tiny'}
        'auto' uses scikit-learn GP if available, else fallback. 'tiny' forces fallback.
    damping : float in (0,1]
        Relaxation factor in the fixed-point update. 1.0 = full step; smaller can stabilize.
    """

    def __init__(self, include_vel: bool = True, dt_hint: float = 0.01, kernel: str = "auto", damping: float = 1.0):
        self.include_vel = bool(include_vel)
        self.dt_hint = float(dt_hint)
        self.kernel = str(kernel)
        self.damping = float(damping)
        if not (0.0 < self.damping <= 1.0):
            raise ValueError("damping must be in (0,1].")
        self.models: Dict[str, _AxisModel] = {}

    # ----------------------- Feature construction ----------------------- #
    def _build_features(self, commanded_xyz: np.ndarray, desired_xyz: np.ndarray, t_sec: Optional[np.ndarray]) -> np.ndarray:
        commanded_xyz = np.asarray(commanded_xyz)
        desired_xyz = np.asarray(desired_xyz)
        if commanded_xyz.shape != desired_xyz.shape:
            raise ValueError("commanded_xyz and desired_xyz must have same shape (T,3)")
        if commanded_xyz.ndim != 2 or commanded_xyz.shape[1] != 3:
            raise ValueError("Inputs must be shaped (T,3)")
        T = commanded_xyz.shape[0]
        if t_sec is None or len(np.atleast_1d(t_sec)) < 2:
            dt = self.dt_hint
            t = np.arange(T) * dt
        else:
            t = np.asarray(t_sec).reshape(-1)
            if t.shape[0] != T:
                raise ValueError("t_sec must have length T")
            dt = float(np.median(np.diff(t))) if T > 1 else self.dt_hint
        feats = [commanded_xyz, desired_xyz, t.reshape(-1, 1)]
        if self.include_vel:
            feats += [
                _finite_diff(commanded_xyz, dt),
                _finite_diff(desired_xyz, dt),
            ]
        X = np.concatenate(feats, axis=1)  # (T, F)
        return X

    def _make_gp(self):
        if self.kernel == "tiny" or not _HAVE_SKLEARN:
            return _TinyGP(noise=1e-3)
        # scikit-learn GP: (C * (RBF + RQ)) + White; no restarts for speed
        k = (
            C(1.0, (1e-3, 1e3))
            * (RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
               + RationalQuadratic(alpha=1.0, length_scale=1.0))
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0, random_state=0)

    # ----------------------------- Fit --------------------------------- #
    def fit(
        self,
        desired_xyz: np.ndarray,
        commanded_xyz: np.ndarray,
        observed_xyz: np.ndarray,
        t_sec: Optional[np.ndarray] = None,
    ) -> None:
        """Fit three independent GPs (x,y,z) on a single dataset (T x 3).

        Targets are axis-wise errors: observed - desired.
        You can call this repeatedly on batches if you want to re-train.
        """
        desired_xyz = np.asarray(desired_xyz)
        commanded_xyz = np.asarray(commanded_xyz)
        observed_xyz = np.asarray(observed_xyz)
        if not (desired_xyz.shape == commanded_xyz.shape == observed_xyz.shape):
            raise ValueError("desired_xyz, commanded_xyz, observed_xyz must have the same shape (T,3)")
        if desired_xyz.ndim != 2 or desired_xyz.shape[1] != 3:
            raise ValueError("Inputs must be shaped (T,3)")

        X_raw = self._build_features(commanded_xyz, desired_xyz, t_sec)
        E = observed_xyz - desired_xyz  # (T,3)

        # Standardize once and reuse for all axes
        Xs, x_stats = _standardize_train(X_raw)

        for i, axis in enumerate(("x", "y", "z")):
            ys, y_stats = _standardize_train(E[:, i : i + 1])
            gp = self._make_gp()
            gp.fit(Xs, ys.ravel())
            self.models[axis] = _AxisModel(gp=gp, x_stats=x_stats, y_stats=y_stats)

    # -------------------------- Inference ------------------------------ #
    def predict_error(self, commanded_xyz: np.ndarray, desired_xyz: np.ndarray, t_sec: Optional[np.ndarray]) -> np.ndarray:
        """Predict axis-wise tracking error (T x 3) for given (cmd, des, t)."""
        if not self.models:
            raise RuntimeError("Model not fitted. Call fit(...) first.")
        X = self._build_features(commanded_xyz, desired_xyz, t_sec)
        errs = np.zeros_like(commanded_xyz, dtype=float)
        for i, axis in enumerate(("x", "y", "z")):
            mdl = self.models[axis]
            Xs = _standardize_apply(X, mdl.x_stats)
            mu = mdl.gp.predict(Xs).reshape(-1, 1)
            mu = mu * mdl.y_stats["std"] + mdl.y_stats["mu"]  # unstandardize
            errs[:, i : i + 1] = mu
        return errs

    def compensate(
        self,
        desired_xyz: np.ndarray,
        t_sec: Optional[np.ndarray] = None,
        initial_cmd: Optional[np.ndarray] = None,
        n_iter: int = 3,
    ) -> np.ndarray:
        """Compute compensated command sequence to track desired_xyz.

        Fixed-point update with optional damping:
            cmd <- desired - damping * predicted_error(cmd)
        """
        desired_xyz = np.asarray(desired_xyz)
        if desired_xyz.ndim != 2 or desired_xyz.shape[1] != 3:
            raise ValueError("desired_xyz must be shaped (T,3)")
        cmd = desired_xyz.copy() if initial_cmd is None else np.asarray(initial_cmd).copy()
        if cmd.shape != desired_xyz.shape:
            raise ValueError("initial_cmd must be (T,3) and match desired_xyz")
        for _ in range(int(n_iter)):
            pred_err = self.predict_error(cmd, desired_xyz, t_sec)
            cmd = desired_xyz - self.damping * pred_err
        return cmd


# ------------------------------- Demo -------------------------------- #

def _simulate_robot(commanded_xyz: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """A tiny toy plant: cross-axis coupling + 1st-order lag + bias + noise.
    This is *only* for the __main__ demo to visualize the concept.
    """
    commanded_xyz = np.asarray(commanded_xyz)
    T = commanded_xyz.shape[0]
    cmd = commanded_xyz.copy()
    A = np.array(
        [
            [0.00, 0.04, -0.02],
            [-0.03, 0.00, 0.03],
            [0.02, -0.01, 0.00],
        ]
    )
    coupled = cmd @ (A.T)
    alpha = 0.2  # discrete-time lag factor
    y = np.zeros_like(cmd)
    for t in range(T):
        y[t] = (1 - alpha) * (cmd[t] + coupled[t]) + (alpha) * (y[t - 1] if t > 0 else 0.0)
    bias = np.array([0.10, -0.07, 0.03])
    noise = 0.0035 * np.random.randn(T, 3)
    return y + bias + noise


def _demo(T: int = 300, dt: float = 0.01, seed: int = 0, kernel: str = "tiny"):
    np.random.seed(seed)
    t = np.arange(T) * dt
    desired = np.zeros((T, 3))
    desired[:, 0] = np.linspace(0.0, 1.0, T)
    desired[:, 1] = 0.0
    desired[:, 2] = 0.0

    cmd0 = desired.copy()
    observed = _simulate_robot(cmd0, dt=dt)

    comp = TrajectoryCompensatorGP(include_vel=True, dt_hint=dt, kernel=kernel, damping=1.0)
    comp.fit(desired, cmd0, observed, t_sec=t)

    cmd_corr = comp.compensate(desired, t_sec=t, n_iter=3)
    observed_corr = _simulate_robot(cmd_corr, dt=dt)

    err0 = observed - desired
    err1 = observed_corr - desired
    rms0 = np.sqrt(np.mean(err0 ** 2, axis=0))
    rms1 = np.sqrt(np.mean(err1 ** 2, axis=0))
    return {
        "rms_before": rms0,
        "rms_after": rms1,
        "improvement_%": (rms0 - rms1) / np.maximum(1e-12, rms0) * 100.0,
    }


if __name__ == "__main__":  # pragma: no cover
    # Lightweight self-test (no plotting to keep this file GUI-free)
    stats = _demo(T=240, dt=0.01, seed=1, kernel="tiny")
    np.set_printoptions(precision=4, suppress=True)
    print("RMS before:", stats["rms_before"])
    print("RMS after :", stats["rms_after"])
    print("Improvement %:", stats["improvement_%"])
