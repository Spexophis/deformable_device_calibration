# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Linear Quadratic Gaussian (LQG) Control for Adaptive Optics
"""

import logging
from typing import Optional, Tuple, Dict

import numpy as np
from scipy.linalg import solve_discrete_are, svd


def setup_default_logger():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    return logging.getLogger(__name__)


class LQGController:
    """
    LQG controller for adaptive optics wavefront correction.

    State-space model (discrete-time):
        x[k+1] = A · x[k] + B · u[k] + w[k]     (state evolution)
        z[k]   = C · x[k] + D · u[k] + v[k]     (measurement)

    where:
        x = state vector (wavefront modes or actuator-space)
        u = actuator commands
        z = WFS slope measurements
        w ~ N(0, Q) = process noise (turbulence evolution)
        v ~ N(0, R) = measurement noise (WFS noise)
    """

    def __init__(
            self,
            A: np.ndarray,
            B: np.ndarray,
            C: np.ndarray,
            D: Optional[np.ndarray] = None,
            Q: Optional[np.ndarray] = None,
            R: Optional[np.ndarray] = None,
            Q_lqr: Optional[np.ndarray] = None,
            R_lqr: Optional[np.ndarray] = None,
            logger: Optional[logging.Logger] = None,
    ):
        """
        Parameters
        ----------
        A : (n_states, n_states) — state transition matrix
        B : (n_states, n_inputs) — control input matrix
        C : (n_outputs, n_states) — measurement/observation matrix
        D : (n_outputs, n_inputs) — feedthrough (usually zero for AO)
        Q : (n_states, n_states) — process noise covariance
        R : (n_outputs, n_outputs) — measurement noise covariance
        Q_lqr : (n_states, n_states) — state cost for LQR
        R_lqr : (n_inputs, n_inputs) — control effort cost for LQR
        """
        self.log = logger or setup_default_logger()

        # Store dimensions
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1]
        self.n_outputs = C.shape[0]

        # Validate dimensions
        assert A.shape == (self.n_states, self.n_states), \
            f"A shape {A.shape} != ({self.n_states}, {self.n_states})"
        assert B.shape == (self.n_states, self.n_inputs), \
            f"B shape {B.shape} != ({self.n_states}, {self.n_inputs})"
        assert C.shape == (self.n_outputs, self.n_states), \
            f"C shape {C.shape} != ({self.n_outputs}, {self.n_states})"

        # System matrices
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.D = D.copy() if D is not None else np.zeros((self.n_outputs, self.n_inputs))

        # Noise covariances
        self.Q = Q.copy() if Q is not None else np.eye(self.n_states)
        self.R = R.copy() if R is not None else np.eye(self.n_outputs)

        # LQR cost matrices
        self.Q_lqr = Q_lqr.copy() if Q_lqr is not None else np.eye(self.n_states)
        self.R_lqr = R_lqr.copy() if R_lqr is not None else np.eye(self.n_inputs)

        # State estimate and covariance
        self.x = np.zeros((self.n_states, 1))
        self.P = np.eye(self.n_states)

        # Desired state (usually zero = flat wavefront)
        self.x_target = np.zeros((self.n_states, 1))

        # Compute gains
        self.K_lqr = self._compute_lqr_gain()
        self.K_kalman_ss = None  # computed on demand

        # Diagnostics
        self.diagnostics = {
            "innovations": [],
            "actuator_rms": [],
            "state_rms": [],
            "step": 0,
        }

    # ══════════════════════════════════════════════════════════════════
    #  CONSTRUCTORS
    # ══════════════════════════════════════════════════════════════════

    @classmethod
    def from_interaction_matrix(
            cls,
            interaction_matrix: np.ndarray,
            n_modes: Optional[int] = None,
            svd_threshold: float = 0.01,
            process_noise: float = 0.1,
            measurement_noise: float = 1.0,
            temporal_decay: float = 0.99,
            lqr_state_weight: float = 1.0,
            lqr_control_weight: float = 0.01,
    ) -> "LQGController":
        """
        Build an LQG controller from a measured DM interaction matrix.

        Parameters
        ----------
        interaction_matrix : (n_slopes, n_actuators)
            Measured slopes per actuator poke. This IS your C·B product.
        n_modes : int or None
            Number of SVD modes to control. If None, use svd_threshold.
        svd_threshold : float
            Keep singular values above this fraction of the maximum.
        process_noise : float
            Diagonal process noise scaling (turbulence strength).
        measurement_noise : float
            Diagonal measurement noise scaling (WFS noise level).
        temporal_decay : float
            Eigenvalue of A. 1.0 = static, <1.0 = decaying.
            For frozen-flow turbulence at high frame rate, use ~0.99.
        lqr_state_weight : float
            Weight on wavefront error in the LQR cost.
        lqr_control_weight : float
            Weight on actuator effort in the LQR cost.
        """
        IM = interaction_matrix  # (n_slopes, n_actuators)
        n_slopes, n_actuators = IM.shape

        # SVD of interaction matrix to find controllable modes
        U, s, Vt = np.linalg.svd(IM, full_matrices=False)

        # Select modes to control
        if n_modes is None:
            n_modes = int(np.sum(s > svd_threshold * s[0]))
        n_modes = min(n_modes, len(s))

        # Truncated SVD
        U_k = U[:, :n_modes]  # (n_slopes, n_modes)
        s_k = s[:n_modes]  # (n_modes,)
        Vt_k = Vt[:n_modes, :]  # (n_modes, n_actuators)

        # State-space model in modal coordinates:
        #   state x = modal amplitudes (n_modes)
        #   input u = actuator commands projected to modes
        #   output z = slope measurements
        #
        # B maps actuator commands to state change: B = Vt_k (project to modes)
        # C maps states to slopes: C = U_k · diag(s_k) (modes to slopes)
        # A = temporal_decay * I (simple AR(1) model for turbulence)

        n_states = n_modes
        n_inputs = n_actuators

        A = temporal_decay * np.eye(n_states)
        B = Vt_k  # (n_modes, n_actuators)
        C = U_k @ np.diag(s_k)  # (n_slopes, n_modes)
        D = np.zeros((n_slopes, n_inputs))

        # Noise covariances
        Q = process_noise * np.eye(n_states)
        R = measurement_noise * np.eye(n_slopes)

        # LQR weights
        Q_lqr = lqr_state_weight * np.eye(n_states)
        R_lqr = lqr_control_weight * np.eye(n_inputs)

        ctrl = cls(A, B, C, D, Q, R, Q_lqr, R_lqr)
        ctrl._IM = IM
        ctrl._U_k = U_k
        ctrl._s_k = s_k
        ctrl._Vt_k = Vt_k
        ctrl.log.info(
            f"LQG controller initialized: {n_states} modes, "
            f"{n_inputs} actuators, {n_slopes} slopes"
        )
        return ctrl

    @classmethod
    def from_calibration_file(cls, filepath: str) -> "LQGController":
        """Load system matrices from a .npz calibration file."""
        data = np.load(filepath)
        kwargs = {
            "A": data["A"], "B": data["B"], "C": data["C"],
        }
        for key in ["D", "Q", "R", "Q_lqr", "R_lqr"]:
            if key in data:
                kwargs[key] = data[key]
        return cls(**kwargs)

    def save_calibration(self, filepath: str):
        """Save system matrices and gains to a .npz file."""
        np.savez(
            filepath,
            A=self.A, B=self.B, C=self.C, D=self.D,
            Q=self.Q, R=self.R,
            Q_lqr=self.Q_lqr, R_lqr=self.R_lqr,
            K_lqr=self.K_lqr,
        )

    # ══════════════════════════════════════════════════════════════════
    #  LQR GAIN (Optimization #1: discrete ARE)
    # ══════════════════════════════════════════════════════════════════

    def _compute_lqr_gain(self) -> np.ndarray:
        """
        Compute the optimal LQR gain using the DISCRETE Algebraic Riccati Equation.

        OPTIMIZATION #1: The original code used solve_continuous_are(), but
        the entire control loop is discrete-time (frame-by-frame). Using the
        continuous ARE gives the wrong gain for a discrete system.

        The DARE solves:
            P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q

        Then: K = (R + B^T P B)^{-1} B^T P A
        """
        try:
            P = solve_discrete_are(self.A, self.B, self.Q_lqr, self.R_lqr)
            BtP = self.B.T @ P
            K = np.linalg.solve(self.R_lqr + BtP @ self.B, BtP @ self.A)
            self.log.info(f"LQR gain computed. Spectral radius of (A-BK): "
                          f"{np.max(np.abs(np.linalg.eigvals(self.A - self.B @ K))):.4f}")
            return K
        except np.linalg.LinAlgError as e:
            self.log.warning(f"DARE failed ({e}), falling back to pseudo-inverse gain.")
            # Fallback: simple least-squares inverse
            return np.linalg.lstsq(self.B, np.eye(self.n_states), rcond=None)[0]

    # ══════════════════════════════════════════════════════════════════
    #  KALMAN FILTER (Optimizations #2, #3, #4)
    # ══════════════════════════════════════════════════════════════════

    def kalman_predict(self, u: Optional[np.ndarray] = None):
        """
        Kalman prediction step.

            x_pre = A · x + B · u
            P_pre = A · P · A^T + Q
        """
        if u is not None:
            self.x = self.A @ self.x + self.B @ u
        else:
            self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def kalman_update(self, z: np.ndarray) -> np.ndarray:
        """
        Kalman update step with numerical stability improvements.

        OPTIMIZATION #2: Joseph stabilized form for covariance update.
        OPTIMIZATION #3: np.linalg.solve() instead of explicit inverse.

        Returns the innovation vector y.
        """
        # Innovation
        y = z - self.C @ self.x

        # Innovation covariance
        S = self.C @ self.P @ self.C.T + self.R

        # Kalman gain: K = P · C^T · S^{-1}
        # Computed as: solve(S^T, (P · C^T)^T)^T — avoids explicit inverse
        # OPTIMIZATION #3
        K = np.linalg.solve(S.T, (self.P @ self.C.T).T).T

        # State update
        self.x = self.x + K @ y

        # OPTIMIZATION #2: Joseph stabilized covariance update
        # P = (I - K·C) · P · (I - K·C)^T + K · R · K^T
        # This guarantees P remains symmetric positive-definite,
        # unlike the simple form P = (I - K·C)·P which drifts over time.
        IKC = np.eye(self.n_states) - K @ self.C
        self.P = IKC @ self.P @ IKC.T + K @ self.R @ K.T

        return y

    def compute_steady_state_kalman_gain(self):
        """
        OPTIMIZATION #4: Pre-compute the steady-state Kalman gain.

        For a time-invariant system, the Kalman gain converges to a
        constant matrix. Computing it once saves O(n³) per frame.

        After calling this, kalman_update_steady_state() uses only
        matrix-vector multiplies — O(n²) per frame.
        """
        try:
            # Solve the DARE for the Kalman filter
            # (dual problem: A^T, C^T, Q, R)
            P_ss = solve_discrete_are(self.A.T, self.C.T, self.Q, self.R)
            S = self.C @ P_ss @ self.C.T + self.R
            self.K_kalman_ss = np.linalg.solve(S.T, (P_ss @ self.C.T).T).T
            self.P = P_ss
            self.log.info("Steady-state Kalman gain computed.")
        except np.linalg.LinAlgError as e:
            self.log.warning(f"Steady-state DARE failed ({e}), using iterative Kalman.")
            self.K_kalman_ss = None

    def kalman_update_steady_state(self, z: np.ndarray) -> np.ndarray:
        """
        Fast Kalman update using pre-computed steady-state gain.
        Only matrix-vector multiplies — O(n²) instead of O(n³).
        """
        y = z - self.C @ self.x
        self.x = self.x + self.K_kalman_ss @ y
        return y

    # ══════════════════════════════════════════════════════════════════
    #  MAIN CONTROL INTERFACE
    # ══════════════════════════════════════════════════════════════════

    def compute_control(
            self,
            measurement: np.ndarray,
            previous_command: Optional[np.ndarray] = None,
            use_steady_state: bool = True,
    ) -> np.ndarray:
        """
        Full LQG control step: estimate state, compute optimal command.

        Parameters
        ----------
        measurement : (n_outputs, 1) or (n_outputs,)
            Current WFS slope measurement.
        previous_command : (n_inputs, 1) or None
            The command sent at the previous step (for prediction).
        use_steady_state : bool
            If True and steady-state gain is available, use it.

        Returns
        -------
        u : (n_inputs, 1) — actuator command to send to DM.
        """
        z = np.atleast_2d(measurement)
        if z.shape[0] == 1 and z.shape[1] > 1:
            z = z.T  # ensure column vector

        # --- Predict ---
        self.kalman_predict(u=previous_command)

        # --- Update ---
        if use_steady_state and self.K_kalman_ss is not None:
            y = self.kalman_update_steady_state(z)
        else:
            y = self.kalman_update(z)

        # --- Optimal control ---
        u = -self.K_lqr @ (self.x - self.x_target)

        # --- Diagnostics ---
        self.diagnostics["innovations"].append(y.ravel())
        self.diagnostics["actuator_rms"].append(np.sqrt(np.mean(u ** 2)))
        self.diagnostics["state_rms"].append(np.sqrt(np.mean(self.x ** 2)))
        self.diagnostics["step"] += 1

        return u

    # ══════════════════════════════════════════════════════════════════
    #  ADAPTIVE NOISE ESTIMATION (Optimization #6)
    # ══════════════════════════════════════════════════════════════════

    def update_noise_covariances(self, window_size: int = 50, alpha: float = 0.1):
        """
        OPTIMIZATION #6: Innovation-based adaptive noise estimation (Mehra 1970).

        Instead of the original heuristic (Q = α·I + (1-α)·Q), this uses
        the innovation sequence to estimate both Q and R from data.

        The innovation covariance should equal:
            E[y·y^T] = C · P_pre · C^T + R

        From this relationship, we can extract updated R and Q estimates.

        Parameters
        ----------
        window_size : int
            Number of recent innovations to use.
        alpha : float
            Exponential smoothing factor for stability (0 = no update, 1 = full).
        """
        innovations = self.diagnostics["innovations"]
        if len(innovations) < window_size:
            return

        # Sample innovation covariance
        Y = np.array(innovations[-window_size:])  # (window, n_outputs)
        C_yy = (Y.T @ Y) / window_size  # (n_outputs, n_outputs)

        # Symmetrize
        C_yy = 0.5 * (C_yy + C_yy.T)

        # Estimated R from innovation covariance
        # C_yy = C · P_pre · C^T + R  →  R_est = C_yy - C · P · C^T
        CPCt = self.C @ self.P @ self.C.T
        R_est = C_yy - CPCt

        # Ensure R stays positive definite
        eigvals = np.linalg.eigvalsh(R_est)
        if eigvals.min() > 0:
            self.R = (1 - alpha) * self.R + alpha * R_est
        else:
            # Floor negative eigenvalues
            R_est = R_est + (abs(eigvals.min()) + 1e-6) * np.eye(self.n_outputs)
            self.R = (1 - alpha) * self.R + alpha * R_est

        # Estimated Q from Kalman gain and innovation covariance
        if self.K_kalman_ss is not None:
            K = self.K_kalman_ss
        else:
            S = CPCt + self.R
            K = np.linalg.solve(S.T, (self.P @ self.C.T).T).T

        Q_est = K @ C_yy @ K.T
        Q_est = 0.5 * (Q_est + Q_est.T)  # symmetrize

        eigvals_q = np.linalg.eigvalsh(Q_est)
        if eigvals_q.min() > 0:
            self.Q = (1 - alpha) * self.Q + alpha * Q_est
        else:
            Q_est = Q_est + (abs(eigvals_q.min()) + 1e-6) * np.eye(self.n_states)
            self.Q = (1 - alpha) * self.Q + alpha * Q_est

    # ══════════════════════════════════════════════════════════════════
    #  LQR WEIGHT TUNING (Optimization #8)
    # ══════════════════════════════════════════════════════════════════

    def set_lqr_weights(
            self,
            state_weight: float = 1.0,
            control_weight: float = 0.01,
            Q_lqr: Optional[np.ndarray] = None,
            R_lqr: Optional[np.ndarray] = None,
    ):
        """
        OPTIMIZATION #8: Expose LQR weight tuning with physical meaning.

        The ratio state_weight / control_weight controls aggressiveness:
          - Large ratio → aggressive correction, risk of actuator saturation
          - Small ratio → gentle correction, more residual wavefront error

        Parameters
        ----------
        state_weight : float
            Scalar weight on wavefront error (diagonal Q_lqr = w·I).
        control_weight : float
            Scalar weight on actuator effort (diagonal R_lqr = w·I).
        Q_lqr, R_lqr : arrays, optional
            Full weight matrices (override scalar weights if provided).
        """
        if Q_lqr is not None:
            self.Q_lqr = Q_lqr.copy()
        else:
            self.Q_lqr = state_weight * np.eye(self.n_states)

        if R_lqr is not None:
            self.R_lqr = R_lqr.copy()
        else:
            self.R_lqr = control_weight * np.eye(self.n_inputs)

        self.K_lqr = self._compute_lqr_gain()

    # ══════════════════════════════════════════════════════════════════
    #  LEAKY INTEGRATOR FALLBACK (Optimization #10)
    # ══════════════════════════════════════════════════════════════════

    def compute_control_integrator(
            self,
            slopes: np.ndarray,
            command_matrix: np.ndarray,
            previous_command: np.ndarray,
            gain: float = 0.5,
            leak: float = 0.01,
    ) -> np.ndarray:
        """
        OPTIMIZATION #10: Simple leaky integrator controller as fallback.

        u[k+1] = (1 - leak) · u[k] - gain · M · s[k]

        where M is the command matrix (pseudo-inverse of interaction matrix)
        and s is the slope vector. No Kalman filter or LQR — just robust,
        well-understood classical AO control.

        Use this when:
          - System model is uncertain
          - You want a baseline to compare against LQG
          - Real-time constraints are very tight

        Parameters
        ----------
        slopes : (n_slopes, 1)
        command_matrix : (n_actuators, n_slopes)
        previous_command : (n_actuators, 1)
        gain : float — loop gain (0.3–0.7 typical)
        leak : float — leak rate (0.01 typical)
        """
        s = np.atleast_2d(slopes)
        if s.shape[0] == 1:
            s = s.T

        u_prev = np.atleast_2d(previous_command)
        if u_prev.shape[0] == 1:
            u_prev = u_prev.T

        u = (1 - leak) * u_prev - gain * command_matrix @ s
        return u

    # ══════════════════════════════════════════════════════════════════
    #  SYSTEM IDENTIFICATION (Optimization #7)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def subspace_identification(
            inputs: np.ndarray,
            outputs: np.ndarray,
            n_states: int,
            n_block_rows: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        OPTIMIZATION #7: Cleaned-up N4SID subspace identification.

        Identifies (A, B, C, D) from input-output data using the
        Numerical Algorithms for Subspace State Space System
        Identification (N4SID) method.

        Parameters
        ----------
        inputs : (N, n_inputs) — input time series
        outputs : (N, n_outputs) — output time series
        n_states : int — desired state dimension
        n_block_rows : int or None — block rows in Hankel matrix
                       (default: 2 * n_states)

        Returns
        -------
        A, B, C, D : system matrices
        """
        N = inputs.shape[0]
        n_inputs = inputs.shape[1]
        n_outputs = outputs.shape[1]
        L = n_block_rows or 2 * n_states

        if N < 2 * L + 1:
            raise ValueError(
                f"Need at least {2 * L + 1} data points, got {N}. "
                f"Reduce n_states or collect more data."
            )

        n_cols = N - 2 * L + 1

        # Build block Hankel matrices
        def hankel(data, n_rows, n_cols_h):
            """Build block Hankel matrix from (N, dim) data."""
            dim = data.shape[1]
            H = np.zeros((n_rows * dim, n_cols_h))
            for i in range(n_rows):
                H[i * dim:(i + 1) * dim, :] = data[i:i + n_cols_h].T
            return H

        # Past and future Hankel matrices
        Y_past = hankel(outputs[:N], L, n_cols)
        Y_future = hankel(outputs[L:N], L, n_cols)
        U_past = hankel(inputs[:N], L, n_cols)
        U_future = hankel(inputs[L:N], L, n_cols)

        # Oblique projection: project Y_future along U_future onto [Y_past; U_past]
        W_p = np.vstack([Y_past, U_past])

        # QR decomposition approach for numerical stability
        combined = np.vstack([U_future, W_p, Y_future])
        Q_qr, R_qr = np.linalg.qr(combined.T, mode='reduced')

        n_uf = U_future.shape[0]
        n_wp = W_p.shape[0]
        n_yf = Y_future.shape[0]

        # Extract R blocks
        R22 = R_qr[n_uf:n_uf + n_wp, n_uf:n_uf + n_wp]
        R32 = R_qr[n_uf:n_uf + n_wp, n_uf + n_wp:]

        # Oblique projection
        Ob = R32.T @ np.linalg.lstsq(R22.T, np.eye(R22.shape[0]), rcond=None)[0].T

        # Not the cleanest but a simplified, robust version:
        # Direct SVD on the weighted output Hankel
        U_svd, s_svd, Vt_svd = svd(Y_future @ W_p.T, full_matrices=False)

        # Truncate to n_states
        U_n = U_svd[:, :n_states]
        S_n = np.diag(np.sqrt(s_svd[:n_states]))
        V_n = Vt_svd[:n_states, :]

        # Observability and controllability matrices
        O_matrix = U_n @ S_n  # (L*n_outputs, n_states)
        C_matrix = (S_n @ V_n)  # (n_states, L*(n_outputs+n_inputs))

        # Extract C from the first block row of O
        C_out = O_matrix[:n_outputs, :]

        # State sequence from the controllability matrix
        X_hat = C_matrix  # (n_states, n_cols)

        # Ensure compatible dimensions
        k = min(X_hat.shape[1] - 1, n_cols - 1)
        X1 = X_hat[:, :k]
        X2 = X_hat[:, 1:k + 1]
        Y1 = outputs[L:L + k].T  # (n_outputs, k)
        U1_data = inputs[L:L + k].T  # (n_inputs, k)

        # Solve for [A, B] from: X2 = A·X1 + B·U1
        XU = np.vstack([X1, U1_data])  # (n_states + n_inputs, k)
        AB = np.linalg.lstsq(XU.T, X2.T, rcond=None)[0].T
        A_out = AB[:, :n_states]
        B_out = AB[:, n_states:n_states + n_inputs]

        # Solve for [C, D] from: Y1 = C·X1 + D·U1
        CD = np.linalg.lstsq(XU.T, Y1.T, rcond=None)[0].T
        C_out2 = CD[:, :n_states]
        D_out = CD[:, n_states:n_states + n_inputs]

        # Use C from least-squares (more accurate than from observability matrix)
        C_final = C_out2

        # Check stability
        eigvals = np.linalg.eigvals(A_out)
        spectral_radius = np.max(np.abs(eigvals))
        if spectral_radius > 1.0:
            logging.warning(
                f"Identified system is UNSTABLE (spectral radius = {spectral_radius:.3f}). "
                f"Consider reducing n_states or collecting more data."
            )

        return A_out, B_out, C_final, D_out

    # ══════════════════════════════════════════════════════════════════
    #  RESET AND UTILITIES
    # ══════════════════════════════════════════════════════════════════

    def reset(self):
        """Reset state estimate and diagnostics."""
        self.x = np.zeros((self.n_states, 1))
        self.P = np.eye(self.n_states)
        self.diagnostics = {
            "innovations": [],
            "actuator_rms": [],
            "state_rms": [],
            "step": 0,
        }

    def get_diagnostics_summary(self) -> Dict:
        """Return summary statistics of the control loop."""
        d = self.diagnostics
        if d["step"] == 0:
            return {"step": 0}

        innovations = np.array(d["innovations"])
        return {
            "step": d["step"],
            "innovation_rms": np.sqrt(np.mean(innovations ** 2)),
            "innovation_rms_last10": np.sqrt(np.mean(innovations[-10:] ** 2)),
            "actuator_rms_mean": np.mean(d["actuator_rms"]),
            "actuator_rms_last": d["actuator_rms"][-1],
            "state_rms_last": d["state_rms"][-1],
            "converged": len(d["actuator_rms"]) > 20 and
                         np.std(d["actuator_rms"][-20:]) / (np.mean(d["actuator_rms"][-20:]) + 1e-10) < 0.1,
        }


# ══════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  LQG Controller for Adaptive Optics — Demo")
    print("=" * 60)

    # --- Simulate a 97-actuator DM with 600 slopes ---
    np.random.seed(42)
    n_actuators = 97
    n_slopes = 600  # ~300 valid lenslets × 2 (x, y)

    # Simulate an interaction matrix
    IM = np.random.randn(n_slopes, n_actuators) * 0.1

    # Build controller from interaction matrix
    ctrl = LQGController.from_interaction_matrix(
        interaction_matrix=IM,
        n_modes=50,
        process_noise=0.05,
        measurement_noise=0.5,
        temporal_decay=0.98,
        lqr_state_weight=1.0,
        lqr_control_weight=0.1,
    )

    # Pre-compute steady-state Kalman gain
    ctrl.compute_steady_state_kalman_gain()

    # --- Simulate a turbulent wavefront evolving over time ---
    N_steps = 200
    true_state = np.random.randn(ctrl.n_states, 1) * 0.5

    measurements = []
    commands = []
    residuals_rms = []
    prev_cmd = None

    print(f"\nRunning {N_steps}-step closed-loop simulation...")

    for step in range(N_steps):
        # Turbulence evolution (AR(1) process)
        true_state = 0.98 * true_state + 0.05 * np.random.randn(ctrl.n_states, 1)

        # Simulated WFS measurement
        z = ctrl.C @ true_state + 0.5 * np.random.randn(n_slopes, 1)

        # If DM was correcting, subtract the correction from the measurement
        if prev_cmd is not None:
            z = z + ctrl.C @ ctrl.B @ prev_cmd  # DM adds correction

        # Compute control
        u = ctrl.compute_control(z, previous_command=prev_cmd)
        prev_cmd = u

        measurements.append(z.ravel()[:5])
        commands.append(u.ravel()[:5])
        residuals_rms.append(np.sqrt(np.mean(z ** 2)))

        # Adaptive noise update every 50 steps
        if step > 0 and step % 50 == 0:
            ctrl.update_noise_covariances(window_size=40, alpha=0.05)

    # --- Results ---
    diag = ctrl.get_diagnostics_summary()
    print(f"\nDiagnostics:")
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Innovation RMS over time
    innov = np.array(ctrl.diagnostics["innovations"])
    innov_rms = np.sqrt(np.mean(innov ** 2, axis=1))
    axes[0, 0].plot(innov_rms, color="steelblue", linewidth=0.8)
    axes[0, 0].set_title("Innovation RMS per frame")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("RMS")
    axes[0, 0].grid(True, alpha=0.3)

    # Actuator RMS
    axes[0, 1].plot(ctrl.diagnostics["actuator_rms"], color="darkorange", linewidth=0.8)
    axes[0, 1].set_title("Actuator command RMS")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].grid(True, alpha=0.3)

    # State RMS
    axes[1, 0].plot(ctrl.diagnostics["state_rms"], color="seagreen", linewidth=0.8)
    axes[1, 0].set_title("Estimated state RMS")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].grid(True, alpha=0.3)

    # Measurement residual RMS
    axes[1, 1].plot(residuals_rms, color="crimson", linewidth=0.8)
    axes[1, 1].set_title("Measurement residual RMS")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("LQG Adaptive Optics Control Loop", fontsize=14)
    plt.tight_layout()
    plt.savefig("lqg_ao_demo.png", dpi=150, bbox_inches="tight")
    print("\nSaved to lqg_ao_demo.png")
    plt.show()
