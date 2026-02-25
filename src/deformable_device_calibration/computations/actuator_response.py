"""
actuator_response.py
====================
Extract a clean actuator influence function from multiple noisy
interferograms using piston-corrected coherent averaging.

Pipeline
--------
1.  Load multi-frame TIFF
2.  Auto-detect carrier frequency (Gaussian fit on FFT magnitude)
3.  Extract complex carrier per frame
4.  Estimate and correct inter-frame piston drift
5.  Coherently average corrected carriers → clean phase
6.  Unwrap, remove tilt, save result

Usage
-----
    python actuator_response.py --input frames.tif --output result/
    python actuator_response.py --input frames.tif --ref_amp -0.05 --half_n 128
    python actuator_response.py --help
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from skimage.restoration import unwrap_phase


# ======================================================================
#  Carrier detection
# ======================================================================

def detect_carrier(frames: np.ndarray,
                   dc_radius: int = 80,
                   search_radius: int = 120,
                   verbose: bool = False) -> tuple[int, int, float, float]:
    """
    Locate the 1st-order carrier frequency in the FFT by fitting a 2D
    Gaussian to the log-magnitude of the mean-frame spectrum.

    Averaging all frames before detection gives better SNR than using
    a single frame.

    Returns
    -------
    (fy_center, fx_center, sigma_y, sigma_x)
    """
    mean_frame = frames.mean(axis=0)
    imf = fftshift(fft2(mean_frame))
    mag = np.abs(imf)
    ny, nx = mag.shape
    cy, cx = ny // 2, nx // 2

    # Blank DC so argmax finds the 1st order
    mag_s = mag.copy()
    mag_s[cy - dc_radius: cy + dc_radius,
    cx - dc_radius: cx + dc_radius] = 0

    py, px = np.unravel_index(np.argmax(mag_s), mag_s.shape)

    # Gaussian fit on log-magnitude in search window
    r = search_radius
    fy0, fy1 = max(0, py - r), min(ny, py + r)
    fx0, fx1 = max(0, px - r), min(nx, px + r)
    region = mag[fy0:fy1, fx0:fx1]
    log_r = np.log1p(region)
    H_, W_ = log_r.shape
    Y_g, X_g = np.mgrid[0:H_, 0:W_]
    com = center_of_mass(log_r)

    def gauss2d(xy, amp, x0, y0, sx, sy, off):
        x, y = xy
        return off + amp * np.exp(
            -((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2))
        )

    fit_ok = False
    try:
        popt, _ = curve_fit(
            gauss2d, (X_g.ravel(), Y_g.ravel()), log_r.ravel(),
            p0=[log_r.max() - log_r.min(), com[1], com[0], 20, 20, log_r.min()],
            maxfev=10000,
        )
        fx_c_fit = int(round(popt[1] + fx0))
        fy_c_fit = int(round(popt[2] + fy0))
        sx_fit, sy_fit = abs(popt[3]), abs(popt[4])

        # Sanity-check: sigma must be >= 1 px and the fitted centre must
        # stay within the search window. A sigma of < 1 px means the fit
        # locked onto a noise spike, not the real carrier blob.
        centre_in_window = (fx0 < fx_c_fit < fx1) and (fy0 < fy_c_fit < fy1)
        sigma_ok = (sx_fit >= 1.0) and (sy_fit >= 1.0)
        fit_ok = centre_in_window and sigma_ok
    except RuntimeError:
        pass

    if fit_ok:
        fx_c, fy_c, sx, sy = fx_c_fit, fy_c_fit, sx_fit, sy_fit
        method = "Gaussian fit"
    else:
        # Fallback: power-weighted centroid — robust when the peak is
        # too sharp (noise spike) or the Gaussian fit diverged
        if verbose:
            if not fit_ok:
                warnings.warn(
                    "[carrier]  Gaussian fit unreliable "
                    f"(sigma=({sy_fit if not fit_ok and 'sy_fit' in dir() else '?'},"
                    f"{sx_fit if not fit_ok and 'sx_fit' in dir() else '?'}) px) "
                    "— falling back to power-weighted centroid"
                )
        thresh = region.max() * 0.2
        masked = np.where(region > thresh, region, 0.0)
        com_fb = center_of_mass(masked)
        fy_c = int(round(com_fb[0] + fy0))
        fx_c = int(round(com_fb[1] + fx0))
        sx = sy = float(np.sqrt(masked.sum() / (np.pi * masked.max() + 1e-12)))
        method = "centroid (Gaussian fit unreliable)"

    if verbose:
        print(f"[carrier]  fy={fy_c}  fx={fx_c}  "
              f"sigma=({sy:.1f}, {sx:.1f}) px  [{method}]")
    return fy_c, fx_c, sy, sx


def detect_and_fix_conjugate_frames(frames: np.ndarray,
                                    fy: int,
                                    fx: int,
                                    half_n: int = 128,
                                    image_shape: tuple[int, int] | None = None,
                                    conjugate_threshold: float = 50.0,
                                    verbose: bool = False,
                                    ) -> tuple[np.ndarray, list[str]]:
    """
    Detect per-frame carrier positions and fix any frames whose carrier
    jumped to the conjugate (−1) order.

    A conjugate-swapped frame has its phase negated relative to normal
    frames.  Simply using such a frame in the coherent average introduces
    a destructive artefact — the corrected phase maps look noisy and show
    a strange ripple pattern inside the pupil.

    The fix: for each conjugate frame, extract the carrier from the
    conjugate position instead of the true position, then take the complex
    conjugate of the result (which negates the phase, restoring the correct sign).

    Parameters
    ----------
    frames : (N, H, W) float array
    fy, fx : int — nominal (true) carrier position
    half_n : int — extraction half-window
    image_shape : (H, W) or None — used to compute DC centre (default: frames.shape[1:])
    conjugate_threshold : float — distance [px] below which a detection
        is considered to match the conjugate position

    Returns
    -------
    carriers : (N, 2*half_n, 2*half_n) complex128 — conjugate-fixed carriers
    statuses : list of str — 'ok' or 'conjugate' for each frame
    """
    if image_shape is None:
        image_shape = frames.shape[1:]
    ny, nx = image_shape
    cy, cx = ny // 2, nx // 2
    fy_conj = 2 * cy - fy
    fx_conj = 2 * cx - fx

    carriers = []
    statuses = []
    n_fixed = 0

    for i, frame in enumerate(frames):
        fy_i, fx_i, _, _ = detect_carrier(frame[np.newaxis], dc_radius=80, search_radius=120, verbose=False)

        dist_true = np.sqrt((fy_i - fy) ** 2 + (fx_i - fx) ** 2)
        dist_conj = np.sqrt((fy_i - fy_conj) ** 2 + (fx_i - fx_conj) ** 2)

        if dist_conj < dist_true and dist_conj < conjugate_threshold:
            # Conjugate swap: extract from conjugate position, negate phase
            cf = np.conj(extract_carrier(frame, int(round(fy_conj)),
                                         int(round(fx_conj)), half_n))
            statuses.append('conjugate')
            n_fixed += 1
        else:
            cf = extract_carrier(frame, fy, fx, half_n)
            statuses.append('ok')
        carriers.append(cf)

    if verbose and n_fixed > 0:
        bad = [i + 1 for i, s in enumerate(statuses) if s == 'conjugate']
        print(f"[conjugate-fix]  {n_fixed} frame(s) had conjugate swap "
              f"and were corrected: frames {bad}")
    elif verbose:
        print(f"[conjugate-fix]  all {len(frames)} frames have correct carrier order")

    return np.array(carriers), statuses

# ======================================================================
#  Complex carrier extraction
# ======================================================================

def extract_carrier(frame: np.ndarray,
                    fy: int, fx: int, half_n: int) -> np.ndarray:
    """
    Extract the 1st-order carrier patch from the shifted FFT.

    Returns
    -------
    cf : (2*half_n, 2*half_n) complex128 array
    """
    imf = fftshift(fft2(frame))
    return imf[fy - half_n: fy + half_n, fx - half_n: fx + half_n].copy()


def extract_all_carriers(frames: np.ndarray,
                         fy: int, fx: int, half_n: int) -> np.ndarray:
    """
    Extract complex carriers for every frame.

    Returns
    -------
    carriers : (N, 2*half_n, 2*half_n) complex128
    """
    return np.array([extract_carrier(f, fy, fx, half_n) for f in frames])


class LiveCarrierTracker:
    """
    Online robust estimator for the interferogram carrier frequency (fx, fy).

    Problem it solves
    -----------------
    In live fringe reconstruction the carrier position should be constant
    as long as the optical setup does not change.  However, single-frame
    auto-detection is noisy and can occasionally snap to the conjugate
    (−1 order) carrier — a mirror of the true position through the DC
    centre — which flips the sign of the reconstructed phase.

    This class maintains a running weighted mean of *accepted* detections
    and rejects outliers that are either:
        (a) too far from the current estimate (noise / vibration), or
        (b) close to the conjugate position (carrier order swap).

    As more frames are accepted the rejection threshold tightens, making
    the estimate progressively more stable.

    Usage
    -----
        tracker = LiveCarrierTracker(image_shape=(1324, 1324))

        # Seed with first frame (no rejection on first N=seed_frames detections)
        for frame in initial_frames:
            tracker.update(frame)

        # Live loop
        while acquiring:
            frame = camera.grab()
            fy, fx = tracker.update(frame)
            phase  = reconstruct(frame, fy, fx)

        # Inspect history
        tracker.plot_history("carrier_history.png")
    """

    def __init__(
            self,
            image_shape: tuple[int, int],
            dc_radius: int = 80,
            search_radius: int = 120,
            # Outlier rejection
            reject_radius_init: float = 30.0,  # px — loose gate during seeding
            reject_radius_min: float = 8.0,  # px — tightest gate at full confidence
            conjugate_radius: float = 20.0,  # px — flag as conjugate swap if within this
            seed_frames: int = 5,  # frames before rejection starts
            # Running average
            ema_alpha: float = 0.05,  # exponential decay for old samples
            # 0 = equal weight, 1 = only latest
            verbose: bool = False,
    ):
        """
        Parameters
        ----------
        image_shape : (H, W)
            Shape of the raw interferogram. Used to compute DC centre and
            the conjugate mirror position.
        dc_radius : int
            Pixels around DC to blank before searching for the carrier peak.
        search_radius : int
            Half-size of the Gaussian-fit search window around the peak.
        reject_radius_init : float
            Outlier gate radius [px] used during the seeding phase (first
            `seed_frames` accepted detections). Intentionally loose so the
            tracker can find the carrier even if the first few frames are noisy.
        reject_radius_min : float
            Outlier gate radius [px] at full confidence (after many accepted
            frames). The gate shrinks from reject_radius_init → reject_radius_min
            as sqrt(n_accepted / seed_frames), capped at reject_radius_min.
        conjugate_radius : float
            If a new detection is within this radius of the conjugate position
            (mirror of the current estimate through DC) it is flagged as a
            carrier order swap rather than a generic outlier.
        seed_frames : int
            Number of accepted frames needed before the estimate is considered
            "seeded" and tighter rejection kicks in.
        ema_alpha : float in [0, 1)
            Exponential moving average decay applied to each accepted sample's
            weight.  alpha=0 gives equal weights (simple mean).  alpha=0.05
            gives mild recency weighting — good for slow optical drift.
        verbose : bool
            Print a one-line summary for each update() call.
        """
        self.ny, self.nx = image_shape
        self.cy, self.cx = self.ny // 2, self.nx // 2
        self.dc_radius = dc_radius
        self.search_radius = search_radius
        self.reject_r_init = reject_radius_init
        self.reject_r_min = reject_radius_min
        self.conjugate_radius = conjugate_radius
        self.seed_frames = seed_frames
        self.ema_alpha = ema_alpha
        self.verbose = verbose

        # State
        self._fy_est: float | None = None  # current best estimate
        self._fx_est: float | None = None
        self._weight_sum: float = 0.0

        # History (one entry per call to update(), accepted or not)
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def fy(self) -> int | None:
        """Current best-estimate fy (rounded to nearest pixel)."""
        return int(round(self._fy_est)) if self._fy_est is not None else None

    @property
    def fx(self) -> int | None:
        """Current best-estimate fx (rounded to nearest pixel)."""
        return int(round(self._fx_est)) if self._fx_est is not None else None

    @property
    def n_accepted(self) -> int:
        return sum(1 for h in self.history if h["accepted"])

    @property
    def n_rejected(self) -> int:
        return sum(1 for h in self.history if not h["accepted"])

    @property
    def is_seeded(self) -> bool:
        return self.n_accepted >= self.seed_frames

    @property
    def confidence(self) -> float:
        """
        Scalar in [0, 1].  Grows as sqrt(n_accepted / seed_frames),
        capped at 1.  Indicates how much to trust the current estimate.
        """
        return min(1.0, np.sqrt(self.n_accepted / max(self.seed_frames, 1)))

    @property
    def reject_radius(self) -> float:
        """Current outlier gate radius [px], shrinking with confidence."""
        r = self.reject_r_init - (self.reject_r_init - self.reject_r_min) * self.confidence
        return max(r, self.reject_r_min)

    def update(self, frame: np.ndarray) -> tuple[int, int]:
        """
        Detect carrier in `frame`, decide accept/reject, update estimate.

        Parameters
        ----------
        frame : (H, W) float array — raw interferogram

        Returns
        -------
        (fy, fx) : current best-estimate carrier position (integers).
                   If not yet seeded, returns the raw detection for this frame.
        """
        # 1. Detect candidate from this frame
        fy_raw, fx_raw, sy, sx = detect_carrier(
            frame[np.newaxis],  # detect_carrier expects (N, H, W)
            dc_radius=self.dc_radius,
            search_radius=self.search_radius,
            verbose=False,
        )

        status = self._classify(fy_raw, fx_raw, sy, sx)

        # 2. Update running estimate if accepted
        if status == "accepted":
            self._absorb(fy_raw, fx_raw)

        # 3. Record
        entry = dict(
            frame_idx=len(self.history),
            fy_raw=fy_raw,
            fx_raw=fx_raw,
            sy=sy,
            sx=sx,
            accepted=(status == "accepted"),
            status=status,
            fy_est=self._fy_est,
            fx_est=self._fx_est,
            confidence=self.confidence,
            reject_r=self.reject_radius,
        )
        self.history.append(entry)

        if self.verbose:
            est_str = (f"est=({self.fy},{self.fx})" if self._fy_est is not None
                       else "est=(?)")
            print(
                f"[tracker]  frame {entry['frame_idx']:4d}  "
                f"raw=({fx_raw:4d},{fy_raw:4d})  {est_str}  "
                f"conf={self.confidence:.2f}  gate={self.reject_radius:.1f}px  "
                f"[{status}]"
            )

        # Return best estimate (or raw detection if not yet seeded)
        if self._fy_est is not None:
            return self.fy, self.fx
        return fy_raw, fx_raw

    # ------------------------------------------------------------------
    #  Classification
    # ------------------------------------------------------------------

    def _classify(self, fy: int, fx: int, sy: float, sx: float) -> str:
        """
        Return one of:
            'accepted'          — within gate, update estimate
            'seeding'           — no estimate yet, always accept
            'conjugate_swap'    — near mirror point, reject + warn
            'outlier'           — too far from estimate, reject
            'bad_fit'           — sigma < 1 px (noise spike)
        """
        # Bad fit: sub-pixel sigma means the fit found a noise spike
        if sy < 1.0 or sx < 1.0:
            return "bad_fit"

        # No estimate yet → seed unconditionally
        if self._fy_est is None:
            return "accepted"  # first detection, always take it

        dist = np.sqrt((fy - self._fy_est) ** 2 + (fx - self._fx_est) ** 2)

        # Conjugate mirror position through DC
        fy_conj = 2 * self.cy - self._fy_est
        fx_conj = 2 * self.cx - self._fx_est
        dist_conj = np.sqrt((fy - fy_conj) ** 2 + (fx - fx_conj) ** 2)

        if dist_conj < self.conjugate_radius:
            return "conjugate_swap"

        if not self.is_seeded:
            # Still in seeding phase: accept anything that isn't conjugate
            return "accepted"

        if dist > self.reject_radius:
            return "outlier"

        return "accepted"

    def _absorb(self, fy: float, fx: float) -> None:
        """
        Incorporate a new accepted detection into the running estimate.

        Weight scheme:
          - Each old sample has its weight decayed by (1 - alpha).
          - The new sample gets weight 1.
        This is equivalent to an exponentially weighted moving average
        but expressed through accumulated weights rather than a recursive
        formula, so the full history is preserved for diagnostics.
        """
        decay = 1.0 - self.ema_alpha

        if self._fy_est is None:
            # Very first acceptance: initialize directly
            self._fy_est = float(fy)
            self._fx_est = float(fx)
            self._weight_sum = 1.0
        else:
            # Decay existing weight, add new sample with weight 1
            w_old = self._weight_sum * decay
            w_new = 1.0
            w_total = w_old + w_new
            self._fy_est = (self._fy_est * w_old + fy * w_new) / w_total
            self._fx_est = (self._fx_est * w_old + fx * w_new) / w_total
            self._weight_sum = w_total

    def summary(self) -> str:
        lines = [
            f"LiveCarrierTracker summary",
            f"  Frames processed : {len(self.history)}",
            f"  Accepted         : {self.n_accepted}",
            f"  Rejected         : {self.n_rejected}",
            f"    - outlier      : {sum(1 for h in self.history if h['status'] == 'outlier')}",
            f"    - conjugate    : {sum(1 for h in self.history if h['status'] == 'conjugate_swap')}",
            f"    - bad fit      : {sum(1 for h in self.history if h['status'] == 'bad_fit')}",
            f"  Current estimate : fy={self.fy}, fx={self.fx}",
            f"  Confidence       : {self.confidence:.3f}",
            f"  Reject gate      : {self.reject_radius:.1f} px",
        ]
        if self.n_accepted >= 2:
            accepted = [h for h in self.history if h["accepted"]]
            fy_std = np.std([h["fy_raw"] for h in accepted])
            fx_std = np.std([h["fx_raw"] for h in accepted])
            lines.append(f"  Accepted std     : fy±{fy_std:.2f}  fx±{fx_std:.2f} px")
        return "\n".join(lines)

    def plot_history(self, save_path: str | Path | None = None):
        """
        Plot the time-series of raw detections, accepted/rejected markers,
        the running estimate, and the shrinking reject gate.
        """
        import matplotlib.pyplot as plt

        if not self.history:
            print("[tracker]  No history to plot.")
            return

        frames_idx = [h["frame_idx"] for h in self.history]
        fy_raw = [h["fy_raw"] for h in self.history]
        fx_raw = [h["fx_raw"] for h in self.history]
        fy_est = [h["fy_est"] for h in self.history]
        fx_est = [h["fx_est"] for h in self.history]
        conf = [h["confidence"] for h in self.history]
        gate = [h["reject_r"] for h in self.history]
        status = [h["status"] for h in self.history]

        color_map = {
            "accepted": "#4fc3f7",
            "seeding": "#66bb6a",
            "outlier": "#ef5350",
            "conjugate_swap": "#ff8a65",
            "bad_fit": "#ce93d8",
        }

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.patch.set_facecolor("#0f0f1a")

        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white")
            for sp in ["bottom", "left"]:
                ax.spines[sp].set_color("white")
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            ax.grid(True, alpha=0.15)

        # ── Row 0: fy history ─────────────────────────────────────────
        for st, col in color_map.items():
            idx = [i for i, s in enumerate(status) if s == st]
            if idx:
                axes[0].scatter([frames_idx[i] for i in idx],
                                [fy_raw[i] for i in idx],
                                c=col, s=20, label=st, zorder=3)
        fy_est_clean = [v for v in fy_est if v is not None]
        if fy_est_clean:
            axes[0].plot(
                [frames_idx[i] for i, v in enumerate(fy_est) if v is not None],
                fy_est_clean, color="white", lw=2, label="estimate", zorder=4)
            # Gate band
            axes[0].fill_between(
                frames_idx,
                [e - g if e else np.nan for e, g in zip(fy_est, gate)],
                [e + g if e else np.nan for e, g in zip(fy_est, gate)],
                color="white", alpha=0.08)
        axes[0].set_ylabel("fy [px]", color="white")
        axes[0].set_title("Carrier fy — raw detections vs running estimate",
                          color="white")
        axes[0].legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e",
                       edgecolor="gray", loc="upper right")

        # ── Row 1: fx history ─────────────────────────────────────────
        for st, col in color_map.items():
            idx = [i for i, s in enumerate(status) if s == st]
            if idx:
                axes[1].scatter([frames_idx[i] for i in idx],
                                [fx_raw[i] for i in idx],
                                c=col, s=20, zorder=3)
        fx_est_clean = [v for v in fx_est if v is not None]
        if fx_est_clean:
            axes[1].plot(
                [frames_idx[i] for i, v in enumerate(fx_est) if v is not None],
                fx_est_clean, color="white", lw=2, zorder=4)
            axes[1].fill_between(
                frames_idx,
                [e - g if e else np.nan for e, g in zip(fx_est, gate)],
                [e + g if e else np.nan for e, g in zip(fx_est, gate)],
                color="white", alpha=0.08)
        axes[1].set_ylabel("fx [px]", color="white")
        axes[1].set_title("Carrier fx — raw detections vs running estimate",
                          color="white")

        # ── Row 2: confidence + gate ──────────────────────────────────
        ax2 = axes[2]
        ax2b = ax2.twinx()
        ax2b.set_facecolor("#1a1a2e")
        ax2b.tick_params(colors="#ff8a65")
        ax2b.spines["right"].set_color("#ff8a65")
        ax2b.spines["top"].set_visible(False)

        ax2.plot(frames_idx, conf, color="#4fc3f7", lw=2, label="confidence")
        ax2.axhline(1.0, color="white", lw=1, ls="--", alpha=0.4)
        ax2.set_ylabel("Confidence", color="#4fc3f7")
        ax2.set_ylim(0, 1.1)
        ax2b.plot(frames_idx, gate, color="#ff8a65", lw=1.5, ls="--",
                  label="reject gate [px]")
        ax2b.set_ylabel("Reject gate [px]", color="#ff8a65")
        ax2.set_xlabel("Frame index", color="white")
        ax2.set_title("Confidence growth & shrinking reject gate", color="white")

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=8, labelcolor="white",
                   facecolor="#1a1a2e", edgecolor="gray")

        plt.suptitle(
            f"LiveCarrierTracker history  "
            f"({self.n_accepted} accepted / {self.n_rejected} rejected  "
            f"out of {len(self.history)} frames)",
            color="white", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            plt.savefig(save_path, dpi=110, bbox_inches="tight",
                        facecolor="#0f0f1a")
            print(f"[tracker]  history plot saved → {save_path}")
        plt.close()


# ======================================================================
#  Phase helpers
# ======================================================================

def carrier_to_phase(cf: np.ndarray) -> np.ndarray:
    """Wrapped phase from a complex carrier patch."""
    ph = ifft2(ifftshift(cf))
    return np.arctan2(ph.imag, ph.real)


def remove_plane(wf: np.ndarray) -> np.ndarray:
    """Subtract a least-squares best-fit plane (piston + tip + tilt)."""
    ny, nx = wf.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    A = np.c_[np.ones(ny * nx), X.ravel(), Y.ravel()]
    c, *_ = np.linalg.lstsq(A, wf.ravel(), rcond=None)
    return wf - (c[0] + c[1] * X + c[2] * Y)


# ======================================================================
#  Piston estimation & correction
# ======================================================================

def estimate_pistons(carriers: np.ndarray,
                     ref_idx: int = 0) -> np.ndarray:
    """
    Estimate the global piston offset of each frame relative to ref_idx.

    Method: the global piston between two interferograms equals the angle
    of the mean complex phasor of their wrapped phase difference.

        piston_k = angle( mean( exp(i*(phase_k - phase_ref)) ) )

    This is robust to large local phase gradients because it averages
    over all pixels — local noise cancels, global offset survives.

    Returns
    -------
    pistons : (N,) array [radians]  — pistons[ref_idx] == 0 by definition
    """
    phase_ref = carrier_to_phase(carriers[ref_idx])
    pistons = np.zeros(len(carriers))
    for k, cf in enumerate(carriers):
        diff = carrier_to_phase(cf) - phase_ref
        pistons[k] = np.angle(np.exp(1j * diff).mean())
    return pistons


def correct_pistons(carriers: np.ndarray,
                    pistons: np.ndarray) -> np.ndarray:
    """
    Rotate each complex carrier by -piston_k to align all frames to a
    common phase reference before averaging.

        cf_corrected_k = cf_k * exp(-i * piston_k)

    This is equivalent to subtracting the global offset from the phase
    of each frame, but done in the complex domain so it is exact (no
    wrapping artifacts).

    Returns
    -------
    carriers_corr : (N, H, W) complex128 — piston-corrected carriers
    """
    return np.array([cf * np.exp(-1j * p)
                     for cf, p in zip(carriers, pistons)])


# ======================================================================
#  Core averaging
# ======================================================================

def coherent_average(carriers: np.ndarray) -> np.ndarray:
    """
    Average complex carriers across frames.

    Averaging in complex carrier space is correct:
      - Coherent signal (DM response) adds constructively → amplitude ×N
      - Incoherent noise averages down → amplitude ×√N
      - Net SNR improvement: ×√N

    Returns
    -------
    cf_avg : (H, W) complex128
    """
    return carriers.mean(axis=0)


# ======================================================================
#  Full extraction pipeline
# ======================================================================

def extract_response(frames: np.ndarray,
                     fy: int, fx: int,
                     half_n: int = 128,
                     do_unwrap: bool = True,
                     do_remove_tilt: bool = True,
                     verbose: bool = False,
                     ) -> dict:
    """
    Full pipeline: carriers → piston correction → coherent average → phase.

    Returns
    -------
    result : dict with keys
        'phase'             : final clean phase (tilt removed if requested)
        'phase_wrapped'     : wrapped phase (before unwrap)
        'carriers'          : (N, H, W) raw complex carriers
        'carriers_corrected': (N, H, W) piston-corrected carriers
        'cf_avg'            : (H, W) averaged carrier
        'pistons'           : (N,) piston offsets [rad]
        'phase_stack'       : (N, H, W) individual unwrapped phases
        'noise_map'         : (H, W) per-pixel std across corrected frames
        'noise_mean'        : scalar — mean noise std before averaging
        'noise_avg'         : scalar — estimated noise after averaging
    """
    N = len(frames)

    # 1. Extract carriers
    # carriers, frame_statuses = detect_and_fix_conjugate_frames(frames, fy, fx, half_n=half_n, verbose=verbose)
    carriers = extract_all_carriers(frames, fy, fx, half_n)
    if verbose:
        print(f"[extract]  carrier shape: {carriers[0].shape}  "
              f"N frames: {N}")

    # 2. Estimate and correct piston drift
    pistons = estimate_pistons(carriers)
    pistons_deg = np.degrees(pistons)
    if verbose:
        print(f"[piston]   per-frame drift vs frame 1 [deg]: "
              f"{', '.join(f'{p:+.1f}' for p in pistons_deg)}")
        bad = np.where(np.abs(pistons_deg) > 10)[0]
        if len(bad):
            print(f"           WARNING: frames {bad + 1} have drift > 10° — "
                  f"naive coherent avg would degrade SNR")

    carriers_corr = correct_pistons(carriers, pistons)

    # 3. Coherent average
    cf_avg = coherent_average(carriers_corr)

    # 4. Phase reconstruction
    wrapped = carrier_to_phase(cf_avg)
    if do_unwrap:
        phase = unwrap_phase(wrapped)
    else:
        phase = wrapped.copy()
    if do_remove_tilt:
        phase = remove_plane(phase)

    # 5. Noise estimation from corrected individual frames
    phase_stack = []
    for cf in carriers_corr:
        ph = carrier_to_phase(cf)
        if do_unwrap:
            ph = unwrap_phase(ph)
        if do_remove_tilt:
            ph = remove_plane(ph)
        phase_stack.append(ph)
    phase_stack = np.array(phase_stack)

    noise_map = phase_stack.std(axis=0)
    noise_mean = float(noise_map.mean())
    noise_avg = noise_mean / np.sqrt(N)

    if verbose:
        print(f"[noise]    single-frame: {noise_mean:.3f} rad  "
              f"→  after ×{N} avg: {noise_avg:.3f} rad  "
              f"(improvement: {noise_mean / noise_avg:.2f}×, "
              f"theory √{N}={np.sqrt(N):.2f}×)")
        print(f"[result]   phase RMS={phase.std():.4f} rad  "
              f"PV={phase.max() - phase.min():.4f} rad")

    return dict(
        phase=phase,
        phase_wrapped=wrapped,
        carriers=carriers,
        carriers_corrected=carriers_corr,
        cf_avg=cf_avg,
        pistons=pistons,
        phase_stack=phase_stack,
        noise_map=noise_map,
        noise_mean=noise_mean,
        noise_avg=noise_avg,
    )


# ======================================================================
#  Push-pull influence function (two separate TIFF halves)
# ======================================================================

def process_push_pull(
        frames_plus: np.ndarray,
        frames_minus: np.ndarray,
        amp: float,
        fy: int,
        fx: int,
        half_n: int = 128,
        do_unwrap: bool = True,
        do_remove_tilt: bool = True,
        verbose: bool = False,
) -> dict:
    """
    Compute an actuator influence function from two sets of frames:
    one acquired at +amplitude and one at -amplitude (push-pull).

    CRITICAL: both halves MUST be extracted using the SAME carrier
    position.  The carrier is auto-detected from frames_plus only
    (or from the manual fy/fx override) and then reused for
    frames_minus.  If each half were allowed to auto-detect its own
    carrier, the second half would likely lock onto the conjugate
    (-1 order) peak, flipping the sign of its phase and causing the
    influence function to be roughly DOUBLED and corrupted.

    Parameters
    ----------
    frames_plus  : (N, H, W) array — interferograms at +amp
    frames_minus : (N, H, W) array — interferograms at -amp
    amp          : float — actuator voltage amplitude (absolute value)
    fy, fx       : int or None — carrier override; if None, auto-detected
                   from frames_plus and reused for frames_minus
    half_n       : int — carrier extraction half-window
    dc_radius    : int — DC exclusion radius for auto-detection

    Returns
    -------
    result : dict with keys
        'influence'     : (H, W) influence function [rad / volt]
        'phase_plus'    : clean phase from +amp frames
        'phase_minus'   : clean phase from -amp frames
        'fy', 'fx'      : carrier position used for BOTH halves
        'result_plus'   : full extract_response dict for +amp frames
        'result_minus'  : full extract_response dict for -amp frames
    """

    if verbose:
        print("[push-pull]  processing +amp frames ...")
    res_plus = extract_response(frames_plus, fy=fy, fx=fx, half_n=half_n,
                                do_unwrap=do_unwrap,
                                do_remove_tilt=do_remove_tilt,
                                verbose=verbose)
    if verbose:
        print("[push-pull]  processing -amp frames ...")
    res_minus = extract_response(frames_minus, fy=fy, fx=fx, half_n=half_n,
                                 do_unwrap=do_unwrap,
                                 do_remove_tilt=do_remove_tilt,
                                 verbose=verbose)

    # ── Influence function ─────────────────────────────────────────────
    # wf_measured = wf_static + C_i * v_i
    # phase_plus  recorded at v_i = +amp  → phase_plus  = static + C_i*( +amp)
    # phase_minus recorded at v_i = -amp  → phase_minus = static + C_i*(-amp)
    # difference cancels static:  phase_plus - phase_minus = 2 * C_i * amp
    influence = (res_plus["phase"] - res_minus["phase"]) / (2.0 * amp)

    if verbose:
        print(f"[push-pull]  influence fn  "
              f"RMS={influence.std():.4f} rad/V  "
              f"PV={influence.max() - influence.min():.4f} rad/V")

    return dict(
        influence=influence,
        phase_plus=res_plus["phase"],
        phase_minus=res_minus["phase"],
        fy=fy,
        fx=fx,
        result_plus=res_plus,
        result_minus=res_minus,
    )


def plot_push_pull(pp_result: dict,
                   amp: float,
                   act_id: int | None = None,
                   save_path: str | Path | None = None):
    """
    Diagnostic plot for a push-pull influence function result.
    Shows: +amp phase | -amp phase | influence function | cross-section.
    """
    import matplotlib.pyplot as plt

    ph_p = pp_result["phase_plus"]
    ph_m = pp_result["phase_minus"]
    inf = pp_result["influence"]
    fy = pp_result["fy"]
    fx = pp_result["fx"]

    vmax_ph = max(np.percentile(np.abs(ph_p), 98),
                  np.percentile(np.abs(ph_m), 98))
    vmax_inf = np.percentile(np.abs(inf), 98)
    mid = inf.shape[0] // 2

    title_str = f"Actuator {act_id}  " if act_id is not None else ""
    title_str += f"amp={amp:+.3f} V  |  carrier: fy={fy}, fx={fx}"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#0f0f1a")

    def style(ax, title, col="white"):
        ax.set_title(title, color=col, fontsize=11)
        ax.axis("off")

    def add_cb(im, ax):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("rad", color="white")
        cb.ax.tick_params(colors="white")

    im = axes[0].imshow(ph_p, cmap="RdBu_r", vmin=-vmax_ph, vmax=vmax_ph)
    style(axes[0], f"+amp phase\nRMS={ph_p.std():.3f} rad")
    add_cb(im, axes[0])

    im = axes[1].imshow(ph_m, cmap="RdBu_r", vmin=-vmax_ph, vmax=vmax_ph)
    style(axes[1], f"-amp phase\nRMS={ph_m.std():.3f} rad")
    add_cb(im, axes[1])

    im = axes[2].imshow(inf, cmap="RdBu_r", vmin=-vmax_inf, vmax=vmax_inf)
    style(axes[2], f"Influence fn  (ph+ - ph-) / 2amp\nRMS={inf.std():.3f} rad/V",
          col="#4fc3f7")
    add_cb(im, axes[2])

    ax = axes[3]
    ax.set_facecolor("#1a1a2e")
    ax.plot(ph_p[mid], color="#ef5350", lw=1.5, label=f"+amp (RMS={ph_p.std():.2f})")
    ax.plot(ph_m[mid], color="#ff8a65", lw=1.5, label=f"-amp (RMS={ph_m.std():.2f})")
    ax.plot(inf[mid] * 2 * amp, color="#4fc3f7", lw=2, label="2·amp·influence")
    ax.set_xlabel("Pixel", color="white")
    ax.set_ylabel("Phase [rad]", color="white")
    ax.set_title("Cross-section (mid row)", color="white")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="gray")
    ax.tick_params(colors="white")
    for sp in ["bottom", "left"]: ax.spines[sp].set_color("white")
    for sp in ["top", "right"]:   ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.2)

    plt.suptitle(title_str, color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="#0f0f1a")
        print(f"[plot]     saved → {save_path}")
    plt.close()


if __name__ == "__main__":
    import tifffile as tf

    input_path = r"C:\Users\Public\Data\20260223\202602231352_influence_function\actuator_11_step_0.1_interferometry.tif"
    output_dir = Path(r"C:\Users\Public\Data\20260223\202602231352_influence_function\results")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 16-frame stack and split into two halves of 8
    data_stack = tf.imread(input_path).astype(np.float64)
    frames_a, frames_b = data_stack[:8], data_stack[8:]

    # Detect carrier from frames_b and reuse for frames_a
    # (avoids conjugate-order mismatch if the two halves drifted in tilt)
    fy, fx, _, _ = detect_carrier(frames_b)

    # Extract response from frames_a using the shared carrier
    result_a = extract_response(frames_a, fy=fy, fx=fx, half_n=128)
    result_b = extract_response(frames_b, fy=fy, fx=fx, half_n=128)

    tf.imwrite(out_dir / "phase_plus.tif", result_b["phase"])
    tf.imwrite(out_dir / "phase_minus.tif", result_a["phase"])

    pp = process_push_pull(frames_b, frames_a, amp=0.1, fy=fy, fx=fx, half_n=128)
    plot_push_pull(pp, amp=0.1, act_id=10, save_path=out_dir / "influence.png")
    tf.imwrite(out_dir / "phase_influence.tif", pp["influence"])
