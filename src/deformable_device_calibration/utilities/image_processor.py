# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np
from numpy.fft import fft2, fftshift
from scipy.optimize import curve_fit
from skimage import filters
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════════
#  Image statistics
# ══════════════════════════════════════════════════════════════════════

def rms(data):
    _nx, _ny = data.shape
    _n = _nx * _ny
    m = np.mean(data, dtype=np.float64)
    a = (data - m) ** 2
    r = np.sqrt(np.sum(a) / _n)
    return r


def img_statistics(img):
    return img.min(), img.max(), rms(img)


def calculate_focus_measure_with_sobel(image):
    edges = filters.sobel(image)
    focus_measure = np.var(edges)
    return focus_measure


def calculate_focus_measure_with_laplacian(image):
    laplacian_image = filters.laplace(image)
    focus_measure = np.var(laplacian_image)
    return focus_measure


# ══════════════════════════════════════════════════════════════════════
#  Extremum detection
# ══════════════════════════════════════════════════════════════════════

def binomial_model(x, a, b, c):
    return a * x ** 2 + b * x + c


def peak_find(x_data, y_data, y_std=None):
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    if y_std is not None:
        y_err = np.asarray(y_std)
        p_opt, p_cov = curve_fit(binomial_model, x, y, sigma=y_err, absolute_sigma=True)
        a, b, c = p_opt
        x_peak = -b / (2 * a)
        if a > 0:
            return "No peak"
        elif x_peak >= x.max():
            return "Peak above maximum"
        elif x_peak <= x.min():
            return "Peak below minimum"
        else:
            sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(p_cov))
            x_peak_err = np.sqrt((b / (2 * a ** 2) * sigma_a) ** 2 + (-1 / (2 * a) * sigma_b) ** 2)
            print(x_peak_err)
            return x_peak
    else:
        a, b, c = np.polyfit(x, y, 2)
        x_peak = -b / (2 * a)
        if a > 0:
            return "No peak"
        elif x_peak >= x.max():
            return "Peak above maximum"
        elif x_peak <= x.min():
            return "Peak below minimum"
        else:
            return x_peak


def valley_find(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    a, b, c = np.polyfit(x, y, 2)
    v = -b / (2 * a)
    if a < 0:
        return "no minimum"
    elif (v >= x.max()) or (v <= x.min()):
        return "minimum exceeding range"
    else:
        return v


# ══════════════════════════════════════════════════════════════════════
#  Matrix decomposition
# ══════════════════════════════════════════════════════════════════════

def pseudo_inverse(in_matrix, n_modes_kept=None, condition_limit=None):
    if n_modes_kept is None and condition_limit is None:
        raise ValueError("Either n_modes_kept or condition_limit must be provided.")
    if n_modes_kept is not None and condition_limit is not None:
        raise ValueError("Only one of n_modes_kept or condition_limit can be provided, not both.")
    if n_modes_kept is not None:
        U, sv, Vt = np.linalg.svd(in_matrix, full_matrices=False)
        sv_inv = np.zeros_like(sv)
        sv_inv[:n_modes_kept] = 1.0 / sv[:n_modes_kept]
        C_inv = (Vt[:n_modes_kept].T * sv_inv[:n_modes_kept]) @ U[:, :n_modes_kept].T
    if condition_limit is not None:
        U, s, Vt = np.linalg.svd(in_matrix, full_matrices=False)
        s_inv = np.where(s / s[0] > 1 / condition_limit, 1 / s, 0.0)
        C_inv = Vt.T @ np.diag(s_inv) @ U.T
    return C_inv


def get_eigen_coefficients(mta, mtb, ng=32):
    mp = pseudo_inverse(mtb, condition_limit=50)
    return np.matmul(mp, mta)


# ══════════════════════════════════════════════════════════════════════
#  Centroid detection
# ══════════════════════════════════════════════════════════════════════

def centroid_cog(img: np.ndarray) -> Tuple[float, float]:
    """
    Basic Center of Gravity - Sensitive to background noise.
    """
    img = img.astype(np.float64)
    total = img.sum()
    if total <= 0:
        return img.shape[1] / 2.0, img.shape[0] / 2.0

    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    cx = (xx * img).sum() / total
    cy = (yy * img).sum() / total
    return cx, cy


def centroid_thresholded(img: np.ndarray, threshold_fraction: float = 0.1) -> Tuple[float, float]:
    """
    Thresholded Center of Gravity.
    Subtracts background and zeros out pixels below a fraction of the peak.
    More robust to noise than basic CoG.

    Parameters
    ----------
    img : 2D array
        Image data to be processed.
    threshold_fraction : float
        Fraction of (peak - background) below which pixels are zeroed.
        0.1 means keep only pixels above 10% of the spot's dynamic range.
    """
    img = img.astype(np.float64)
    bg = np.median(img)
    img_bg = img - bg
    peak = img_bg.max()

    if peak <= 0:
        return img.shape[1] / 2.0, img.shape[0] / 2.0

    threshold = threshold_fraction * peak
    img_thresh = np.where(img_bg > threshold, img_bg, 0.0)

    total = img_thresh.sum()
    if total <= 0:
        return img.shape[1] / 2.0, img.shape[0] / 2.0

    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    cx = (xx * img_thresh).sum() / total
    cy = (yy * img_thresh).sum() / total
    return cx, cy


def centroid_iwcog(img: np.ndarray, n_iter: int = 5,
                   initial_sigma: float = 3.0) -> Tuple[float, float]:
    """
    Iterative Weighted Center of Gravity.

    Starts with thresholded CoG, then iteratively applies a Gaussian
    weight centered on the current estimate. Converges to a stable,
    noise-robust centroid.

    Parameters
    ----------
    img : 2D array
        Image data to be processed.
    n_iter : int
        Number of iterations (3-5 is usually enough).
    initial_sigma : float
        Initial Gaussian window width in pixels.
    """
    img = img.astype(np.float64)
    bg = np.median(img)
    img_bg = np.clip(img - bg, 0, None)

    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]

    # Initialize with thresholded CoG
    cx, cy = centroid_thresholded(img)

    sigma = initial_sigma
    for _ in range(n_iter):
        # Gaussian weight centered on current estimate
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        weight = np.exp(-r2 / (2 * sigma ** 2))
        weighted = img_bg * weight

        total = weighted.sum()
        if total <= 0:
            break

        cx = (xx * weighted).sum() / total
        cy = (yy * weighted).sum() / total

        # Optionally tighten the window
        sigma = max(sigma * 0.9, 1.5)

    return cx, cy


def centroid_gaussian(img: np.ndarray, fit_radius: int = 4) -> Tuple[float, float]:
    """
    Gaussian fit centroiding — best for laser spots.

    Fits a 2D Gaussian to the region around the peak using
    linearized least-squares on log(intensity).

    Parameters
    ----------
    img : 2D array
        Image data to be processed.
    fit_radius : int
        Half-width of the fitting window around the peak.

    Returns
    -------
    cx, cy : float
        img-pixel center of the fitted Gaussian.
    """
    img = img.astype(np.float64)
    ny, nx = img.shape

    # Find peak
    peak_pos = np.unravel_index(np.argmax(img), img.shape)
    py, px = peak_pos

    # Extract fitting window
    r = fit_radius
    y0 = max(py - r, 0)
    y1 = min(py + r + 1, ny)
    x0 = max(px - r, 0)
    x1 = min(px + r + 1, nx)

    window = img[y0:y1, x0:x1]
    wy, wx = window.shape
    yy, xx = np.mgrid[:wy, :wx]

    # Subtract background and clip
    bg = np.percentile(img, 25)
    window_bg = np.clip(window - bg, 1e-10, None)

    # Linearize: ln(I) = ln(A) - (x-cx)²/(2σx²) - (y-cy)²/(2σy²)
    # Rearrange as: ln(I) = c0 + c1*x + c2*x² + c3*y + c4*y²
    log_I = np.log(window_bg)
    valid = np.isfinite(log_I) & (window_bg > bg * 0.05)

    if valid.sum() < 6:
        # Not enough points, fall back to thresholded CoG
        return centroid_thresholded(img)

    x_flat = xx[valid].ravel()
    y_flat = yy[valid].ravel()
    z_flat = log_I[valid].ravel()

    # Design matrix: [1, x, x², y, y²]
    A = np.column_stack([
        np.ones(len(x_flat)),
        x_flat,
        x_flat ** 2,
        y_flat,
        y_flat ** 2,
    ])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    except np.linalg.LinAlgError:
        return centroid_thresholded(img)

    # c2 = -1/(2σx²) → cx = -c1/(2*c2)
    # c4 = -1/(2σy²) → cy = -c3/(2*c4)
    c0, c1, c2, c3, c4 = coeffs

    if c2 >= 0 or c4 >= 0:
        # Not a peak (concave up), fall back
        return centroid_thresholded(img)

    cx_local = -c1 / (2 * c2)
    cy_local = -c3 / (2 * c4)

    # Convert back to full sub-image coordinates
    cx = cx_local + x0
    cy = cy_local + y0

    # Sanity check: result should be within the sub-image
    cx = np.clip(cx, 0, nx - 1)
    cy = np.clip(cy, 0, ny - 1)

    return cx, cy


def centroid_crosscorr(img: np.ndarray, reference: np.ndarray,
                       upsample: int = 10) -> Tuple[float, float]:
    """
    Cross-correlation centroiding.

    Computes the shift between the image and a reference spot
    using FFT-based cross-correlation with sub-pixel upsampling.

    Parameters
    ----------
    img : 2D array
        Image data to be processed.
    reference : 2D array
        Reference spot image (same size as img).
        Typically, the image from a flat wavefront measurement.
    upsample : int
        Upsampling factor for sub-pixel precision.
        10 gives 0.1 pixel accuracy, 100 gives 0.01.

    Returns
    -------
    cx, cy : float
        Position relative to the reference (shift_x, shift_y in pixels).
        NOTE: unlike other methods, this returns SHIFTS directly,
        not absolute positions.
    """
    from numpy.fft import fft2, ifft2, fftshift

    img = img.astype(np.float64)
    ref = reference.astype(np.float64)

    # Subtract means
    img = img - img.mean()
    ref = ref - ref.mean()

    ny, nx = img.shape

    # Cross-correlation via FFT
    F_img = fft2(img)
    F_ref = fft2(ref)
    cross = ifft2(F_img * np.conj(F_ref))
    cross = np.abs(fftshift(cross))

    # Coarse peak
    peak_pos = np.unravel_index(np.argmax(cross), cross.shape)

    # Sub-pixel refinement via parabolic interpolation around peak
    py, px = peak_pos
    cy_coarse = py - ny // 2  # shift relative to center
    cx_coarse = px - nx // 2

    # Parabolic refinement in x
    if 0 < px < nx - 1:
        left = cross[py, px - 1]
        center = cross[py, px]
        right = cross[py, px + 1]
        denom = 2 * (2 * center - left - right)
        dx = (left - right) / denom if abs(denom) > 1e-10 else 0.0
    else:
        dx = 0.0

    # Parabolic refinement in y
    if 0 < py < ny - 1:
        top = cross[py - 1, px]
        center = cross[py, px]
        bottom = cross[py + 1, px]
        denom = 2 * (2 * center - top - bottom)
        dy = (top - bottom) / denom if abs(denom) > 1e-10 else 0.0
    else:
        dy = 0.0

    shift_x = cx_coarse + dx
    shift_y = cy_coarse + dy

    return shift_x, shift_y


# ══════════════════════════════════════════════════════════════════════
#  Gaussian fit
# ══════════════════════════════════════════════════════════════════════


FWHM_FACTOR = np.sqrt(2 * np.log(2))


def fit_gaussian_2d(image, verbose=False, plot=False, allow_rotation=True,
                    bounds=None, p0=None):
    """
    Fit 2D Gaussian directly to image (more accurate than 1D projections)

    Args:
        image: 2D numpy array
        verbose: Print fit parameters
        plot: Show fit visualization
        allow_rotation: Allow ellipse rotation (slower but more accurate)
        bounds: Custom bounds for curve_fit
        p0: Initial guess (auto-computed if None)

    Returns:
        params: Fitted parameters
            If allow_rotation=False: [bg, I0, x0, y0, wx, wy]
            If allow_rotation=True:  [bg, I0, x0, y0, wx, wy, theta]
        residual: RMS fitting residual
    """
    # Input validation
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")

    y_px, x_px = image.shape

    # Create coordinate arrays
    x = np.arange(x_px, dtype=np.float64)
    y = np.arange(y_px, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Flatten for fitting
    coords = np.vstack((xx.ravel(), yy.ravel()))
    data = image.ravel()

    # Compute statistics for bounds and initial guess
    img_min = np.min(image)
    img_max = np.max(image)

    #  Compute smart initial guess from image moments
    if p0 is None:
        p0 = _compute_2d_initial_guess(image, allow_rotation)

    #  Set bounds
    if bounds is None:
        bounds = _compute_2d_bounds(x_px, y_px, img_min, img_max, allow_rotation)

    #  Choose fitting function
    if allow_rotation:
        fit_func = lambda coords, bg, I0, x0, y0, wx, wy, theta: \
            gaussian_2d_rotated(coords, bg, I0, x0, y0, wx, wy, theta)
    else:
        fit_func = lambda coords, bg, I0, x0, y0, wx, wy: \
            gaussian_2d_simple(coords, bg, I0, x0, y0, wx, wy)

    #  Fit with error handling
    try:
        popt, pcov = curve_fit(fit_func, coords, data, p0=p0, bounds=bounds, maxfev=10000)
    except ValueError as e:
        raise ValueError(f"2D Gaussian fit failed: {e}") from e

    # Calculate residual
    fitted_data = fit_func(coords, *popt)
    residual = np.sqrt(np.mean((data - fitted_data) ** 2))

    if verbose:
        _print_2d_params(popt, residual, allow_rotation)

    if plot:
        _plot_2d_fit(image, xx, yy, popt, residual, allow_rotation)

    return popt, residual, fitted_data


def gaussian_2d_simple(coords, bg, I0, x0, y0, wx, wy):
    """
    2D Gaussian without rotation (axis-aligned ellipse)

    Args:
        coords: (2, N) array of [x, y] coordinates
        bg: Background intensity
        I0: Peak intensity
        x0, y0: Center position
        wx, wy: Width in x and y (1/e^2 radius)

    Returns:
        Intensity values at each coordinate
    """
    x, y = coords
    return bg + I0 * np.exp(-2 * ((x - x0) / wx) ** 2 - 2 * ((y - y0) / wy) ** 2)


def gaussian_2d_rotated(coords, bg, I0, x0, y0, wx, wy, theta):
    """
    2D Gaussian with rotation (full ellipse)

    Args:
        coords: (2, N) array of [x, y] coordinates
        bg: Background intensity
        I0: Peak intensity
        x0, y0: Center position
        wx, wy: Width in x and y (1/e^2 radius)
        theta: Rotation angle in radians

    Returns:
        Intensity values at each coordinate
    """
    x, y = coords

    # Rotation coefficients
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_t_sq = cos_t ** 2
    sin_t_sq = sin_t ** 2

    # Rotated Gaussian formula
    a = cos_t_sq / (2 * wx ** 2) + sin_t_sq / (2 * wy ** 2)
    b = -np.sin(2 * theta) / (4 * wx ** 2) + np.sin(2 * theta) / (4 * wy ** 2)
    c = sin_t_sq / (2 * wx ** 2) + cos_t_sq / (2 * wy ** 2)

    dx = x - x0
    dy = y - y0

    return bg + I0 * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))


def _compute_2d_initial_guess(image, allow_rotation):
    """
    Compute smart initial guess from image moments

    Args:
        image: 2D numpy array
        allow_rotation: Include rotation parameter

    Returns:
        p0: Initial guess array
    """
    y_px, x_px = image.shape

    # Background: percentile of lowest values
    bg = np.percentile(image, 10)

    # Peak intensity
    I0 = np.max(image) - bg
    I0 = max(I0, 1.0)

    # Center of mass
    img_shifted = image - bg
    img_shifted = np.maximum(img_shifted, 0)

    total = np.sum(img_shifted)
    if total > 0:
        x = np.arange(x_px)
        y = np.arange(y_px)
        xx, yy = np.meshgrid(x, y)

        x0 = np.sum(xx * img_shifted) / total
        y0 = np.sum(yy * img_shifted) / total

        # Second moments for width estimation
        var_x = np.sum((xx - x0) ** 2 * img_shifted) / total
        var_y = np.sum((yy - y0) ** 2 * img_shifted) / total

        wx = max(1.0, np.sqrt(var_x))
        wy = max(1.0, np.sqrt(var_y))

        # Estimate rotation from covariance
        if allow_rotation:
            cov_xy = np.sum((xx - x0) * (yy - y0) * img_shifted) / total
            theta = 0.5 * np.arctan2(2 * cov_xy, var_x - var_y)
        else:
            theta = None
    else:
        # Fallback for empty/uniform image
        x0 = x_px / 2
        y0 = y_px / 2
        wx = x_px / 4
        wy = y_px / 4
        theta = 0.0 if allow_rotation else None

    if allow_rotation:
        return [bg, I0, x0, y0, wx, wy, theta]
    else:
        return [bg, I0, x0, y0, wx, wy]


def _compute_2d_bounds(x_px, y_px, img_min, img_max, allow_rotation):
    """
    Compute bounds for 2D Gaussian fit

    Args:
        x_px, y_px: Image dimensions
        img_min, img_max: Image intensity range
        allow_rotation: Include rotation bounds

    Returns:
        bounds: ((lower_bounds), (upper_bounds))
    """
    # Background
    bg_min = min(0, 1.5 * img_min)
    bg_max = max(10, img_min * 10) if img_min > 0 else 10

    # Peak intensity
    I0_min = max(0, img_min)
    I0_max = max(I0_min + 1, img_max * 2)

    # Position
    x0_min, x0_max = -x_px * 0.2, x_px * 1.2  # Allow slightly outside
    y0_min, y0_max = -y_px * 0.2, y_px * 1.2

    # Width
    wx_min, wy_min = 0.5, 0.5
    wx_max, wy_max = x_px * 2, y_px * 2

    if allow_rotation:
        # Rotation angle: -π/2 to π/2 (sufficient due to symmetry)
        theta_min, theta_max = -np.pi / 2, np.pi / 2

        lower = [bg_min, I0_min, x0_min, y0_min, wx_min, wy_min, theta_min]
        upper = [bg_max, I0_max, x0_max, y0_max, wx_max, wy_max, theta_max]
    else:
        lower = [bg_min, I0_min, x0_min, y0_min, wx_min, wy_min]
        upper = [bg_max, I0_max, x0_max, y0_max, wx_max, wy_max]

    return [lower, upper]


def gaussian_integral_fwhm_2d(params, allow_rotation=False, bg=False):
    """
    Compute integral of 2D Gaussian within FWHM ellipse (analytical)

    Args:
        params: Fitted parameters from fit_gaussian_2d
                [bg, I0, x0, y0, wx, wy] or [bg, I0, x0, y0, wx, wy, theta]
        allow_rotation: Whether rotation parameter is included

    Returns:
        integral: Total intensity within FWHM ellipse
    """
    bg, I0, x0, y0, wx, wy = params[:6]
    theta = params[6] if allow_rotation and len(params) > 6 else 0

    # Total integral of 2D Gaussian (infinite extent): I0 * π * wx * wy / 2
    total_gaussian_integral = I0 * np.pi * wx * wy / 2

    # Fraction of power within FWHM ellipse (derived from error function)
    # For 1/e^2 definition, FWHM corresponds to ~1.177 * w0
    # Integral within FWHM ellipse is ~0.5 of total (50% power)
    fwhm_fraction = 1 - np.exp(-2 * np.log(2))  # ≈ 0.75 or 75%

    gaussian_within_fwhm = total_gaussian_integral * fwhm_fraction

    if bg:
        # Background contribution (area of FWHM ellipse)
        fwhm_x = wx * np.sqrt(2 * np.log(2))
        fwhm_y = wy * np.sqrt(2 * np.log(2))
        fwhm_area = np.pi * fwhm_x * fwhm_y
        background_contribution = bg * fwhm_area
        total_integral = gaussian_within_fwhm + background_contribution
        return total_integral
    else:
        return gaussian_within_fwhm


def _print_2d_params(params, residual, allow_rotation):
    """Print fitted parameters"""
    bg, I0, x0, y0, wx, wy = params[:6]

    print(f"\n2D Gaussian Fit Results:")
    print(f"  Background: {bg:.2f}")
    print(f"  Peak intensity: {I0:.2f}")
    print(f"  Center: ({x0:.2f}, {y0:.2f})")
    print(f"  Width X: {wx:.2f} px (FWHM: {wx * FWHM_FACTOR:.2f})")
    print(f"  Width Y: {wy:.2f} px (FWHM: {wy * FWHM_FACTOR:.2f})")
    print(f"  Ellipticity: {max(wx, wy) / min(wx, wy):.3f}")

    if allow_rotation and len(params) > 6:
        theta = params[6]
        print(f"  Rotation: {np.degrees(theta):.2f}°")

    print(f"  RMS residual: {residual:.2f}")


def _plot_2d_fit(image, xx, yy, params, residual, allow_rotation):
    """
    Plot 2D Gaussian fit results

    Args:
        image: Original image
        xx, yy: Coordinate meshgrids
        params: Fitted parameters
        residual: RMS residual
        allow_rotation: Whether rotation was used
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError:
        print("Warning: matplotlib not available, skipping plot")
        return

    # Generate fitted image
    coords = np.vstack((xx.ravel(), yy.ravel()))
    if allow_rotation:
        fitted_flat = gaussian_2d_rotated(coords, *params)
    else:
        fitted_flat = gaussian_2d_simple(coords, *params)

    fitted = fitted_flat.reshape(image.shape)
    residual_img = image - fitted

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    im0 = axes[0, 0].imshow(image, cmap='viridis', origin='lower')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im0, ax=axes[0, 0])

    # Fitted image
    im1 = axes[0, 1].imshow(fitted, cmap='viridis', origin='lower')
    axes[0, 1].set_title('Fitted Gaussian')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 1])

    # Residual
    im2 = axes[0, 2].imshow(residual_img, cmap='RdBu', origin='lower')
    axes[0, 2].set_title(f'Residual (RMS={residual:.2f})')
    axes[0, 2].set_xlabel('X (pixels)')
    axes[0, 2].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[0, 2])

    # X projection
    x_proj_orig = np.max(image, axis=0)
    x_proj_fit = np.max(fitted, axis=0)
    axes[1, 0].plot(xx[0, :], x_proj_orig, 'o', alpha=0.5, label='Original')
    axes[1, 0].plot(xx[0, :], x_proj_fit, '-', linewidth=2, label='Fitted')
    axes[1, 0].set_title('X Projection')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Y projection
    y_proj_orig = np.max(image, axis=1)
    y_proj_fit = np.max(fitted, axis=1)
    axes[1, 1].plot(yy[:, 0], y_proj_orig, 'o', alpha=0.5, label='Original')
    axes[1, 1].plot(yy[:, 0], y_proj_fit, '-', linewidth=2, label='Fitted')
    axes[1, 1].set_title('Y Projection')
    axes[1, 1].set_xlabel('Y (pixels)')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Contour plot with ellipse
    bg, I0, x0, y0, wx, wy = params[:6]
    theta = params[6] if allow_rotation and len(params) > 6 else 0

    axes[1, 2].imshow(image, cmap='gray', origin='lower', alpha=0.5)

    # Draw contours of fitted Gaussian
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    contours = axes[1, 2].contour(xx, yy, fitted,
                                  levels=[bg + I0 * l for l in levels],
                                  colors='red', linewidths=2)

    # Draw ellipse at 1/e^2 level
    ellipse = Ellipse(
        xy=(x0, y0),
        width=2 * wx,
        height=2 * wy,
        angle=np.degrees(theta),
        fill=False,
        edgecolor='cyan',
        linewidth=2,
        linestyle='--',
        label='1/e² radius'
    )
    axes[1, 2].add_patch(ellipse)

    # Mark center
    axes[1, 2].plot(x0, y0, 'r+', markersize=15, markeredgewidth=2)

    axes[1, 2].set_title('Contours & Ellipse')
    axes[1, 2].set_xlabel('X (pixels)')
    axes[1, 2].set_ylabel('Y (pixels)')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()


def gauss_metric(img, s=True):
    try:
        params, residual, ftg = fit_gaussian_2d(img, verbose=False, allow_rotation=True, plot=False)
        if s:
            ss = gaussian_integral_fwhm_2d(params, True)
            return ss
        else:
            return params[1]
    except Exception as e:
        return f"Gaussian Fitting Error: {e}"

