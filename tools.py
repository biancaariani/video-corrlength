"""
tools.py
────────────────
Utility functions for the collective-behaviour analysis pipeline.

Key change vs. original
────────────────────────
``extract_control_param(fname, param_prefix)`` is a new *generic* extractor
that replaces the hard-coded ``extract_strength`` / ``extract_threshold`` /
``extract_stimulus_strength`` trio for the purposes of the batch pipeline.
Pass the exact prefix string that appears before the value in the filename,
e.g. ``"strength"`` → matches ``...__strength_0.42__...``.

The original specialised extractors are kept for backward compatibility.
"""

import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, label
from skimage.filters import threshold_otsu


# ══════════════════════════════════════════════════════════════════════════════
# File-name parsing
# ══════════════════════════════════════════════════════════════════════════════

def extract_control_param(fname: str, param_prefix: str) -> float:
    """
    Extract a numeric control-parameter value from a filename.

    The function searches for the pattern ``<param_prefix>_<number>`` inside
    the basename of *fname* (underscores separate tokens, double-underscores
    separate fields).  Scientific notation (``1.5e-3``) is supported.

    Parameters
    ----------
    fname : str
        Full path or bare filename, e.g.
        ``"run__strength_0.42__seed_3_last20sec.mp4"``
    param_prefix : str
        The label that immediately precedes the value, e.g. ``"strength"``,
        ``"threshold"``, ``"stimulus_strength"``.

    Returns
    -------
    float
        The extracted numeric value.

    Raises
    ------
    ValueError
        If no matching pattern is found in the filename.

    Examples
    --------
    >>> extract_control_param("run__strength_0.42__seed_3.mp4", "strength")
    0.42
    >>> extract_control_param("run__stimulus_strength_1.5e-3__seed_0.mp4",
    ...                       "stimulus_strength")
    0.0015
    """
    basename = os.path.basename(fname)
    # Build a regex: prefix followed by underscore and a float / sci-notation number
    pattern = re.escape(param_prefix) + r"_([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    match = re.search(pattern, basename)
    if match:
        return float(match.group(1))
    raise ValueError(
        f"Could not find '{param_prefix}_<value>' in filename: {basename}"
    )


# ── legacy specialised extractors (kept for backward compatibility) ───────────

def extract_strength(fname: str) -> float:
    """Extract the ``strength_<value>`` token from a filename."""
    return extract_control_param(fname, "strength")


def extract_threshold(fname: str) -> float:
    """Extract the ``threshold_<value>`` token from a filename."""
    return extract_control_param(fname, "threshold")


def extract_stimulus_strength(fname: str) -> float:
    """Extract the ``stimulus_strength_<value>`` token from a filename."""
    return extract_control_param(fname, "stimulus_strength")


# ══════════════════════════════════════════════════════════════════════════════
# Binarization
# ══════════════════════════════════════════════════════════════════════════════

def frame_to_binary(frame_gray: np.ndarray,
                    method: str = "otsu",
                    threshold: int = 128) -> np.ndarray:
    """
    Binarize a grayscale frame.

    Parameters
    ----------
    frame_gray : ndarray
        2-D uint8 grayscale image.
    method : {"otsu", "fixed"}
        ``"otsu"``  – automatic Otsu threshold (default).
        ``"fixed"`` – use the literal *threshold* value.
    threshold : int
        Used only when *method* = ``"fixed"``.

    Returns
    -------
    ndarray of uint8
        Binary image (0 / 1).
    """
    if method == "otsu":
        t = threshold_otsu(frame_gray)
        return (frame_gray >= t).astype(np.uint8)
    return (frame_gray >= threshold).astype(np.uint8)


def frame_to_binary_smoothing(frame_gray: np.ndarray,
                               method: str = "otsu",
                               threshold: int = 128,
                               min_cluster_size: int = 1,
                               hole_smooth_size: int = 0) -> np.ndarray:
    """
    Binarize a grayscale frame with optional post-processing.

    Parameters
    ----------
    frame_gray : ndarray
        2-D uint8 grayscale image.
    method : {"otsu", "fixed"}
        Thresholding strategy.
    threshold : int
        Used only when *method* = ``"fixed"``.
    min_cluster_size : int
        Remove connected foreground regions smaller than this many pixels.
        Set to 1 (default) to disable.
    hole_smooth_size : int
        Side-length of the square structuring element used for binary closing
        (fills small holes).  Set to 0 (default) to disable.

    Returns
    -------
    ndarray of uint8
        Binary image (0 / 1).
    """
    if method == "otsu":
        t = threshold_otsu(frame_gray)
        binary = frame_gray >= t
    else:
        binary = frame_gray >= threshold

    if min_cluster_size > 1:
        labeled, _ = label(binary)
        counts = np.bincount(labeled.ravel())
        remove = counts < min_cluster_size
        remove[0] = False
        binary[remove[labeled]] = 0

    if hole_smooth_size > 0:
        structure = np.ones((hole_smooth_size, hole_smooth_size))
        binary = binary_closing(binary, structure=structure)

    return binary.astype(np.uint8)


def check_binarization_quality(filepath: str,
                                method: str = "otsu",
                                param: str = "strength",
                                smoothing: bool = True) -> None:
    """
    Display a side-by-side comparison of the raw mid-frame and its
    binarized version for a quick sanity check.

    Parameters
    ----------
    filepath : str
        Path to the .mp4 video file.
    method : {"otsu", "fixed"}
        Binarization method.
    param : str
        Control-parameter prefix used to annotate the plot title
        (e.g. ``"strength"``, ``"threshold"``).
    smoothing : bool
        Whether to apply ``frame_to_binary_smoothing`` (True) or
        plain ``frame_to_binary`` (False).
    """
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[WARNING] Could not read frame from {os.path.basename(filepath)}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if smoothing:
        binary = frame_to_binary_smoothing(gray, method=method,
                                           min_cluster_size=2,
                                           hole_smooth_size=2)
    else:
        binary = frame_to_binary(gray, method=method)

    fname = os.path.basename(filepath)
    try:
        val = extract_control_param(fname, param)
    except ValueError:
        val = "?"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title(f"Raw Frame  ({param} = {val})")
    axes[1].imshow(binary, cmap="binary")
    axes[1].set_title(f"Binarized  (method = {method})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Structure factor
# ══════════════════════════════════════════════════════════════════════════════

def compute_structure_factor(frame_bin: np.ndarray) -> np.ndarray:
    """
    Compute the 2-D power spectrum (structure factor) of a binary frame.

    The mean is subtracted before the FFT so the DC component does not
    dominate the spectrum.

    Parameters
    ----------
    frame_bin : ndarray
        Binary (0/1) 2-D array.

    Returns
    -------
    ndarray
        2-D power spectrum (|FFT|²), same shape as input, *not* shifted.
    """
    frame = frame_bin.astype(float)
    frame -= frame.mean()
    fft = np.fft.fft2(frame)
    return np.abs(fft) ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Anisotropy metric
# ══════════════════════════════════════════════════════════════════════════════

def compute_anisotropy_metric(S: np.ndarray, n_bins: int = 36) -> float:
    """
    Quantify angular anisotropy of the (shifted) 2-D structure factor.

    The power spectrum is sampled in an annular region
    ``r_min ≤ r ≤ r_max`` and binned by polar angle.  Anisotropy is
    defined as the coefficient of variation of the angular bin means.

    Parameters
    ----------
    S : ndarray
        Shifted 2-D power spectrum (output of ``np.fft.fftshift``).
    n_bins : int
        Number of angular bins (default 36 → 10° per bin).

    Returns
    -------
    float
        Anisotropy index (std / mean of angular profile).
        Returns ``nan`` if the mean power is zero.
    """
    y, x = np.indices(S.shape)
    center = np.array(S.shape) // 2
    dx = x - center[1]
    dy = y - center[0]
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    r_min, r_max = 5, min(S.shape) // 4
    mask = (r >= r_min) & (r <= r_max)

    theta_vals = theta[mask]
    S_vals = S[mask]
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    bin_means = []
    for i in range(n_bins):
        sel = (theta_vals >= bins[i]) & (theta_vals < bins[i + 1])
        if np.any(sel):
            bin_means.append(np.mean(S_vals[sel]))

    bin_means = np.array(bin_means)
    if bin_means.mean() == 0:
        return np.nan
    return float(np.std(bin_means) / np.mean(bin_means))


# ══════════════════════════════════════════════════════════════════════════════
# Radial average
# ══════════════════════════════════════════════════════════════════════════════

def radial_average_2d(arr: np.ndarray) -> np.ndarray:
    """
    Compute the radially averaged profile of a 2-D array.

    Pixels are grouped by their integer distance from the array centre;
    the mean value in each ring is returned.

    Parameters
    ----------
    arr : ndarray
        2-D array (e.g. autocorrelation map).

    Returns
    -------
    ndarray, shape (r_max + 1,)
        Mean value as a function of integer radius.
    """
    y, x = np.indices(arr.shape)
    center = np.array(arr.shape) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), arr.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr


# ══════════════════════════════════════════════════════════════════════════════
# Autocorrelation
# ══════════════════════════════════════════════════════════════════════════════

def compute_autocorrelation_from_S(S_mean: np.ndarray) -> np.ndarray:
    """
    Recover the spatial autocorrelation function from an averaged structure factor.

    Uses the Wiener–Khinchin theorem: the autocorrelation is the inverse
    Fourier transform of the power spectrum.

    Parameters
    ----------
    S_mean : ndarray
        Shifted, time-averaged 2-D power spectrum.

    Returns
    -------
    ndarray
        Normalised (peak = 1) 2-D autocorrelation map.
    """
    C = np.fft.ifft2(np.fft.ifftshift(S_mean)).real
    C = np.fft.fftshift(C)
    C /= np.max(C)
    return C


# ══════════════════════════════════════════════════════════════════════════════
# Correlation length
# ══════════════════════════════════════════════════════════════════════════════

def correlation_length_from_autocorr(radial_C: np.ndarray,
                                      threshold: float = np.exp(-1)) -> float:
    """
    Estimate the correlation length as the 1/e radius of the autocorrelation.

    Parameters
    ----------
    radial_C : ndarray
        Radially averaged autocorrelation (normalised so that C(0) = 1).
    threshold : float
        Decay threshold (default ``1/e ≈ 0.368``).

    Returns
    -------
    float
        Radius (in pixels) at which C first falls below *threshold*,
        or ``nan`` if it never does.
    """
    for r in range(1, len(radial_C)):
        if radial_C[r] <= threshold:
            return float(r)
    return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Per-video processing
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlation_length(filepath: str,
                                binarization: str = "otsu",
                                visualize_intermediate: bool = False
                                ) -> tuple[float, np.ndarray | None]:
    """
    Compute the spatial correlation length for a single video.

    The 2-D power spectrum is accumulated over all frames, then converted to
    a spatial autocorrelation from which the 1/e correlation length ξ is
    extracted.  A Hanning window is applied to each frame to suppress
    spectral leakage.

    Parameters
    ----------
    filepath : str
        Path to the .mp4 file.
    binarization : {"otsu", "fixed"}
        Binarization strategy applied to each frame.
    visualize_intermediate : bool
        If True, display the structure factor, autocorrelation map, and
        radial decay curve for this file.

    Returns
    -------
    xi : float
        Correlation length in pixels (``nan`` if no frames could be read).
    radial_C : ndarray or None
        Radially averaged autocorrelation profile.
    """
    cap = cv2.VideoCapture(filepath)
    S_accum = None
    n_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = frame_to_binary(gray, method=binarization)

        f = binary.astype(float)
        f -= f.mean()
        window = np.outer(np.hanning(f.shape[0]), np.hanning(f.shape[1]))
        f *= window

        S = np.abs(np.fft.fft2(f)) ** 2
        S_accum = S if S_accum is None else S_accum + S
        n_frames += 1

    cap.release()

    if n_frames == 0:
        return np.nan, None

    S_mean = np.fft.fftshift(S_accum / n_frames)
    anisotropy = compute_anisotropy_metric(S_mean)
    C = compute_autocorrelation_from_S(S_mean)
    radial_C = radial_average_2d(C)
    xi = correlation_length_from_autocorr(radial_C)

    fname = os.path.basename(filepath)
    print(f"  {fname}  |  anisotropy={anisotropy:.3f}  |  ξ={xi:.2f} px")

    if visualize_intermediate:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].imshow(np.log1p(S_mean), cmap="inferno")
        axes[0].set_title(f"2D Structure Factor\nanisotropy={anisotropy:.3f}")

        axes[1].imshow(C, cmap="viridis")
        axes[1].set_title("Spatial Autocorrelation")

        r = np.arange(len(radial_C))
        axes[2].plot(r, radial_C, "k")
        axes[2].axhline(np.exp(-1), color="r", linestyle="--", label="1/e")
        axes[2].axvline(xi, color="g", linestyle="--", label=f"ξ={xi:.1f}")
        axes[2].set_xlabel("r (pixels)")
        axes[2].set_ylabel("C(r)")
        axes[2].set_title("Autocorrelation decay")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        for ax in axes[:2]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return xi, radial_C


# ══════════════════════════════════════════════════════════════════════════════
# Batch pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(folder: str,
                 param_prefix: str = "strength",
                 seeds: list[int] | None = None,
                 params_to_skip: list[float] | None = None,
                 binarization: str = "otsu",
                 check_binarization: bool = False,
                 visualize_intermediate: bool = False,
                 debug: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run the full correlation-length pipeline over a folder of videos.

    Videos are grouped by their control-parameter value (extracted from the
    filename using *param_prefix*).  For each parameter value the mean and
    standard deviation of ξ across seeds are computed.  A publication-style
    summary plot is produced at the end.

    Parameters
    ----------
    folder : str
        Directory containing the .mp4 clips (typically a ``last{N}sec/``
        subfolder produced by ``extract_video_snippets.py``).
    param_prefix : str
        Token that precedes the numeric value in the filename.
        Examples: ``"strength"``, ``"threshold"``, ``"stimulus_strength"``.
    seeds : list of int or None
        If given, only files whose name contains ``seed_<s>`` for some
        ``s`` in *seeds* are included.  Pass ``None`` to include all files.
    params_to_skip : list of float or None
        Parameter values to exclude from processing.
    binarization : {"otsu", "fixed"}
        Binarization strategy forwarded to ``compute_correlation_length``.
    check_binarization : bool
        If True, display a binarization quality check for the first video
        of each parameter value.
    visualize_intermediate : bool
        Forward to ``compute_correlation_length`` for per-video diagnostics.
    debug : bool
        If True, print per-seed ξ values.

    Returns
    -------
    sorted_params : ndarray
        Sorted unique parameter values.
    xi_means : ndarray
        Mean correlation length for each parameter value.
    xi_by_param : dict
        Full result dict: ``{param_value: {"mean": float, "std": float}}``.
    """
    params_to_skip = set(params_to_skip or [])

    # ── collect and group files ───────────────────────────────────────────────
    all_videos = [f for f in os.listdir(folder) if f.lower().endswith(".mp4")]
    videos_by_param: dict[float, list[str]] = {}

    for fname in all_videos:
        # optional seed filter
        if seeds is not None:
            if not any(f"seed_{s}" in fname for s in seeds):
                continue

        try:
            param_val = extract_control_param(fname, param_prefix)
        except ValueError:
            print(f"[SKIP] Could not extract '{param_prefix}' from: {fname}")
            continue

        if param_val in params_to_skip:
            continue

        videos_by_param.setdefault(param_val, []).append(
            os.path.join(folder, fname)
        )

    sorted_params = sorted(videos_by_param.keys())
    print(f"Found {param_prefix} values: {sorted_params}\n")

    # ── process ───────────────────────────────────────────────────────────────
    xi_by_param: dict[float, dict] = {}

    for param_val in sorted_params:
        vids = videos_by_param[param_val]
        print(f"── {param_prefix} = {param_val}  ({len(vids)} videos) ──")

        if check_binarization and vids:
            check_binarization_quality(vids[0], method=binarization,
                                       param=param_prefix)

        xis = []
        for vpath in vids:
            try:
                xi, _ = compute_correlation_length(
                    vpath,
                    binarization=binarization,
                    visualize_intermediate=visualize_intermediate,
                )
                xis.append(xi)
                if debug:
                    print(f"    {os.path.basename(vpath)}  →  ξ = {xi:.3f}")
            except Exception as exc:
                print(f"[ERROR] {os.path.basename(vpath)}: {exc}")
                xis.append(np.nan)

        xis = np.array(xis)
        xi_by_param[param_val] = {
            "mean": float(np.nanmean(xis)) if np.any(~np.isnan(xis)) else np.nan,
            "std":  float(np.nanstd(xis))  if np.any(~np.isnan(xis)) else np.nan,
        }
        print(f"    → mean ξ = {xi_by_param[param_val]['mean']:.2f}")

    # ── summary plot ──────────────────────────────────────────────────────────
    _plot_xi_vs_param(sorted_params, xi_by_param, param_prefix)

    xi_means = np.array([xi_by_param[p]["mean"] for p in sorted_params])
    return np.array(sorted_params), xi_means, xi_by_param


def _plot_xi_vs_param(sorted_params: list[float],
                       xi_by_param: dict,
                       param_prefix: str) -> None:
    """Publication-style plot of ξ vs control parameter."""
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "axes.linewidth": 1.2, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.25, "grid.linestyle": "--",
    })

    x = np.array(sorted_params)
    y = np.array([xi_by_param[p]["mean"] for p in sorted_params])
    yerr = np.array([xi_by_param[p]["std"]  for p in sorted_params])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color="tab:blue", marker="o", markersize=7,
            linewidth=2.5, label="Mean ξ")
    ax.fill_between(x, y - yerr, y + yerr, color="tab:blue",
                    alpha=0.2, label="±1 std")

    imax = np.nanargmax(y)
    ax.axvline(x[imax], linestyle="--", color="black", linewidth=1.5, alpha=0.4)
    ax.scatter(x[imax], y[imax], color="red", s=70, zorder=5)
    ax.annotate(
        rf"$\mathrm{{{param_prefix}}}^c = {x[imax]:.3g}$",
        xy=(x[imax], y[imax]),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=12,
    )

    ax.set_xlabel(f"Control parameter  ({param_prefix})")
    ax.set_ylabel("Correlation length ξ (pixels)")
    ax.set_title(f"Correlation Length vs {param_prefix}")
    ax.set_xlim(0)
    ax.grid(True)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
