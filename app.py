"""
CSE 438: Digital Image Processing — Classical Restoration System
Department of Computer Science & Engineering
Course Project Demonstration Interface
"""

from pathlib import Path
import io
import time
from typing import Optional, Tuple, List, Dict, Any, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Custom classical image processing modules
from src.image_io import load_image, save_image
from src.utils.image_helpers import to_float, to_uint8, rgb_to_gray, ensure_grayscale
from src.utils.validation import validate_image, get_image_info
from src.degradation.gaussian_noise import add_gaussian_noise
from src.degradation.salt_pepper import add_salt_and_pepper_noise
from src.spatial.gaussian_filter import create_gaussian_kernel, apply_gaussian_filter
from src.degradation.motion_blur import create_motion_blur_kernel, apply_motion_blur
from src.spatial.mean_filter import apply_mean_filter
from src.spatial.median_filter import apply_median_filter
from src.frequency.dft_visualization import compute_dft2, compute_magnitude_spectrum
from src.frequency.psf import psf_to_otf
from src.frequency.inverse_filter import apply_inverse_filter
from src.frequency.wiener_filter import apply_wiener_filter
from src.evaluation.psnr import compute_psnr, compute_mse
from src.evaluation.ssim import compute_custom_ssim, compute_reference_ssim
from src.evaluation.runtime import time_function

# -----------------------------------------------------------------------------
# Configuration & Professional Academic Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CSE 438 — Image Restoration System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DATA_DIR = BASE_DIR / "data" / "original"
REAL_PHOTOS_DIR = BASE_DIR / "data" / "real_photos"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

# Professional, clean editorial styling (no neon gradients, no AI clichés)
st.markdown(
    """
    <style>
    /* Typography & Core Layout */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    
    /* Academic Header */
    .dept-header {
        border-bottom: 2px solid #27272a;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .dept-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fafafa;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .dept-subtitle {
        font-size: 0.95rem;
        color: #a1a1aa;
        margin-top: 0.3rem;
    }
    
    /* Clean Metric Boxes */
    div[data-testid="stMetric"] {
        background-color: #18181b;
        border: 1px solid #27272a;
        padding: 0.75rem 1rem;
        border-radius: 6px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #a1a1aa;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Subtle File Uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed #3f3f46;
        border-radius: 8px;
        padding: 0.75rem;
        background: #18181b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def array_to_bytes(img: np.ndarray) -> bytes:
    """Encode numpy image array to PNG bytes for downloading."""
    u8 = to_uint8(img)
    if u8.ndim == 3 and u8.shape[2] == 3:
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    else:
        bgr = u8
    success, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if success else b""


def decode_uploaded_file(uploaded_file, as_gray: bool = False) -> Optional[np.ndarray]:
    """Decode an uploaded file, preserving native RGB colors by default."""
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    if as_gray:
        return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_samples() -> Dict[str, Path]:
    if not ORIGINAL_DATA_DIR.exists():
        return {}
    return {p.name: p for p in sorted(ORIGINAL_DATA_DIR.glob("*.*")) if p.suffix.lower() in [".png", ".jpg", ".jpeg"]}


def get_real_photos() -> Dict[str, Path]:
    if not REAL_PHOTOS_DIR.exists():
        return {}
    return {p.name: p for p in sorted(REAL_PHOTOS_DIR.glob("*.*")) if p.suffix.lower() in [".png", ".jpg", ".jpeg"]}


# -----------------------------------------------------------------------------
# Department Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="dept-header">
        <h1 class="dept-title">CSE 438: Image Restoration & Deblurring System</h1>
        <p class="dept-subtitle">Classical Digital Image Processing Laboratory • Pure NumPy Implementation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Navigation Tabs (Simple, Clean, Logical)
# -----------------------------------------------------------------------------
tab_studio, tab_compare, tab_real, tab_benchmarks = st.tabs([
    "Interactive Restoration Studio",
    "Compare Restored Images (Drag & Drop)",
    "Real Historical Photographs (Track B)",
    "Benchmark Experiments & Spectral Analysis",
])


# =============================================================================
# TAB 1: Interactive Restoration Studio
# =============================================================================
with tab_studio:
    samples = get_samples()

    st.markdown("##### 1. Source Image Selection")
    col_upload, col_sample = st.columns([1.5, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload image (Drag & Drop PNG, JPG, BMP):",
            type=["png", "jpg", "jpeg", "bmp"],
            key="studio_upload",
        )
    with col_sample:
        sample_pick = st.selectbox(
            "Or select from benchmark dataset:",
            ["-- Use Uploaded Image --"] + list(samples.keys()),
        )
        as_gray = st.checkbox("Convert to single-channel Grayscale", value=False)

    if uploaded_file is not None:
        source_img = decode_uploaded_file(uploaded_file, as_gray=as_gray)
        src_label = uploaded_file.name
    elif sample_pick != "-- Use Uploaded Image --":
        source_img = load_image(samples[sample_pick], as_gray=as_gray)
        src_label = sample_pick
    else:
        default_file = "checkerboard.png" if "checkerboard.png" in samples else list(samples.keys())[0]
        source_img = load_image(samples[default_file], as_gray=as_gray)
        src_label = f"{default_file} (Default)"

    st.markdown("---")
    st.markdown("##### 2. Degradation & Restoration Pipeline")

    col_setup, col_exec = st.columns([1, 1.2])

    with col_setup:
        st.markdown("**A. Degradation Model (Simulated Distortion)**")
        deg_mode = st.selectbox(
            "Select Degradation:",
            [
                "None (Restore image as-is / already degraded upload)",
                "Salt-and-Pepper Impulse Noise",
                "Additive Gaussian Noise",
                "Directional Motion Blur",
                "Gaussian Optical Defocus Blur",
            ],
        )

        # Apply degradation
        if deg_mode == "Salt-and-Pepper Impulse Noise":
            d_dens = st.slider("Impulse Noise Density:", 0.01, 0.30, 0.08, step=0.01)
            degraded_img = add_salt_and_pepper_noise(source_img, density=d_dens, seed=42)
            deg_desc = f"Salt-and-Pepper Noise (Density: {int(d_dens*100)}%)"

        elif deg_mode == "Additive Gaussian Noise":
            d_sig = st.slider("Gaussian Noise Sigma (σ):", 0.01, 0.25, 0.08, step=0.01)
            degraded_img = add_gaussian_noise(source_img, sigma=d_sig, seed=42)
            deg_desc = f"Gaussian Noise (σ = {d_sig})"

        elif deg_mode == "Directional Motion Blur":
            m_len = st.slider("Motion Length (pixels):", 3, 29, 11, step=2)
            m_ang = st.slider("Motion Angle (degrees):", 0.0, 180.0, 45.0, step=5.0)
            psf_motion = create_motion_blur_kernel(length=m_len, angle=m_ang)
            blurred = apply_motion_blur(source_img, length=m_len, angle=m_ang)
            add_noise = st.checkbox("Include Sensor Noise (σ=0.01)", value=True)
            degraded_img = add_gaussian_noise(blurred, sigma=0.01, seed=42) if add_noise else blurred
            deg_desc = f"Motion Blur (L={m_len}px, {int(m_ang)}°)"

        elif deg_mode == "Gaussian Optical Defocus Blur":
            g_sz = st.select_slider("Kernel Size:", options=[3, 5, 7, 9, 15], value=7)
            g_sg = st.slider("Sigma (Spread):", 0.5, 3.5, 1.5, step=0.1)
            psf_gauss = create_gaussian_kernel(kernel_size=g_sz, sigma=g_sg)
            blurred = apply_gaussian_blur(source_img, kernel_size=g_sz, sigma=g_sg)
            add_noise = st.checkbox("Include Sensor Noise (σ=0.01)", value=True, key="gb_noise")
            degraded_img = add_gaussian_noise(blurred, sigma=0.01, seed=42) if add_noise else blurred
            deg_desc = f"Gaussian Blur (K={g_sz}, σ={g_sg})"

        else:
            degraded_img = source_img.copy()
            deg_desc = "No Artificial Degradation"

        st.markdown("**B. Classical Restoration Algorithm**")
        algo_choice = st.selectbox(
            "Select Restoration Filter:",
            [
                "Median Filter (Non-linear Order Statistic)",
                "Wiener Filter (Minimum Mean Square Error Deblur)",
                "Gaussian Smoothing Filter",
                "Mean (Box) Filter",
                "Direct Inverse Filter (Frequency Domain)",
            ],
        )

        # Algorithm parameters
        if "Median" in algo_choice:
            k_val = st.select_slider("Neighborhood Window Size:", options=[3, 5, 7, 9], value=3)
            restored_img, elapsed_ms = time_function(apply_median_filter, degraded_img, kernel_size=k_val)

        elif "Wiener" in algo_choice:
            w_k = st.select_slider("Noise-to-Signal Ratio (K):", options=[0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2], value=0.01)
            # Use matched PSF if motion or blur was selected; otherwise default motion PSF
            if deg_mode == "Directional Motion Blur":
                psf_use = psf_motion
            elif deg_mode == "Gaussian Optical Defocus Blur":
                psf_use = psf_gauss
            else:
                psf_use = create_motion_blur_kernel(length=11, angle=45.0)
            restored_img, elapsed_ms = time_function(apply_wiener_filter, degraded_img, psf_use, K=w_k)

        elif "Gaussian" in algo_choice:
            gk_val = st.select_slider("Gaussian Kernel Size:", options=[3, 5, 7, 9], value=5)
            gs_val = st.slider("Filter Sigma:", 0.5, 3.5, 1.0, step=0.1)
            restored_img, elapsed_ms = time_function(apply_gaussian_filter, degraded_img, kernel_size=gk_val, sigma=gs_val)

        elif "Mean" in algo_choice:
            mk_val = st.select_slider("Averaging Kernel Size:", options=[3, 5, 7, 9], value=3)
            restored_img, elapsed_ms = time_function(apply_mean_filter, degraded_img, kernel_size=mk_val)

        else:  # Direct Inverse Filter
            st.caption("Theoretical test: Inverse filtering divides frequency response by H(u, v).")
            psf_use = psf_motion if deg_mode == "Directional Motion Blur" else create_gaussian_kernel(kernel_size=5, sigma=1.0)
            restored_img, elapsed_ms = time_function(apply_inverse_filter, degraded_img, psf_use, threshold=None)

    with col_exec:
        st.markdown("**C. Quantitative Quality Evaluation**")
        psnr_deg = compute_psnr(source_img, degraded_img)
        psnr_rest = compute_psnr(source_img, restored_img)
        delta_psnr = psnr_rest - psnr_deg

        ssim_deg = compute_custom_ssim(source_img, degraded_img)
        ssim_rest = compute_custom_ssim(source_img, restored_img)
        delta_ssim = ssim_rest - ssim_deg

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Restored PSNR", f"{psnr_rest:.2f} dB", delta=f"{delta_psnr:+.2f} dB" if deg_mode != "None (Restore image as-is / already degraded upload)" else None)
        m_col2.metric("Restored SSIM", f"{ssim_rest:.4f}", delta=f"{delta_ssim:+.4f}" if deg_mode != "None (Restore image as-is / already degraded upload)" else None)
        m_col3.metric("CPU Latency", f"{elapsed_ms:.2f} ms")

        # Visual Display
        disp_col1, disp_col2 = st.columns(2)
        with disp_col1:
            st.caption(f"Input / Degraded: {deg_desc}")
            st.image(degraded_img, use_container_width=True)
        with disp_col2:
            st.caption(f"Output / Restored: {algo_choice.split('(')[0].strip()}")
            st.image(restored_img, use_container_width=True)

        st.download_button(
            "Download Restored Image (.PNG)",
            data=array_to_bytes(restored_img),
            file_name=f"restored_{src_label}.png",
            mime="image/png",
            use_container_width=True,
        )


# =============================================================================
# TAB 2: Compare Restored Images (Dual Drag & Drop)
# =============================================================================
with tab_compare:
    st.markdown("##### Independent Restoration Comparison & Quality Diagnostics")
    st.caption("Drag and drop your clean reference image and your restored image to compute exact mathematical error metrics and view the difference heatmap.")

    c_file1, c_file2 = st.columns(2)
    with c_file1:
        st.markdown("**Reference Clean Ground Truth (Image A)**")
        file_a = st.file_uploader("Drop Reference Image (Clean):", type=["png", "jpg", "jpeg", "bmp"], key="cmp_a")
    with c_file2:
        st.markdown("**Your Restored Output (Image B)**")
        file_b = st.file_uploader("Drop Restored Image:", type=["png", "jpg", "jpeg", "bmp"], key="cmp_b")

    if file_a is not None and file_b is not None:
        img_a = decode_uploaded_file(file_a, as_gray=False)
        img_b = decode_uploaded_file(file_b, as_gray=False)
        name_a, name_b = file_a.name, file_b.name
    else:
        st.info("Showing default comparison pair. Drag & drop your files above to evaluate your own images.")
        samples = get_samples()
        demo_name = "color_test.jpg" if "color_test.jpg" in samples else list(samples.keys())[0]
        img_a = load_image(samples[demo_name], as_gray=False)
        img_b = apply_median_filter(add_salt_and_pepper_noise(img_a, density=0.08, seed=42), kernel_size=3)
        name_a, name_b = f"{demo_name} (Reference)", "Median Filter Restored Demo"

    # Channel match check
    if img_a.ndim == 2 and img_b.ndim == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)
    elif img_a.ndim == 3 and img_b.ndim == 2:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)

    # Dimension match check
    if img_a.shape[:2] != img_b.shape[:2]:
        st.warning(f"Spatial resolution mismatch: Image A is {img_a.shape[:2]}, but Image B is {img_b.shape[:2]}. Automatically aligning Image B for calculation.")
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    # Compute metrics
    val_mse = compute_mse(img_a, img_b)
    val_psnr = compute_psnr(img_a, img_b)
    val_ssim = compute_custom_ssim(img_a, img_b)
    val_ref_ssim = compute_reference_ssim(img_a, img_b)

    # Metrics row
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("PSNR (dB)", f"{val_psnr:.2f} dB" if np.isfinite(val_psnr) else "∞ (Identical)")
    sc2.metric("Custom SSIM", f"{val_ssim:.4f}")
    sc3.metric("Reference SSIM (skimage)", f"{val_ref_ssim:.4f}")
    sc4.metric("Mean Squared Error (MSE)", f"{val_mse:.2f}")

    # Side-by-side images + heatmap
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.caption(f"Image A: {name_a}")
        st.image(img_a, use_container_width=True)
    with v_col2:
        st.caption(f"Image B: {name_b}")
        st.image(img_b, use_container_width=True)
    with v_col3:
        st.caption("Absolute Error Heatmap (|Image A - Image B|)")
        err = np.abs(img_a.astype(float) - img_b.astype(float))
        err_2d = np.mean(err, axis=-1) if err.ndim == 3 else err

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(err_2d, cmap="magma", vmin=0, vmax=max(40.0, float(np.max(err_2d))))
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)


# =============================================================================
# TAB 3: Real Historical Photographs (Track B)
# =============================================================================
with tab_real:
    st.markdown("##### Track B: Unpaired Real & Vintage Photographs")
    st.caption(
        "Evaluation on authentic historical photography without synthetic ground truth. "
        "In accordance with academic standards, no artificial PSNR/SSIM numbers are reported for unpaired images."
    )

    photos = get_real_photos()

    c_sel1, c_sel2 = st.columns([1, 1])
    with c_sel1:
        photo_choice = st.selectbox("Select Historical Photograph:", list(photos.keys()))
        raw_photo = load_image(photos[photo_choice], as_gray=True)

    with c_sel2:
        v_filter = st.selectbox(
            "Select Appropriate Classical Restoration:",
            [
                "Median Filter 3×3 (Dust, Speckles & Film Grain)",
                "Wiener Filter (Lens Defocus & Softness Deblur)",
                "Gaussian Filter (Continuous Emulsion Grain Smoothing)",
            ],
        )

    if "Median" in v_filter:
        restored_photo, t_photo = time_function(apply_median_filter, raw_photo, kernel_size=3)
        improvement = "Suppressed photographic emulsion grain and sensor dust spots while preserving character and edge boundaries."
        tradeoff = "Very fine sub-pixel micro-scratches are slightly smoothed."
    elif "Wiener" in v_filter:
        psf_v = create_gaussian_kernel(kernel_size=5, sigma=1.0)
        restored_photo, t_photo = time_function(apply_wiener_filter, raw_photo, psf_v, K=0.03)
        improvement = "Sharpens lens softness and enhances facial and geometric contours."
        tradeoff = "May introduce subtle ringing if blur kernel estimate deviates from physical optics."
    else:
        restored_photo, t_photo = time_function(apply_gaussian_filter, raw_photo, kernel_size=3, sigma=0.8)
        improvement = "Suppresses high-frequency photographic grain."
        tradeoff = "Slightly softens high-contrast edges."

    col_view_orig, col_view_rest = st.columns(2)
    with col_view_orig:
        st.caption(f"Archival Original: {photo_choice}")
        st.image(raw_photo, use_container_width=True)
    with col_view_rest:
        st.caption(f"Restored ({v_filter.split('(')[0].strip()}) — Latency: {t_photo:.2f} ms")
        st.image(restored_photo, use_container_width=True)

    st.markdown(f"**Observed Improvement:** {improvement}")
    st.markdown(f"**Physical Trade-off / Remaining Artifacts:** {tradeoff}")


# =============================================================================
# TAB 4: Benchmark Experiments & Spectral Analysis
# =============================================================================
with tab_benchmarks:
    st.markdown("##### Quantitative Benchmarking & Frequency Domain Analysis")

    subtab_bench, subtab_fft = st.tabs(["Track A Quantitative Benchmark Data", "2D Discrete Fourier Transform Analysis"])

    with subtab_bench:
        csv_file = RESULTS_DIR / "quantitative_experiments.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            st.caption(f"Displaying {len(df)} systematic experimental benchmark runs across Gaussian noise, Salt & Pepper noise, and Motion blur:")
            st.dataframe(df, use_container_width=True)

            summary_img = RESULTS_DIR / "track_a_visual_summary.png"
            if summary_img.exists():
                st.image(str(summary_img), caption="Track A Controlled Benchmark Visual Summary", use_container_width=True)
        else:
            st.warning("Benchmark CSV not found. Run experiments/run_experiments.py to generate data.")

    with subtab_fft:
        samples = get_samples()
        fft_pick = st.selectbox("Select Image for Spectral Analysis:", list(samples.keys()), key="fft_sel")
        fft_img = load_image(samples[fft_pick], as_gray=True)

        _, f_shift = compute_dft2(fft_img)
        mag_spec = compute_magnitude_spectrum(f_shift, log_scale=True)

        c_fft1, c_fft2 = st.columns(2)
        with c_fft1:
            st.caption("Spatial Domain f(x, y)")
            st.image(fft_img, use_container_width=True)
        with c_fft2:
            st.caption("2D DFT Magnitude Spectrum log(1 + |F(u, v)|)")
            st.image(mag_spec, use_container_width=True)
